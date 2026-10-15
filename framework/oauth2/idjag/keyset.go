package idjag

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/singleflight"
)

// Key-set refresh policy: keys are reused for keyTTL, and an unknown kid can
// force a refetch at most once per refetchCooldown so a stream of forged kids
// cannot turn the token endpoint into an amplifier against the IdP.
const (
	keyTTL          = time.Hour
	refetchCooldown = 30 * time.Second
	// failedFetchCooldown spaces retries while no keys are cached, so an
	// unreachable IdP is not refetched on every request yet recovers quickly.
	failedFetchCooldown = 5 * time.Second
	maxDocBytes         = 1 << 20
)

// keySet caches one issuer's signing keys. The mutex guards only the cached
// state; fetches run outside it, collapsed by singleflight, so a slow IdP never
// blocks verifications that the cached keys can already answer.
type keySet struct {
	issuer  string
	jwksURL string
	client  func() *http.Client
	fetches singleflight.Group

	mu          sync.RWMutex
	keys        map[string]crypto.PublicKey
	fetchedAt   time.Time
	lastAttempt time.Time
	lastErr     error
}

// newKeySet returns an empty cache; jwksURL may be empty to use discovery.
func newKeySet(issuer, jwksURL string, client func() *http.Client) *keySet {
	return &keySet{issuer: issuer, jwksURL: jwksURL, client: client}
}

// key returns the verification key for kid, refreshing the set when it is stale
// or the kid is unknown. An empty kid matches only a single-key set.
func (k *keySet) key(ctx context.Context, kid string) (crypto.PublicKey, error) {
	if pub, ok, fresh := k.lookup(kid); ok && fresh {
		return pub, nil
	}
	// The cooldown is checked inside the shared fetch, so concurrent callers still
	// join a fetch in flight while sequential ones inside the cooldown skip it.
	_, _, _ = k.fetches.Do("jwks", func() (any, error) {
		if !k.claimRefetch() {
			return nil, nil
		}
		// Detached from the caller: every joined caller shares this result, and a
		// failure starts the cooldown. The HTTP client's timeout still bounds it.
		keys, err := k.fetch(context.WithoutCancel(ctx))
		k.mu.Lock()
		defer k.mu.Unlock()
		if err == nil {
			k.keys, k.fetchedAt, k.lastErr = keys, time.Now(), nil
		} else {
			k.lastErr = err
		}
		return nil, err
	})
	if pub, ok, _ := k.lookup(kid); ok {
		return pub, nil
	}
	k.mu.RLock()
	lastErr := k.lastErr
	k.mu.RUnlock()
	if lastErr != nil {
		return nil, fmt.Errorf("%w: kid %q: %v", ErrKey, kid, lastErr)
	}
	return nil, fmt.Errorf("%w: kid %q", ErrKey, kid)
}

// lookup finds kid in the cached set and reports whether the set is fresh.
func (k *keySet) lookup(kid string) (pub crypto.PublicKey, ok, fresh bool) {
	k.mu.RLock()
	defer k.mu.RUnlock()
	fresh = !k.fetchedAt.IsZero() && time.Since(k.fetchedAt) < keyTTL
	if kid == "" && len(k.keys) == 1 {
		for _, p := range k.keys {
			return p, true, fresh
		}
	}
	pub, ok = k.keys[kid]
	return pub, ok, fresh
}

// claimRefetch reports whether a fetch may run now: at most once per
// refetchCooldown while keys are cached, and once per failedFetchCooldown while
// none are, so a down IdP is not hit on every request.
func (k *keySet) claimRefetch() bool {
	k.mu.Lock()
	defer k.mu.Unlock()
	cooldown := refetchCooldown
	if k.keys == nil {
		cooldown = failedFetchCooldown
	}
	if !k.lastAttempt.IsZero() && time.Since(k.lastAttempt) < cooldown {
		return false
	}
	k.lastAttempt = time.Now()
	return true
}

// fetch downloads and parses the JWKS, discovering its URL first when needed.
func (k *keySet) fetch(ctx context.Context) (map[string]crypto.PublicKey, error) {
	jwksURL := k.jwksURL
	if jwksURL == "" {
		discovered, err := k.discoverJWKSURL(ctx)
		if err != nil {
			return nil, err
		}
		jwksURL = discovered
	}
	var doc struct {
		Keys []jwk `json:"keys"`
	}
	if err := k.getJSON(ctx, jwksURL, &doc); err != nil {
		return nil, fmt.Errorf("fetch jwks: %w", err)
	}
	keys := make(map[string]crypto.PublicKey, len(doc.Keys))
	for _, j := range doc.Keys {
		if j.Use != "" && j.Use != "sig" {
			continue
		}
		if pub, err := j.publicKey(); err == nil {
			keys[j.Kid] = pub
		}
	}
	if len(keys) == 0 {
		return nil, fmt.Errorf("jwks at %s holds no usable signing keys", jwksURL)
	}
	return keys, nil
}

// discoverJWKSURL reads jwks_uri from the issuer's OIDC or RFC 8414 metadata and
// requires the document to name the same issuer, so a redirecting proxy cannot
// substitute another IdP's keys.
func (k *keySet) discoverJWKSURL(ctx context.Context) (string, error) {
	base := strings.TrimRight(k.issuer, "/")
	var lastErr error
	for _, path := range []string{"/.well-known/openid-configuration", "/.well-known/oauth-authorization-server"} {
		var meta struct {
			Issuer  string `json:"issuer"`
			JWKSURI string `json:"jwks_uri"`
		}
		if err := k.getJSON(ctx, base+path, &meta); err != nil {
			lastErr = err
			continue
		}
		if !sameURL(meta.Issuer, k.issuer) || meta.JWKSURI == "" {
			lastErr = fmt.Errorf("metadata at %s names issuer %q", base+path, meta.Issuer)
			continue
		}
		return meta.JWKSURI, nil
	}
	return "", fmt.Errorf("discover jwks_uri for %s: %w", k.issuer, lastErr)
}

// getJSON GETs url and decodes a bounded JSON body into out.
func (k *keySet) getJSON(ctx context.Context, url string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")
	resp, err := k.client().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: status %d", url, resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxDocBytes))
	if err != nil {
		return err
	}
	return json.Unmarshal(body, out)
}

// jwk is the subset of RFC 7517 fields needed for RSA and EC verification keys.
type jwk struct {
	Kty string `json:"kty"`
	Kid string `json:"kid"`
	Use string `json:"use"`
	N   string `json:"n"`
	E   string `json:"e"`
	Crv string `json:"crv"`
	X   string `json:"x"`
	Y   string `json:"y"`
}

// publicKey converts the JWK into a crypto public key.
func (j jwk) publicKey() (crypto.PublicKey, error) {
	switch j.Kty {
	case "RSA":
		n, err := decodeBigInt(j.N)
		if err != nil {
			return nil, err
		}
		e, err := decodeBigInt(j.E)
		if err != nil {
			return nil, err
		}
		if !e.IsInt64() || e.Int64() > 1<<31-1 {
			return nil, fmt.Errorf("rsa exponent out of range")
		}
		return &rsa.PublicKey{N: n, E: int(e.Int64())}, nil
	case "EC":
		var curve elliptic.Curve
		switch j.Crv {
		case "P-256":
			curve = elliptic.P256()
		case "P-384":
			curve = elliptic.P384()
		case "P-521":
			curve = elliptic.P521()
		default:
			return nil, fmt.Errorf("unsupported curve %q", j.Crv)
		}
		// RFC 7518 §6.2.1.2: x and y are exactly the curve's coordinate size, so
		// they are checked raw rather than through big.Int, which drops leading
		// zeros. ParseUncompressedPublicKey rejects points off the curve.
		size := (curve.Params().BitSize + 7) / 8
		x, errX := base64.RawURLEncoding.DecodeString(j.X)
		y, errY := base64.RawURLEncoding.DecodeString(j.Y)
		if errX != nil || errY != nil || len(x) != size || len(y) != size {
			return nil, fmt.Errorf("ec coordinates must be %d bytes each", size)
		}
		point := make([]byte, 0, 1+2*size)
		point = append(append(append(point, 4), x...), y...)
		pub, err := ecdsa.ParseUncompressedPublicKey(curve, point)
		if err != nil {
			return nil, fmt.Errorf("invalid ec public key: %w", err)
		}
		return pub, nil
	}
	return nil, fmt.Errorf("unsupported key type %q", j.Kty)
}

// decodeBigInt decodes an unpadded base64url big-endian integer.
func decodeBigInt(s string) (*big.Int, error) {
	b, err := base64.RawURLEncoding.DecodeString(s)
	if err != nil || len(b) == 0 {
		return nil, fmt.Errorf("invalid jwk integer")
	}
	return new(big.Int).SetBytes(b), nil
}
