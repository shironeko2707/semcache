// Package semcache is a correctness-aware semantic cache for LLM responses,
// built for regulated workloads where a wrong cache hit is a defect, not just a
// missed optimization. It pairs the usual embedding-similarity lookup with a
// similarity floor, a pluggable second-stage verifier, PII-aware keying,
// deterministic staleness via knowledge epochs, and a negative cache for
// near-misses. The default path is zero-dependency and in-memory.
package semcache

import (
	"context"
	"errors"
	"time"

	"github.com/bbbb/semcache/store"
)

// ErrNoEmbedder is returned by New when no Embedder is supplied.
var ErrNoEmbedder = errors.New("semcache: an Embedder is required")

// Query is a lookup or store request. Text is the raw user text; it is redacted
// and normalized internally before it is ever embedded, keyed, or stored.
type Query struct {
	Text      string            // raw input; redacted internally before keying
	Namespace string            // tenant/collection isolation
	Epoch     string            // knowledge epoch; mismatch => miss when enforced
	Meta      map[string]string // filters that must match for a hit when enforced
}

// Entry is a cached model response plus light metadata.
type Entry struct {
	Response  string
	Meta      map[string]string
	CreatedAt time.Time
	Epoch     string
}

// Cache is the public contract. Lookup returns a verified, non-stale,
// above-floor match or found=false (the caller should then call the model and
// Store the result).
type Cache interface {
	Lookup(ctx context.Context, q Query) (Entry, bool, error)
	Store(ctx context.Context, q Query, resp Entry) error
	Stats() Stats
}

// SemanticCache is the default Cache implementation.
type SemanticCache struct {
	embed    Embedder
	store    store.Store
	redactor *Redactor
	verifier Verifier
	policy   Policy
	metrics  *Metrics
	now      func() time.Time
}

// Option configures a SemanticCache.
type Option func(*SemanticCache)

// WithStore sets the backend. Defaults to an in-memory store.
func WithStore(s store.Store) Option { return func(c *SemanticCache) { c.store = s } }

// WithPolicy sets the correctness policy. Defaults to DefaultPolicy.
func WithPolicy(p Policy) Option { return func(c *SemanticCache) { c.policy = p } }

// WithVerifier sets the second-stage verifier. Defaults to a LexicalVerifier.
func WithVerifier(v Verifier) Option { return func(c *SemanticCache) { c.verifier = v } }

// WithRedactor sets the PII redactor. Defaults to NewRedactor.
func WithRedactor(r *Redactor) Option { return func(c *SemanticCache) { c.redactor = r } }

// WithClock overrides the time source (tests). Defaults to time.Now.
func WithClock(now func() time.Time) Option { return func(c *SemanticCache) { c.now = now } }

// New builds a SemanticCache. An Embedder is required; everything else has a
// sensible default (in-memory store, default policy, lexical verifier, default
// redactor). The metrics-wired eviction hook is attached only to the default
// store; a caller-supplied store should wire WithEvictHook itself if it wants
// eviction counted.
func New(embed Embedder, opts ...Option) (*SemanticCache, error) {
	if embed == nil {
		return nil, ErrNoEmbedder
	}
	c := &SemanticCache{
		embed:    embed,
		redactor: NewRedactor(),
		verifier: LexicalVerifier{},
		policy:   DefaultPolicy(),
		metrics:  &Metrics{},
		now:      time.Now,
	}
	for _, o := range opts {
		o(c)
	}
	if c.store == nil {
		c.store = store.NewMemory(
			store.WithEvictHook(c.metrics.incEviction),
			store.WithClock(c.now), // share the cache's clock so TTL is consistent
		)
	}
	if c.policy.Candidates <= 0 {
		c.policy.Candidates = 1
	}
	return c, nil
}

// Lookup returns a cached response when a candidate clears the similarity floor,
// passes epoch/meta gating, is not a negative anchor, and passes verification.
// Otherwise found is false.
func (c *SemanticCache) Lookup(ctx context.Context, q Query) (Entry, bool, error) {
	canonical := c.redactor.Canonicalize(q.Text)
	vec, err := c.embed.Embed(ctx, canonical)
	if err != nil {
		return Entry{}, false, err
	}
	matches, err := c.store.Nearest(ctx, q.Namespace, vec, c.policy.Candidates)
	if err != nil {
		return Entry{}, false, err
	}

	for _, m := range matches {
		// Candidates are sorted by descending score; once we drop below the
		// floor, nothing further can qualify.
		if m.Score < c.policy.SimilarityFloor {
			break
		}
		rec := m.Record
		if c.policy.EnforceEpoch && rec.Epoch != q.Epoch {
			continue
		}
		if c.policy.EnforceMeta && !metaSubset(q.Meta, rec.Meta) {
			continue
		}
		// A negative anchor means a query this close was previously judged to be
		// a different question. Refuse to serve and stop — do not fall through to
		// a lower-scored positive that the negative was meant to shield.
		if rec.Negative {
			c.metrics.incMiss()
			return Entry{}, false, nil
		}
		ok, err := c.verifier.Verify(ctx, canonical, rec)
		if err != nil {
			return Entry{}, false, err
		}
		if !ok {
			// Cleared the floor but failed verification: a wrong answer refused.
			c.metrics.incFalseHitAvoided()
			if c.policy.NegativeCache {
				c.storeNegative(ctx, q, canonical, vec)
			}
			continue
		}
		entry, _ := rec.Payload.(Entry)
		c.metrics.incHit()
		return entry, true, nil
	}

	c.metrics.incMiss()
	return Entry{}, false, nil
}

// Store records a model response under the redacted+normalized form of the
// query. Exact repeats (same canonical text, namespace, epoch) overwrite.
func (c *SemanticCache) Store(ctx context.Context, q Query, resp Entry) error {
	canonical := c.redactor.Canonicalize(q.Text)
	vec, err := c.embed.Embed(ctx, canonical)
	if err != nil {
		return err
	}
	now := c.now()
	if resp.CreatedAt.IsZero() {
		resp.CreatedAt = now
	}
	if resp.Epoch == "" {
		resp.Epoch = q.Epoch
	}
	rec := store.Record{
		Key:       deriveKey(q.Namespace, q.Epoch, canonical),
		Namespace: q.Namespace,
		Epoch:     q.Epoch,
		Text:      canonical,
		Vector:    vec,
		Payload:   resp,
		Meta:      q.Meta,
		CreatedAt: now,
		ExpiresAt: c.expiry(now),
	}
	if err := c.store.Set(ctx, rec); err != nil {
		return err
	}
	c.metrics.incStore()
	return nil
}

// storeNegative records a near-miss anchor so future near-identical queries
// short-circuit to a miss instead of being re-verified. Best-effort: failures
// are swallowed (the lookup already returned the safe answer).
func (c *SemanticCache) storeNegative(ctx context.Context, q Query, canonical string, vec []float32) {
	now := c.now()
	rec := store.Record{
		Key:       "neg:" + deriveKey(q.Namespace, q.Epoch, canonical),
		Namespace: q.Namespace,
		Epoch:     q.Epoch,
		Text:      canonical,
		Vector:    vec,
		Meta:      q.Meta,
		CreatedAt: now,
		ExpiresAt: c.expiry(now),
		Negative:  true,
	}
	_ = c.store.Set(ctx, rec)
}

// expiry returns the absolute expiry time for an entry created at now, or the
// zero time when TTL is disabled.
func (c *SemanticCache) expiry(now time.Time) time.Time {
	if c.policy.TTL <= 0 {
		return time.Time{}
	}
	return now.Add(c.policy.TTL)
}

// Stats returns a snapshot of the cache counters and current store size.
func (c *SemanticCache) Stats() Stats {
	s := c.metrics.snapshot()
	s.Size = c.store.Len()
	return s
}

var _ Cache = (*SemanticCache)(nil)
