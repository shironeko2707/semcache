package semcache

import (
	"context"
	"testing"
	"time"

	"github.com/bbbb/semcache/store"
)

// newTestCache builds a cache with a deterministic hash embedder and a fixed
// clock so TTL behaviour is testable.
func newTestCache(t *testing.T, p Policy, clock func() time.Time) *SemanticCache {
	t.Helper()
	c, err := New(NewHashEmbedder(256), WithPolicy(p), WithClock(clock))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func TestNewRequiresEmbedder(t *testing.T) {
	if _, err := New(nil); err != ErrNoEmbedder {
		t.Fatalf("want ErrNoEmbedder, got %v", err)
	}
}

func TestStoreThenLookupHits(t *testing.T) {
	ctx := context.Background()
	c := newTestCache(t, DefaultPolicy(), time.Now)

	q := Query{Text: "What is the daily transfer limit?", Namespace: "faq"}
	if err := c.Store(ctx, q, Entry{Response: "20,000,000 VND"}); err != nil {
		t.Fatalf("Store: %v", err)
	}

	// Exact repeat must hit.
	got, found, err := c.Lookup(ctx, q)
	if err != nil || !found {
		t.Fatalf("exact repeat: found=%v err=%v", found, err)
	}
	if got.Response != "20,000,000 VND" {
		t.Fatalf("wrong response: %q", got.Response)
	}

	// Casing/whitespace paraphrase canonicalizes to the same text -> hit.
	got, found, _ = c.Lookup(ctx, Query{Text: "  what is the DAILY   transfer limit? ", Namespace: "faq"})
	if !found || got.Response != "20,000,000 VND" {
		t.Fatalf("canonical paraphrase should hit: found=%v resp=%q", found, got.Response)
	}
}

func TestNamespaceIsolation(t *testing.T) {
	ctx := context.Background()
	c := newTestCache(t, DefaultPolicy(), time.Now)
	q := Query{Text: "account balance", Namespace: "tenant-a"}
	_ = c.Store(ctx, q, Entry{Response: "A"})

	if _, found, _ := c.Lookup(ctx, Query{Text: "account balance", Namespace: "tenant-b"}); found {
		t.Fatal("lookup crossed namespace boundary")
	}
}

func TestPIIRedactionKeysIdentically(t *testing.T) {
	ctx := context.Background()
	c := newTestCache(t, DefaultPolicy(), time.Now)

	// Two queries that differ only in PII must collapse to the same canonical
	// entry, and the raw PII must never appear in the stored text.
	_ = c.Store(ctx, Query{Text: "transfer status for account 1234567890", Namespace: "ops"}, Entry{Response: "pending"})

	_, found, err := c.Lookup(ctx, Query{Text: "transfer status for account 9876543210", Namespace: "ops"})
	if err != nil || !found {
		t.Fatalf("PII-only difference should hit same entry: found=%v err=%v", found, err)
	}

	r := NewRedactor()
	if got := r.Canonicalize("call me at 0912345678 or email a@b.com, cccd 012345678901"); got != "call me at <phone> or email <email>, cccd <cccd>" {
		t.Fatalf("redaction wrong: %q", got)
	}
}

func TestEpochGating(t *testing.T) {
	ctx := context.Background()
	c := newTestCache(t, DefaultPolicy(), time.Now)
	q := Query{Text: "current base interest rate", Namespace: "rates", Epoch: "2026Q1"}
	_ = c.Store(ctx, q, Entry{Response: "4.5%"})

	// Same text, newer epoch -> deterministic miss.
	if _, found, _ := c.Lookup(ctx, Query{Text: "current base interest rate", Namespace: "rates", Epoch: "2026Q2"}); found {
		t.Fatal("epoch mismatch must miss")
	}
	// Same epoch -> hit.
	if _, found, _ := c.Lookup(ctx, q); !found {
		t.Fatal("same epoch must hit")
	}
}

func TestMetaGating(t *testing.T) {
	ctx := context.Background()
	c := newTestCache(t, DefaultPolicy(), time.Now)
	_ = c.Store(ctx, Query{Text: "show my statements", Namespace: "u", Meta: map[string]string{"lang": "vi"}}, Entry{Response: "vi"})

	if _, found, _ := c.Lookup(ctx, Query{Text: "show my statements", Namespace: "u", Meta: map[string]string{"lang": "en"}}); found {
		t.Fatal("meta mismatch must miss")
	}
	if _, found, _ := c.Lookup(ctx, Query{Text: "show my statements", Namespace: "u", Meta: map[string]string{"lang": "vi"}}); !found {
		t.Fatal("meta match must hit")
	}
}

func TestTTLExpiry(t *testing.T) {
	ctx := context.Background()
	now := time.Unix(1_700_000_000, 0)
	clock := func() time.Time { return now }
	p := DefaultPolicy()
	p.TTL = time.Minute
	c := newTestCache(t, p, clock)

	q := Query{Text: "weather today", Namespace: "x"}
	_ = c.Store(ctx, q, Entry{Response: "sunny"})
	if _, found, _ := c.Lookup(ctx, q); !found {
		t.Fatal("fresh entry should hit")
	}
	now = now.Add(2 * time.Minute) // advance past TTL
	if _, found, _ := c.Lookup(ctx, q); found {
		t.Fatal("expired entry must miss")
	}
}

// TestFalseHitAvoided is the differentiator: two queries are vector-close but
// differ in a decisive token. A floor-only cache would serve the wrong answer;
// the lexical verifier must reject it and count it as a false hit avoided.
func TestFalseHitAvoided(t *testing.T) {
	ctx := context.Background()
	p := DefaultPolicy()
	p.SimilarityFloor = 0.5 // force the near-miss above the floor
	p.NegativeCache = true
	// A single negation token gives ~0.8 Jaccard, which a lax verifier waves
	// through — that is precisely the dangerous case. A regulated deployment
	// raises the bar (here MinOverlap 0.9) or adds an LLM-judge verifier.
	c, err := New(NewHashEmbedder(256), WithPolicy(p), WithVerifier(LexicalVerifier{MinOverlap: 0.9}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_ = c.Store(ctx, Query{Text: "is international transfer allowed", Namespace: "faq"}, Entry{Response: "yes"})

	// Differs by the negation "not" — semantically opposite, lexically close.
	_, found, err := c.Lookup(ctx, Query{Text: "is international transfer not allowed", Namespace: "faq"})
	if err != nil {
		t.Fatalf("lookup: %v", err)
	}
	if found {
		t.Fatal("near-miss with decisive token must NOT be served")
	}
	if s := c.Stats(); s.FalseHitsAvoided == 0 {
		t.Fatalf("expected a false hit to be avoided, stats=%+v", s)
	}
}

func TestStatsHitRate(t *testing.T) {
	ctx := context.Background()
	c := newTestCache(t, DefaultPolicy(), time.Now)
	_ = c.Store(ctx, Query{Text: "q one", Namespace: "n"}, Entry{Response: "1"})

	c.Lookup(ctx, Query{Text: "q one", Namespace: "n"})   // hit
	c.Lookup(ctx, Query{Text: "totally other", Namespace: "n"}) // miss

	s := c.Stats()
	if s.Hits != 1 || s.Misses != 1 {
		t.Fatalf("want 1 hit 1 miss, got %+v", s)
	}
	if hr := s.HitRate(); hr != 0.5 {
		t.Fatalf("want hit rate 0.5, got %v", hr)
	}
	if s.Size == 0 {
		t.Fatal("size should reflect stored entries")
	}
}

func TestMemoryEviction(t *testing.T) {
	ctx := context.Background()
	evicted := 0
	m := store.NewMemory(store.WithMaxEntries(2), store.WithEvictHook(func() { evicted++ }))
	c, err := New(NewHashEmbedder(64), WithStore(m))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for _, txt := range []string{"alpha", "bravo", "charlie"} {
		_ = c.Store(ctx, Query{Text: txt, Namespace: "n"}, Entry{Response: txt})
	}
	if m.Len() != 2 {
		t.Fatalf("want 2 entries after eviction, got %d", m.Len())
	}
	if evicted != 1 {
		t.Fatalf("want 1 eviction, got %d", evicted)
	}
}
