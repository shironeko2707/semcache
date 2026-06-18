package store

import (
	"context"
	"fmt"
	"math/rand"
	"sort"
	"testing"
	"time"
)

func randVec(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return Normalize(v)
}

// bruteTopK returns the keys of the k highest-cosine records for q.
func bruteTopK(recs []Record, q []float32, k int) map[string]bool {
	type sc struct {
		key string
		s   float32
	}
	scored := make([]sc, len(recs))
	for i, r := range recs {
		scored[i] = sc{r.Key, Dot(q, r.Vector)}
	}
	sort.Slice(scored, func(i, j int) bool { return scored[i].s > scored[j].s })
	out := make(map[string]bool, k)
	for i := 0; i < k && i < len(scored); i++ {
		out[scored[i].key] = true
	}
	return out
}

// TestHNSWRecall is the correctness gate for the ANN index: its top-k must
// closely match the exact (brute-force) top-k on random data.
func TestHNSWRecall(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))
	const (
		dim = 64
		n   = 2000
		k   = 10
	)
	h := NewHNSW(WithHNSWSeed(7))

	recs := make([]Record, n)
	for i := 0; i < n; i++ {
		v := randVec(rng, dim)
		recs[i] = Record{Key: fmt.Sprintf("k%d", i), Namespace: "n", Vector: v}
		if err := h.Set(ctx, recs[i]); err != nil {
			t.Fatalf("Set: %v", err)
		}
	}
	if h.Len() != n {
		t.Fatalf("Len=%d, want %d", h.Len(), n)
	}

	const queries = 200
	var totalRecall float64
	for i := 0; i < queries; i++ {
		q := randVec(rng, dim)
		truth := bruteTopK(recs, q, k)
		got, err := h.Nearest(ctx, "n", q, k)
		if err != nil {
			t.Fatalf("Nearest: %v", err)
		}
		hit := 0
		for _, m := range got {
			if truth[m.Record.Key] {
				hit++
			}
		}
		totalRecall += float64(hit) / float64(k)
	}
	recall := totalRecall / queries
	t.Logf("HNSW recall@%d over %d queries on %d×%dd vectors: %.3f", k, queries, n, dim, recall)
	if recall < 0.90 {
		t.Fatalf("recall %.3f below 0.90 — ANN index is not finding true neighbours", recall)
	}
}

// TestHNSWResultsSortedAndExact checks that for an exact-match query the record
// itself is returned first with similarity ~1.
func TestHNSWResultsSortedAndExact(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(1))
	h := NewHNSW(WithHNSWSeed(3))
	var target Record
	for i := 0; i < 500; i++ {
		r := Record{Key: fmt.Sprintf("k%d", i), Namespace: "n", Vector: randVec(rng, 48)}
		_ = h.Set(ctx, r)
		if i == 250 {
			target = r
		}
	}
	got, _ := h.Nearest(ctx, "n", target.Vector, 5)
	if len(got) == 0 || got[0].Record.Key != target.Key {
		t.Fatalf("exact query did not return its own record first: %+v", got)
	}
	if got[0].Score < 0.999 {
		t.Fatalf("self-match similarity %.4f, want ~1", got[0].Score)
	}
	for i := 1; i < len(got); i++ {
		if got[i].Score > got[i-1].Score {
			t.Fatal("results not sorted by descending similarity")
		}
	}
}

func TestHNSWDelete(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(2))
	h := NewHNSW()
	v := randVec(rng, 32)
	_ = h.Set(ctx, Record{Key: "a", Namespace: "n", Vector: v})
	_ = h.Set(ctx, Record{Key: "b", Namespace: "n", Vector: randVec(rng, 32)})

	if err := h.Delete(ctx, "n", "a"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if h.Len() != 1 {
		t.Fatalf("Len=%d after delete, want 1", h.Len())
	}
	got, _ := h.Nearest(ctx, "n", v, 5)
	for _, m := range got {
		if m.Record.Key == "a" {
			t.Fatal("deleted record was returned")
		}
	}
}

func TestHNSWTTLExpiry(t *testing.T) {
	ctx := context.Background()
	now := time.Unix(1_700_000_000, 0)
	h := NewHNSW(WithHNSWClock(func() time.Time { return now }))
	rng := rand.New(rand.NewSource(4))
	v := randVec(rng, 32)
	_ = h.Set(ctx, Record{Key: "a", Namespace: "n", Vector: v, ExpiresAt: now.Add(time.Minute)})

	if got, _ := h.Nearest(ctx, "n", v, 1); len(got) != 1 {
		t.Fatal("fresh entry should be returned")
	}
	now = now.Add(2 * time.Minute)
	if got, _ := h.Nearest(ctx, "n", v, 1); len(got) != 0 {
		t.Fatal("expired entry must not be returned")
	}
}

func TestHNSWEviction(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(5))
	evicted := 0
	h := NewHNSW(WithHNSWMaxEntries(50), WithHNSWEvictHook(func() { evicted++ }))
	base := time.Unix(1_700_000_000, 0)
	for i := 0; i < 60; i++ {
		_ = h.Set(ctx, Record{
			Key: fmt.Sprintf("k%d", i), Namespace: "n",
			Vector: randVec(rng, 32), CreatedAt: base.Add(time.Duration(i) * time.Second),
		})
	}
	if h.Len() != 50 {
		t.Fatalf("Len=%d, want 50 after eviction", h.Len())
	}
	if evicted != 10 {
		t.Fatalf("evicted=%d, want 10", evicted)
	}
}

// buildStore fills a Store with n random dim-vectors for benchmarking.
func buildStore(s Store, n, dim int) {
	rng := rand.New(rand.NewSource(99))
	for i := 0; i < n; i++ {
		_ = s.Set(context.Background(), Record{Key: fmt.Sprintf("k%d", i), Namespace: "n", Vector: randVec(rng, dim)})
	}
}

// BenchmarkNearestFlat / BenchmarkNearestHNSW contrast the exact O(n) flat scan
// with the sub-linear HNSW graph on the same index size.
func BenchmarkNearestFlat(b *testing.B) {
	const n, dim = 20000, 128
	s := NewMemory()
	buildStore(s, n, dim)
	q := randVec(rand.New(rand.NewSource(1)), dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = s.Nearest(context.Background(), "n", q, 10)
	}
}

func BenchmarkNearestHNSW(b *testing.B) {
	const n, dim = 20000, 128
	s := NewHNSW(WithHNSWSeed(1))
	buildStore(s, n, dim)
	q := randVec(rand.New(rand.NewSource(1)), dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = s.Nearest(context.Background(), "n", q, 10)
	}
}

func TestHNSWNamespaceIsolation(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(6))
	h := NewHNSW()
	v := randVec(rng, 32)
	_ = h.Set(ctx, Record{Key: "a", Namespace: "x", Vector: v})

	if got, _ := h.Nearest(ctx, "y", v, 5); len(got) != 0 {
		t.Fatal("namespace y should be empty")
	}
	if got, _ := h.Nearest(ctx, "x", v, 5); len(got) != 1 {
		t.Fatal("namespace x should return its record")
	}
}
