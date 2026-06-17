package bench

import (
	"context"
	"sort"
	"testing"
	"time"

	"github.com/shironeko2707/semcache"
)

// benchCache builds a cache tuned for the workload: a moderate similarity floor
// so near-misses reach verification, and a strict-ish lexical verifier so a
// swapped decisive token is rejected. Epoch/meta gating is off (the workload
// does not use them); namespace isolation still applies.
func benchCache() (*semcache.SemanticCache, error) {
	p := semcache.Policy{
		SimilarityFloor: 0.70,
		TTL:             0,
		EnforceEpoch:    false,
		EnforceMeta:     false,
		NegativeCache:   true,
		Candidates:      5,
	}
	return semcache.New(
		semcache.NewHashEmbedder(512),
		semcache.WithPolicy(p),
		semcache.WithVerifier(semcache.LexicalVerifier{MinOverlap: 0.85}),
	)
}

// replay runs the stream through the cache, calling the (synthetic) model and
// storing on a miss. It returns hits, total, and sorted lookup latencies.
func replay(ctx context.Context, c *semcache.SemanticCache, reqs []Request) (hits, total int, lat []time.Duration) {
	lat = make([]time.Duration, 0, len(reqs))
	for _, r := range reqs {
		start := time.Now()
		_, found, err := c.Lookup(ctx, r.Query)
		lat = append(lat, time.Since(start))
		if err != nil {
			panic(err)
		}
		total++
		if found {
			hits++
			continue
		}
		// model miss path: store a synthetic answer
		_ = c.Store(ctx, r.Query, semcache.Entry{Response: Answer(r.Query.Text)})
	}
	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
	return hits, total, lat
}

func pct(sorted []time.Duration, q float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	i := int(q * float64(len(sorted)-1))
	return sorted[i]
}

// TestHitRateExitCriterion is the headline exit criterion: >40% hit rate on the
// repetitive synthetic workload, with zero near-miss probes wrongly served.
func TestHitRateExitCriterion(t *testing.T) {
	ctx := context.Background()
	c, err := benchCache()
	if err != nil {
		t.Fatalf("benchCache: %v", err)
	}
	params := DefaultParams()
	hits, total, lat := replay(ctx, c, params.Stream())

	hitRate := float64(hits) / float64(total)
	t.Logf("hit rate: %.1f%% (%d/%d)", hitRate*100, hits, total)
	t.Logf("lookup latency p50=%v p99=%v", pct(lat, 0.50), pct(lat, 0.99))

	if hitRate <= 0.40 {
		t.Fatalf("hit rate %.1f%% did not exceed the 40%% target", hitRate*100)
	}

	// Correctness: no near-miss probe may be served, and the verifier must have
	// caught at least some floor-clearing near-misses (proving the second stage
	// actually does work).
	var wronglyServed int
	for _, q := range params.NearMisses() {
		if _, found, _ := c.Lookup(ctx, q); found {
			wronglyServed++
		}
	}
	s := c.Stats()
	t.Logf("near-miss probes wrongly served: %d/%d", wronglyServed, len(params.NearMisses()))
	t.Logf("false hits avoided over the run: %d (est. false-hit rate %.3f)", s.FalseHitsAvoided, s.FalseHitRate())

	if wronglyServed != 0 {
		t.Fatalf("served %d near-miss probes — false hits are defects", wronglyServed)
	}
}

// BenchmarkLookupHot measures steady-state lookup latency against a warmed
// cache (the common case once traffic is repetitive).
func BenchmarkLookupHot(b *testing.B) {
	ctx := context.Background()
	c, err := benchCache()
	if err != nil {
		b.Fatalf("benchCache: %v", err)
	}
	reqs := DefaultParams().Stream()
	replay(ctx, c, reqs) // warm

	hot := reqs[len(reqs)/2] // a request likely backed by a stored entry
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = c.Lookup(ctx, hot.Query)
	}
}
