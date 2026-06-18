package redis_test

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/shironeko2707/semcache"
	redisstore "github.com/shironeko2707/semcache/store/redis"
)

// redisAddr is taken from REDIS_ADDR, defaulting to the conventional local
// Redis Stack port. Tests skip (not fail) when no Redis is reachable, so the
// module still passes in environments without one.
func redisAddr() string {
	if a := os.Getenv("REDIS_ADDR"); a != "" {
		return a
	}
	return "localhost:6379"
}

const dim = 256

// newCache builds a SemanticCache backed by a fresh, uniquely-named Redis index
// so subtests don't collide. It skips the test when Redis is unreachable.
func newCache(t *testing.T, p semcache.Policy, verifier semcache.Verifier) *semcache.SemanticCache {
	t.Helper()
	ctx := context.Background()
	uniq := fmt.Sprintf("%s_%d", t.Name(), time.Now().UnixNano())
	rs, err := redisstore.New(ctx, redisstore.Config{
		Addr:      redisAddr(),
		Dim:       dim,
		IndexName: "idx_" + uniq,
		Prefix:    "sc_" + uniq + ":",
	})
	if err != nil {
		t.Skipf("Redis not available at %s (%v) — skipping integration test", redisAddr(), err)
	}
	t.Cleanup(func() {
		_ = rs.DropIndex(ctx)
		_ = rs.Close()
	})
	opts := []semcache.Option{semcache.WithStore(rs), semcache.WithPolicy(p)}
	if verifier != nil {
		opts = append(opts, semcache.WithVerifier(verifier))
	}
	c, err := semcache.New(semcache.NewHashEmbedder(dim), opts...)
	if err != nil {
		t.Fatalf("New cache: %v", err)
	}
	return c
}

func TestRedisHitAndPIICollapse(t *testing.T) {
	ctx := context.Background()
	c := newCache(t, semcache.DefaultPolicy(), nil)

	q := semcache.Query{Text: "what is the transfer status for account 1234567890", Namespace: "ops"}
	if err := c.Store(ctx, q, semcache.Entry{Response: "pending"}); err != nil {
		t.Fatalf("Store: %v", err)
	}

	// Exact repeat hits and round-trips the payload through Redis.
	got, found, err := c.Lookup(ctx, q)
	if err != nil || !found {
		t.Fatalf("exact repeat: found=%v err=%v", found, err)
	}
	if got.Response != "pending" {
		t.Fatalf("payload round-trip wrong: %q", got.Response)
	}

	// A different account number redacts to the same canonical entry -> hit.
	if _, found, _ := c.Lookup(ctx, semcache.Query{Text: "what is the transfer status for account 9999999999", Namespace: "ops"}); !found {
		t.Fatal("PII-only difference should hit the same entry")
	}
}

func TestRedisNamespaceIsolation(t *testing.T) {
	ctx := context.Background()
	c := newCache(t, semcache.DefaultPolicy(), nil)
	_ = c.Store(ctx, semcache.Query{Text: "account balance", Namespace: "tenant-a"}, semcache.Entry{Response: "A"})

	if _, found, _ := c.Lookup(ctx, semcache.Query{Text: "account balance", Namespace: "tenant-b"}); found {
		t.Fatal("KNN filter leaked across namespaces")
	}
	if _, found, _ := c.Lookup(ctx, semcache.Query{Text: "account balance", Namespace: "tenant-a"}); !found {
		t.Fatal("same namespace should hit")
	}
}

func TestRedisEpochGating(t *testing.T) {
	ctx := context.Background()
	c := newCache(t, semcache.DefaultPolicy(), nil)
	q := semcache.Query{Text: "current base interest rate", Namespace: "rates", Epoch: "2026Q1"}
	_ = c.Store(ctx, q, semcache.Entry{Response: "4.5%"})

	if _, found, _ := c.Lookup(ctx, semcache.Query{Text: "current base interest rate", Namespace: "rates", Epoch: "2026Q2"}); found {
		t.Fatal("epoch mismatch must miss")
	}
	if _, found, _ := c.Lookup(ctx, q); !found {
		t.Fatal("same epoch must hit")
	}
}

func TestRedisServerSideTTL(t *testing.T) {
	ctx := context.Background()
	p := semcache.DefaultPolicy()
	p.TTL = time.Second // short, so we can observe server-side expiry
	c := newCache(t, p, nil)

	q := semcache.Query{Text: "weather today", Namespace: "x"}
	_ = c.Store(ctx, q, semcache.Entry{Response: "sunny"})
	if _, found, _ := c.Lookup(ctx, q); !found {
		t.Fatal("fresh entry should hit")
	}
	time.Sleep(1500 * time.Millisecond) // let the Redis key TTL elapse
	if _, found, _ := c.Lookup(ctx, q); found {
		t.Fatal("entry must expire via Redis key TTL")
	}
}

func TestRedisFalseHitAvoided(t *testing.T) {
	ctx := context.Background()
	p := semcache.DefaultPolicy()
	p.SimilarityFloor = 0.5 // force the near-miss above the floor
	c := newCache(t, p, semcache.LexicalVerifier{MinOverlap: 0.9})

	_ = c.Store(ctx, semcache.Query{Text: "is international transfer allowed", Namespace: "faq"}, semcache.Entry{Response: "yes"})

	_, found, err := c.Lookup(ctx, semcache.Query{Text: "is international transfer not allowed", Namespace: "faq"})
	if err != nil {
		t.Fatalf("lookup: %v", err)
	}
	if found {
		t.Fatal("near-miss with a decisive token must not be served")
	}
	if s := c.Stats(); s.FalseHitsAvoided == 0 {
		t.Fatalf("expected a false hit avoided, stats=%+v", s)
	}
}
