// Command gateway_middleware shows how an LLM gateway would put semcache in
// front of a model call — as a thin wrapper, with no import cycle and no
// provider lock-in. The cache is a library; the gateway owns the model call and
// simply asks the cache first. Run: go run ./examples
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shironeko2707/semcache"
)

// callModel stands in for the expensive provider call the gateway would make.
func callModel(_ context.Context, prompt string) string {
	return "model answer for: " + prompt
}

// cachedComplete is the middleware: look up, and on a miss call the model and
// store the result. The cache decides correctness (floor + verification + epoch
// + PII keying); the gateway just supplies the model.
func cachedComplete(ctx context.Context, c semcache.Cache, q semcache.Query) (string, bool, error) {
	if entry, found, err := c.Lookup(ctx, q); err != nil {
		return "", false, err
	} else if found {
		return entry.Response, true, nil
	}
	resp := callModel(ctx, q.Text)
	if err := c.Store(ctx, q, semcache.Entry{Response: resp}); err != nil {
		return "", false, err
	}
	return resp, false, nil
}

func main() {
	ctx := context.Background()
	c, err := semcache.New(semcache.NewHashEmbedder(256))
	if err != nil {
		log.Fatal(err)
	}

	// Two queries that differ only in PII collapse to one cached entry.
	q1 := semcache.Query{Text: "what is the transfer status for account 1234567890", Namespace: "ops"}
	q2 := semcache.Query{Text: "what is the transfer status for account 9876543210", Namespace: "ops"}

	for i, q := range []semcache.Query{q1, q2} {
		resp, hit, err := cachedComplete(ctx, c, q)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("request %d: hit=%v resp=%q\n", i+1, hit, resp)
	}

	s := c.Stats()
	fmt.Printf("stats: hits=%d misses=%d size=%d hitRate=%.0f%%\n",
		s.Hits, s.Misses, s.Size, s.HitRate()*100)
}
