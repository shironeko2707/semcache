// Package bench generates a synthetic, repetitive workload for measuring the
// semantic cache's hit rate and — crucially — its false-hit rate. All data is
// synthetic and Vietnamese-banking-flavoured; nothing here derives from any
// real dataset. The workload deliberately mixes three things a real traffic
// stream has: exact/surface repeats, PII-varying repeats that must collapse to
// one entry, and a long tail of unique novel queries that must miss.
package bench

import (
	"fmt"
	"math/rand"
	"strings"

	"github.com/shironeko2707/semcache"
)

// base is a repeatable intent. nearMiss, when set, is a lexically-close but
// semantically-different probe that must NOT be served from base's entry.
type base struct {
	q        string
	a        string
	nearMiss string
}

// piiBase is a repeatable intent whose query embeds a PII value (filled into the
// single %s slot). Varying the PII must not create new cache entries because it
// is redacted before keying.
type piiBase struct {
	tmpl string
	a    string
}

func bases() []base {
	return []base{
		{"what is the domestic transfer limit", "20,000,000 VND/day", "what is the international transfer limit"},
		{"how do i reset my internet banking password", "use forgot-password on the login screen", "how do i reset my card pin"},
		{"is the savings account interest paid monthly", "yes, monthly", "is the savings account interest paid yearly"},
		{"what documents do i need to open an account", "id card and proof of address", "what documents do i need to close an account"},
		{"how long does an interbank transfer take", "within 5 minutes via napas", "how long does an international transfer take"},
		{"can i increase my credit card limit online", "yes, in the app under card settings", "can i decrease my credit card limit online"},
		{"what is the fee for an atm withdrawal", "1,100 VND per withdrawal", "what is the fee for an atm balance enquiry"},
		{"is foreign currency exchange available at branches", "yes, at selected branches", "is foreign currency exchange available at atms"},
	}
}

func piiBases() []piiBase {
	return []piiBase{
		{"what is the transfer status for account %s", "your transfer is pending settlement"},
		{"please confirm the balance on card %s", "balance available in the app"},
		{"register mobile %s for otp notifications", "otp notifications enabled"},
	}
}

// Request is one item in the replayed stream.
type Request struct {
	Query semcache.Query
}

// Params control the synthetic stream shape.
type Params struct {
	N           int     // total requests in the stream
	RepeatRatio float64 // fraction drawn from the repeatable intent pool; rest are unique
	Namespace   string
	Seed        int64
}

// DefaultParams returns a stream that lands comfortably above the 40% hit-rate
// target without being trivially 100%: ~60% repeatable traffic over a small
// intent pool, ~40% unique novel queries forming a realistic long tail.
func DefaultParams() Params {
	return Params{N: 5000, RepeatRatio: 0.6, Namespace: "bench", Seed: 1}
}

// Stream builds the replayable request sequence for the given params.
func (p Params) Stream() []Request {
	rng := rand.New(rand.NewSource(p.Seed))
	bs, ps := bases(), piiBases()
	reqs := make([]Request, 0, p.N)
	for i := 0; i < p.N; i++ {
		var text string
		switch {
		case rng.Float64() < p.RepeatRatio && rng.Intn(3) > 0:
			// surface-variant repeat of a plain intent
			text = surfaceVariant(rng, bs[rng.Intn(len(bs))].q)
		case rng.Float64() < p.RepeatRatio:
			// PII-varying repeat: same intent, different (synthetic) PII value
			pb := ps[rng.Intn(len(ps))]
			text = fmt.Sprintf(pb.tmpl, syntheticPII(rng))
		default:
			// unique novel query — must miss
			text = uniqueQuery(rng, i)
		}
		reqs = append(reqs, Request{Query: semcache.Query{Text: text, Namespace: p.Namespace}})
	}
	return reqs
}

// NearMisses returns one probe per intent that defines a near-miss. Each probe
// is vector-close to a stored entry but semantically different and must not be
// served. Use these after replaying a Stream to measure the false-hit rate.
func (p Params) NearMisses() []semcache.Query {
	var out []semcache.Query
	for _, b := range bases() {
		if b.nearMiss != "" {
			out = append(out, semcache.Query{Text: b.nearMiss, Namespace: p.Namespace})
		}
	}
	return out
}

// Answer returns the canonical synthetic answer for a query's intent, used to
// populate the cache on a miss during replay. Falls back to a generic string.
func Answer(text string) string {
	for _, b := range bases() {
		if strings.Contains(strings.ToLower(text), b.q) {
			return b.a
		}
	}
	return "synthetic answer"
}

// surfaceVariant perturbs casing and whitespace only, so the canonical form
// (and thus the cache key) is unchanged — exactly the easy repeat a cache must
// catch.
func surfaceVariant(rng *rand.Rand, q string) string {
	words := strings.Fields(q)
	var b strings.Builder
	for i, w := range words {
		if i > 0 {
			b.WriteString(strings.Repeat(" ", 1+rng.Intn(2)))
		}
		if rng.Intn(4) == 0 {
			w = strings.ToUpper(w)
		}
		b.WriteString(w)
	}
	return b.String()
}

// syntheticPII returns a random PII-shaped value (12-digit CCCD or 0-prefixed
// mobile) that the redactor collapses to a placeholder.
func syntheticPII(rng *rand.Rand) string {
	if rng.Intn(2) == 0 {
		return fmt.Sprintf("%012d", rng.Int63n(1e12)) // CCCD shape
	}
	return fmt.Sprintf("0%09d", rng.Int63n(1e9)) // mobile shape
}

var topics = []string{"loan", "mortgage", "deposit", "insurance", "fx rate", "branch hours", "card", "fee", "statement", "dispute"}

// uniqueQuery fabricates a one-off query unlikely to recur, forming the long
// tail that should always miss.
func uniqueQuery(rng *rand.Rand, i int) string {
	return fmt.Sprintf("tell me about %s option %d variant %d", topics[rng.Intn(len(topics))], i, rng.Intn(1<<30))
}
