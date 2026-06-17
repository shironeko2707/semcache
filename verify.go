package semcache

import (
	"context"

	"github.com/shironeko2707/semcache/store"
)

// Verifier is the second stage of a hit decision. A high cosine score clears the
// similarity floor but is necessary, not sufficient: the Verifier decides
// whether the candidate is genuinely the same question. This is the core of the
// correctness-aware thesis — it is where false hits are caught before serving.
type Verifier interface {
	// Verify reports whether candidate may be served for the given canonical
	// query text. qText is already redacted+normalized.
	Verify(ctx context.Context, qText string, candidate store.Record) (bool, error)
}

// VerifierFunc adapts a function to the Verifier interface. Use this to plug in
// a cheap LLM-judge or any custom check. Kept off by default.
type VerifierFunc func(ctx context.Context, qText string, candidate store.Record) (bool, error)

// Verify calls the wrapped function.
func (f VerifierFunc) Verify(ctx context.Context, qText string, candidate store.Record) (bool, error) {
	return f(ctx, qText, candidate)
}

// FloorVerifier trusts the similarity floor alone and always accepts. Use only
// when false hits are cheap; not recommended for regulated workloads.
type FloorVerifier struct{}

// Verify always returns true.
func (FloorVerifier) Verify(context.Context, string, store.Record) (bool, error) {
	return true, nil
}

// LexicalVerifier accepts only if the token Jaccard overlap between the query
// and the candidate's stored text meets MinOverlap. This cheaply rejects the
// classic semantic-cache failure mode where two queries are vector-close but
// differ in a decisive token (a negation, a different entity, a different
// amount) — exactly the case that hurts in banking.
type LexicalVerifier struct {
	MinOverlap float64 // Jaccard threshold in [0,1]; default 0.6 if zero
}

// Verify computes token Jaccard overlap against candidate.Text.
func (v LexicalVerifier) Verify(_ context.Context, qText string, candidate store.Record) (bool, error) {
	min := v.MinOverlap
	if min == 0 {
		min = 0.6
	}
	return jaccard(tokenize(qText), tokenize(candidate.Text)) >= min, nil
}

// jaccard returns |A∩B| / |A∪B| over token sets. Two empty sets count as 1.
func jaccard(a, b []string) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1
	}
	sa := make(map[string]struct{}, len(a))
	for _, t := range a {
		sa[t] = struct{}{}
	}
	sb := make(map[string]struct{}, len(b))
	for _, t := range b {
		sb[t] = struct{}{}
	}
	inter := 0
	for t := range sa {
		if _, ok := sb[t]; ok {
			inter++
		}
	}
	union := len(sa) + len(sb) - inter
	if union == 0 {
		return 1
	}
	return float64(inter) / float64(union)
}
