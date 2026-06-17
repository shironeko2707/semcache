package semcache

import "time"

// Policy captures the correctness knobs that distinguish a regulated-workload
// cache from a generic one. Defaults are deliberately conservative: a high
// similarity floor and epoch/meta gating on, so the cache errs toward a miss
// (call the model) rather than risk a wrong hit.
type Policy struct {
	// SimilarityFloor is the minimum cosine similarity for a candidate to even
	// be considered. Necessary but not sufficient — verification still runs.
	SimilarityFloor float32

	// TTL bounds how long an entry may be served. Zero means no expiry.
	TTL time.Duration

	// EnforceEpoch requires the query's knowledge epoch to equal the entry's.
	// A mismatch is always a miss, regardless of similarity — this is how
	// answers that depend on changing facts expire deterministically.
	EnforceEpoch bool

	// EnforceMeta requires every key/value in the query's Meta to be present
	// and equal in the candidate's Meta (tenant/filter isolation).
	EnforceMeta bool

	// NegativeCache records verified near-misses as negative anchors so a
	// semantically-close-but-different query is not re-evaluated into a wrong
	// hit and is short-circuited to a miss.
	NegativeCache bool

	// Candidates is how many nearest neighbours to verify before giving up.
	Candidates int
}

// DefaultPolicy returns the recommended starting policy: a 0.92 floor, 24h TTL,
// epoch and meta gating enabled, negative cache on, top-5 candidate verification.
func DefaultPolicy() Policy {
	return Policy{
		SimilarityFloor: 0.92,
		TTL:             24 * time.Hour,
		EnforceEpoch:    true,
		EnforceMeta:     true,
		NegativeCache:   true,
		Candidates:      5,
	}
}

// metaSubset reports whether every key/value in want is present and equal in
// have. An empty want always matches.
func metaSubset(want, have map[string]string) bool {
	for k, v := range want {
		if have[k] != v {
			return false
		}
	}
	return true
}
