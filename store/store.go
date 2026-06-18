// Package store defines the pluggable backend interface for semcache and ships
// a zero-dependency in-memory implementation. Backends hold opaque payloads so
// the store layer never imports the cache layer (no import cycle).
package store

import (
	"context"
	"math"
	"time"
)

// Record is a single cached item as seen by a backend. The payload is an opaque
// byte blob: the cache layer owns its (de)serialization, so backends — in-memory
// or remote — never need to know the cache's value type. Text is the
// redacted+normalized query text (safe to persist — PII has been stripped) and
// is used by second-stage verification.
type Record struct {
	Key       string
	Namespace string
	Epoch     string
	Text      string
	Vector    []float32
	Payload   []byte
	Meta      map[string]string
	CreatedAt time.Time
	ExpiresAt time.Time // zero value means no expiry
	Negative  bool      // a known near-miss anchor; never served, used to short-circuit
}

// Expired reports whether the record is past its TTL as of now.
func (r Record) Expired(now time.Time) bool {
	return !r.ExpiresAt.IsZero() && now.After(r.ExpiresAt)
}

// Match is a record paired with its cosine similarity to the query vector.
type Match struct {
	Record Record
	Score  float32
}

// Store is the minimal backend contract. Implementations must be safe for
// concurrent use. Nearest returns up to k matches in a namespace ordered by
// descending cosine similarity, skipping expired records.
type Store interface {
	Set(ctx context.Context, rec Record) error
	Nearest(ctx context.Context, namespace string, vec []float32, k int) ([]Match, error)
	Delete(ctx context.Context, namespace, key string) error
	Len() int
	Close() error
}

// Normalize returns an L2-normalized copy of v. A zero vector is returned
// unchanged. Storing normalized vectors lets cosine similarity reduce to a dot
// product on lookup.
func Normalize(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return v
	}
	inv := float32(1 / math.Sqrt(sum))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

// Dot returns the dot product of two equal-length vectors. For L2-normalized
// inputs this equals cosine similarity.
func Dot(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
