package store

import (
	"context"
	"sort"
	"sync"
	"time"
)

// Memory is an in-process, zero-dependency Store backed by a flat cosine scan.
// It is the default backend: predictable, dependency-free, and fast enough for
// the tens-of-thousands-of-entries range a per-instance cache targets. Vectors
// are stored L2-normalized so Nearest is a dot product. For very large indexes,
// swap in an ANN-backed Store behind the same interface.
type Memory struct {
	mu      sync.RWMutex
	ns      map[string]map[string]Record // namespace -> key -> record
	max     int                          // 0 = unbounded
	now     func() time.Time
	onEvict func()
}

// MemoryOption configures a Memory store.
type MemoryOption func(*Memory)

// WithMaxEntries bounds the total number of records; on overflow the oldest
// record (by CreatedAt) is evicted. Zero means unbounded.
func WithMaxEntries(n int) MemoryOption {
	return func(m *Memory) { m.max = n }
}

// WithClock overrides the time source (tests). Defaults to time.Now.
func WithClock(now func() time.Time) MemoryOption {
	return func(m *Memory) { m.now = now }
}

// WithEvictHook registers a callback invoked once per eviction (metrics).
func WithEvictHook(fn func()) MemoryOption {
	return func(m *Memory) { m.onEvict = fn }
}

// NewMemory returns an empty in-memory store.
func NewMemory(opts ...MemoryOption) *Memory {
	m := &Memory{ns: make(map[string]map[string]Record), now: time.Now}
	for _, o := range opts {
		o(m)
	}
	return m
}

// Set inserts or replaces a record, normalizing its vector. It enforces the
// max-entries bound by evicting the oldest record when necessary.
func (m *Memory) Set(_ context.Context, rec Record) error {
	rec.Vector = Normalize(rec.Vector)
	m.mu.Lock()
	defer m.mu.Unlock()

	bucket := m.ns[rec.Namespace]
	if bucket == nil {
		bucket = make(map[string]Record)
		m.ns[rec.Namespace] = bucket
	}
	_, replacing := bucket[rec.Key]
	bucket[rec.Key] = rec

	if m.max > 0 && !replacing {
		for m.lenLocked() > m.max {
			if !m.evictOldestLocked() {
				break
			}
		}
	}
	return nil
}

// Nearest returns up to k non-expired records in namespace ordered by
// descending cosine similarity. Expired records encountered during the scan are
// dropped (lazy expiry).
func (m *Memory) Nearest(_ context.Context, namespace string, vec []float32, k int) ([]Match, error) {
	if k <= 0 {
		return nil, nil
	}
	q := Normalize(vec)
	now := m.now()

	m.mu.Lock() // write lock: we may drop expired entries
	defer m.mu.Unlock()
	bucket := m.ns[namespace]
	if len(bucket) == 0 {
		return nil, nil
	}

	matches := make([]Match, 0, len(bucket))
	for key, rec := range bucket {
		if rec.Expired(now) {
			delete(bucket, key)
			continue
		}
		matches = append(matches, Match{Record: rec, Score: Dot(q, rec.Vector)})
	}
	sort.Slice(matches, func(i, j int) bool { return matches[i].Score > matches[j].Score })
	if len(matches) > k {
		matches = matches[:k]
	}
	return matches, nil
}

// Delete removes a record by namespace and key. Missing keys are not an error.
func (m *Memory) Delete(_ context.Context, namespace, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if bucket := m.ns[namespace]; bucket != nil {
		delete(bucket, key)
	}
	return nil
}

// Len returns the total number of records across all namespaces.
func (m *Memory) Len() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.lenLocked()
}

// Close releases all entries. The store is reusable afterward.
func (m *Memory) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ns = make(map[string]map[string]Record)
	return nil
}

func (m *Memory) lenLocked() int {
	n := 0
	for _, bucket := range m.ns {
		n += len(bucket)
	}
	return n
}

// evictOldestLocked removes the single oldest record across all namespaces.
// Returns false if there was nothing to evict.
func (m *Memory) evictOldestLocked() bool {
	var (
		oldestNS, oldestKey string
		oldestAt            time.Time
		found               bool
	)
	for ns, bucket := range m.ns {
		for key, rec := range bucket {
			if !found || rec.CreatedAt.Before(oldestAt) {
				oldestNS, oldestKey, oldestAt, found = ns, key, rec.CreatedAt, true
			}
		}
	}
	if !found {
		return false
	}
	delete(m.ns[oldestNS], oldestKey)
	if m.onEvict != nil {
		m.onEvict()
	}
	return true
}
