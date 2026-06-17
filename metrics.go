package semcache

import "sync/atomic"

// Metrics holds lock-free counters for cache behaviour. The headline figure for
// the correctness thesis is FalseHitsAvoided: candidates that cleared the
// similarity floor but were rejected by verification — wrong answers the cache
// refused to serve. A generic cache never measures this.
type Metrics struct {
	hits             atomic.Uint64
	misses           atomic.Uint64
	falseHitsAvoided atomic.Uint64
	stores           atomic.Uint64
	evictions        atomic.Uint64
}

// Stats is an immutable snapshot of the counters plus the current store size.
type Stats struct {
	Hits             uint64
	Misses           uint64
	FalseHitsAvoided uint64
	Stores           uint64
	Evictions        uint64
	Size             int
}

// HitRate is hits / (hits + misses). Zero when there has been no lookup.
func (s Stats) HitRate() float64 {
	total := s.Hits + s.Misses
	if total == 0 {
		return 0
	}
	return float64(s.Hits) / float64(total)
}

// FalseHitRate estimates rejected-over-served-plus-rejected among floor-clearing
// candidates: FalseHitsAvoided / (Hits + FalseHitsAvoided). It approximates how
// often a floor-only cache would have served a wrong answer. Zero when no
// candidate ever cleared the floor.
func (s Stats) FalseHitRate() float64 {
	denom := s.Hits + s.FalseHitsAvoided
	if denom == 0 {
		return 0
	}
	return float64(s.FalseHitsAvoided) / float64(denom)
}

func (m *Metrics) incHit()             { m.hits.Add(1) }
func (m *Metrics) incMiss()            { m.misses.Add(1) }
func (m *Metrics) incFalseHitAvoided() { m.falseHitsAvoided.Add(1) }
func (m *Metrics) incStore()           { m.stores.Add(1) }
func (m *Metrics) incEviction()        { m.evictions.Add(1) }

// snapshot reads the counters into a Stats (size filled in by the caller).
func (m *Metrics) snapshot() Stats {
	return Stats{
		Hits:             m.hits.Load(),
		Misses:           m.misses.Load(),
		FalseHitsAvoided: m.falseHitsAvoided.Load(),
		Stores:           m.stores.Load(),
		Evictions:        m.evictions.Load(),
	}
}
