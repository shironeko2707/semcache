package store

import (
	"container/heap"
	"context"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// HNSW is an in-process approximate-nearest-neighbour Store using a Hierarchical
// Navigable Small World graph (Malkov & Yashunin, 2016). It is pure Go with no
// external dependencies and gives sub-linear lookups, trading exactness for
// speed at larger sizes than the flat Memory store comfortably handles.
//
// One graph is kept per namespace. Deletes and TTL expiry are handled by
// soft-deletion: an expired or deleted node remains in the graph as a routing
// hop but is never returned. Nearest is read-only (no graph mutation), so reads
// run concurrently under a read lock.
type HNSW struct {
	mu      sync.RWMutex
	ns      map[string]*hnswGraph
	now     func() time.Time
	max     int
	onEvict func()

	m              int // max neighbours per node on layers > 0
	mMax0          int // max neighbours on layer 0
	efConstruction int
	efSearch       int
	ml             float64
	seed           int64
}

// HNSWOption configures an HNSW store.
type HNSWOption func(*HNSW)

// WithHNSWClock overrides the time source (tests). Defaults to time.Now.
func WithHNSWClock(now func() time.Time) HNSWOption { return func(h *HNSW) { h.now = now } }

// WithHNSWMaxEntries bounds live entries across all namespaces; on overflow the
// oldest live node (by CreatedAt) is soft-deleted. Zero means unbounded.
func WithHNSWMaxEntries(n int) HNSWOption { return func(h *HNSW) { h.max = n } }

// WithHNSWEvictHook registers a callback invoked once per eviction.
func WithHNSWEvictHook(fn func()) HNSWOption { return func(h *HNSW) { h.onEvict = fn } }

// WithHNSWParams tunes the graph: m neighbours per node, efConstruction (build
// breadth), and efSearch (query breadth). Larger values raise recall and cost.
// Non-positive values keep the defaults (m=16, efConstruction=200, efSearch=64).
func WithHNSWParams(m, efConstruction, efSearch int) HNSWOption {
	return func(h *HNSW) {
		if m > 0 {
			h.m = m
		}
		if efConstruction > 0 {
			h.efConstruction = efConstruction
		}
		if efSearch > 0 {
			h.efSearch = efSearch
		}
	}
}

// WithHNSWSeed fixes the level-assignment RNG seed for reproducible builds.
func WithHNSWSeed(seed int64) HNSWOption { return func(h *HNSW) { h.seed = seed } }

// NewHNSW returns an empty HNSW store.
func NewHNSW(opts ...HNSWOption) *HNSW {
	h := &HNSW{
		ns:             make(map[string]*hnswGraph),
		now:            time.Now,
		m:              16,
		efConstruction: 200,
		efSearch:       64,
		seed:           1,
	}
	for _, o := range opts {
		o(h)
	}
	h.mMax0 = 2 * h.m
	h.ml = 1 / math.Log(float64(h.m))
	return h
}

type hnswNode struct {
	rec       Record
	vec       []float32 // normalized
	neighbors [][]int32 // neighbors[level] -> node indices
	deleted   bool
}

type hnswGraph struct {
	nodes    []*hnswNode
	byKey    map[string]int32
	entry    int32 // -1 when empty
	maxLevel int
	live     int
	rng      *rand.Rand
}

// Set inserts or replaces a record. Replacing a key soft-deletes the old node
// and inserts a fresh one (the vector may have changed, invalidating edges).
func (h *HNSW) Set(_ context.Context, rec Record) error {
	rec.Vector = Normalize(rec.Vector)
	h.mu.Lock()
	defer h.mu.Unlock()

	g := h.ns[rec.Namespace]
	if g == nil {
		g = &hnswGraph{byKey: make(map[string]int32), entry: -1, rng: rand.New(rand.NewSource(h.seed))}
		h.ns[rec.Namespace] = g
	}
	if old, ok := g.byKey[rec.Key]; ok && !g.nodes[old].deleted {
		g.nodes[old].deleted = true
		g.live--
	}
	idx := h.insert(g, rec)
	g.byKey[rec.Key] = idx
	g.live++

	if h.max > 0 {
		for h.liveTotalLocked() > h.max {
			if !h.evictOldestLocked() {
				break
			}
		}
	}
	return nil
}

// insert adds a node to the graph and wires its neighbours. Caller holds the lock.
func (h *HNSW) insert(g *hnswGraph, rec Record) int32 {
	level := int(-math.Log(g.rng.Float64()) * h.ml)
	node := &hnswNode{rec: rec, vec: rec.Vector, neighbors: make([][]int32, level+1)}
	idx := int32(len(g.nodes))
	g.nodes = append(g.nodes, node)

	if g.entry == -1 {
		g.entry = idx
		g.maxLevel = level
		return idx
	}

	ep := []int32{g.entry}
	for l := g.maxLevel; l > level; l-- {
		w := h.searchLayer(g, node.vec, ep, 1, l)
		ep = []int32{w[0].idx}
	}

	start := level
	if g.maxLevel < start {
		start = g.maxLevel
	}
	for l := start; l >= 0; l-- {
		w := h.searchLayer(g, node.vec, ep, h.efConstruction, l)
		neighbors := h.selectNeighbors(g, node.vec, w, h.m)
		node.neighbors[l] = neighbors
		mMax := h.m
		if l == 0 {
			mMax = h.mMax0
		}
		for _, nb := range neighbors {
			g.nodes[nb].neighbors[l] = append(g.nodes[nb].neighbors[l], idx)
			if len(g.nodes[nb].neighbors[l]) > mMax {
				g.nodes[nb].neighbors[l] = h.pruneNeighbors(g, nb, l, mMax)
			}
		}
		ep = make([]int32, len(w))
		for i, it := range w {
			ep[i] = it.idx
		}
	}

	if level > g.maxLevel {
		g.entry = idx
		g.maxLevel = level
	}
	return idx
}

// pruneNeighbors trims node nb's connections on a level back to mMax using the
// neighbour-selection heuristic, keeping the most useful edges.
func (h *HNSW) pruneNeighbors(g *hnswGraph, nb int32, level, mMax int) []int32 {
	cur := g.nodes[nb].neighbors[level]
	cands := make([]distItem, len(cur))
	for i, c := range cur {
		cands[i] = distItem{idx: c, d: cosDist(g.nodes[nb].vec, g.nodes[c].vec)}
	}
	return h.selectNeighbors(g, g.nodes[nb].vec, cands, mMax)
}

// searchLayer is the core greedy graph traversal: from entry points eps, return
// the ef nodes closest to q on the given level. Routing passes through deleted
// and expired nodes; filtering happens at the top level in Nearest.
func (h *HNSW) searchLayer(g *hnswGraph, q []float32, eps []int32, ef, level int) []distItem {
	visited := make(map[int32]struct{}, ef*2)
	cand := &minDist{}
	res := &maxDist{}
	heap.Init(cand)
	heap.Init(res)

	for _, e := range eps {
		d := cosDist(q, g.nodes[e].vec)
		visited[e] = struct{}{}
		heap.Push(cand, distItem{e, d})
		heap.Push(res, distItem{e, d})
	}

	for cand.Len() > 0 {
		c := heap.Pop(cand).(distItem)
		if res.Len() >= ef && c.d > (*res)[0].d {
			break
		}
		for _, nb := range g.nodes[c.idx].neighbors[level] {
			if _, seen := visited[nb]; seen {
				continue
			}
			visited[nb] = struct{}{}
			d := cosDist(q, g.nodes[nb].vec)
			if res.Len() < ef || d < (*res)[0].d {
				heap.Push(cand, distItem{nb, d})
				heap.Push(res, distItem{nb, d})
				if res.Len() > ef {
					heap.Pop(res)
				}
			}
		}
	}

	out := make([]distItem, res.Len())
	for i := len(out) - 1; i >= 0; i-- { // pop farthest-first -> reverse to nearest-first
		out[i] = heap.Pop(res).(distItem)
	}
	return out
}

// selectNeighbors is the HNSW heuristic (Algorithm 4): prefer candidates that
// are closer to q than to any already-selected neighbour, which spreads edges
// and improves recall over plain top-M. Falls back to closest-remaining to reach m.
func (h *HNSW) selectNeighbors(g *hnswGraph, q []float32, cands []distItem, m int) []int32 {
	sortByDist(cands)
	result := make([]int32, 0, m)
	for _, e := range cands {
		if len(result) >= m {
			break
		}
		good := true
		for _, r := range result {
			if cosDist(g.nodes[e.idx].vec, g.nodes[r].vec) < e.d {
				good = false
				break
			}
		}
		if good {
			result = append(result, e.idx)
		}
	}
	if len(result) < m {
		for _, e := range cands {
			if len(result) >= m {
				break
			}
			if !containsIdx(result, e.idx) {
				result = append(result, e.idx)
			}
		}
	}
	return result
}

// Nearest returns up to k non-deleted, non-expired records ordered by
// descending cosine similarity. It is read-only and safe under a read lock.
func (h *HNSW) Nearest(_ context.Context, namespace string, vec []float32, k int) ([]Match, error) {
	if k <= 0 {
		return nil, nil
	}
	q := Normalize(vec)
	now := h.now()

	h.mu.RLock()
	defer h.mu.RUnlock()
	g := h.ns[namespace]
	if g == nil || g.entry == -1 {
		return nil, nil
	}

	ep := []int32{g.entry}
	for l := g.maxLevel; l > 0; l-- {
		w := h.searchLayer(g, q, ep, 1, l)
		ep = []int32{w[0].idx}
	}
	ef := h.efSearch
	if ef < k {
		ef = k
	}
	w := h.searchLayer(g, q, ep, ef, 0)

	matches := make([]Match, 0, k)
	for _, it := range w {
		n := g.nodes[it.idx]
		if n.deleted || n.rec.Expired(now) {
			continue
		}
		matches = append(matches, Match{Record: n.rec, Score: 1 - it.d})
		if len(matches) == k {
			break
		}
	}
	return matches, nil
}

// Delete soft-deletes a record. A missing key is not an error.
func (h *HNSW) Delete(_ context.Context, namespace, key string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if g := h.ns[namespace]; g != nil {
		if idx, ok := g.byKey[key]; ok && !g.nodes[idx].deleted {
			g.nodes[idx].deleted = true
			g.live--
			delete(g.byKey, key)
		}
	}
	return nil
}

// Len returns the number of live (non-deleted) records across all namespaces.
func (h *HNSW) Len() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.liveTotalLocked()
}

// Close drops all graphs. The store is reusable afterward.
func (h *HNSW) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.ns = make(map[string]*hnswGraph)
	return nil
}

func (h *HNSW) liveTotalLocked() int {
	n := 0
	for _, g := range h.ns {
		n += g.live
	}
	return n
}

// evictOldestLocked soft-deletes the oldest live node across namespaces.
func (h *HNSW) evictOldestLocked() bool {
	var (
		bestG   *hnswGraph
		bestIdx int32 = -1
		bestAt  time.Time
		found   bool
	)
	for _, g := range h.ns {
		for i, n := range g.nodes {
			if n.deleted {
				continue
			}
			if !found || n.rec.CreatedAt.Before(bestAt) {
				bestG, bestIdx, bestAt, found = g, int32(i), n.rec.CreatedAt, true
			}
		}
	}
	if !found {
		return false
	}
	bestG.nodes[bestIdx].deleted = true
	bestG.live--
	delete(bestG.byKey, bestG.nodes[bestIdx].rec.Key)
	if h.onEvict != nil {
		h.onEvict()
	}
	return true
}

var _ Store = (*HNSW)(nil)

// cosDist is cosine distance for L2-normalized vectors: 1 - dot, in [0,2].
func cosDist(a, b []float32) float32 { return 1 - Dot(a, b) }

func containsIdx(s []int32, v int32) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

// distItem pairs a node index with its distance to the query.
type distItem struct {
	idx int32
	d   float32
}

func sortByDist(items []distItem) {
	sort.Slice(items, func(i, j int) bool { return items[i].d < items[j].d })
}

// minDist is a min-heap (closest first) of candidates to expand.
type minDist []distItem

func (m minDist) Len() int           { return len(m) }
func (m minDist) Less(i, j int) bool { return m[i].d < m[j].d }
func (m minDist) Swap(i, j int)      { m[i], m[j] = m[j], m[i] }
func (m *minDist) Push(x any)        { *m = append(*m, x.(distItem)) }
func (m *minDist) Pop() any          { old := *m; n := len(old); it := old[n-1]; *m = old[:n-1]; return it }

// maxDist is a max-heap (farthest first) bounding the result set to ef.
type maxDist []distItem

func (m maxDist) Len() int           { return len(m) }
func (m maxDist) Less(i, j int) bool { return m[i].d > m[j].d }
func (m maxDist) Swap(i, j int)      { m[i], m[j] = m[j], m[i] }
func (m *maxDist) Push(x any)        { *m = append(*m, x.(distItem)) }
func (m *maxDist) Pop() any          { old := *m; n := len(old); it := old[n-1]; *m = old[:n-1]; return it }
