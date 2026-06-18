// Package redis provides a RediSearch-backed store.Store for semcache, kept in
// its own module so the semcache root stays zero-dependency. It uses a
// RediSearch vector index (cosine) for nearest-neighbour lookup and Redis key
// TTLs for deterministic, server-side expiry. Requires Redis Stack (or any
// Redis with the RediSearch module loaded).
package redis

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	goredis "github.com/redis/go-redis/v9"
	"github.com/shironeko2707/semcache/store"
)

// Store implements store.Store against RediSearch. It is safe for concurrent use
// (the underlying go-redis client is). Vectors are stored normalized; lookups
// use a KNN query filtered by namespace.
type Store struct {
	rdb    *goredis.Client
	index  string
	prefix string
	dim    int
}

// Config configures the Redis-backed store.
type Config struct {
	Addr     string // host:port, e.g. "localhost:6379"
	Password string
	DB       int

	// Dim is the embedding dimension and MUST match the Embedder. It is fixed at
	// index creation; changing it requires dropping the index.
	Dim int

	IndexName string // default "semcache_idx"
	Prefix    string // hash key prefix, default "semcache:"
	UseHNSW   bool   // HNSW index when true; FLAT (exact) otherwise
}

const scoreField = "score" // KNN distance alias returned by FT.SEARCH

// New connects to Redis, verifies the connection, and ensures the search index
// exists. The client speaks RESP2 for predictable FT.SEARCH reply parsing.
func New(ctx context.Context, cfg Config) (*Store, error) {
	if cfg.Dim <= 0 {
		return nil, fmt.Errorf("redis store: Dim must be > 0")
	}
	if cfg.IndexName == "" {
		cfg.IndexName = "semcache_idx"
	}
	if cfg.Prefix == "" {
		cfg.Prefix = "semcache:"
	}
	rdb := goredis.NewClient(&goredis.Options{
		Addr:     cfg.Addr,
		Password: cfg.Password,
		DB:       cfg.DB,
		Protocol: 2, // RESP2: FT.SEARCH returns the classic flat array
	})
	if err := rdb.Ping(ctx).Err(); err != nil {
		_ = rdb.Close()
		return nil, fmt.Errorf("redis store: ping: %w", err)
	}
	s := &Store{rdb: rdb, index: cfg.IndexName, prefix: cfg.Prefix, dim: cfg.Dim}
	if err := s.ensureIndex(ctx, cfg.UseHNSW); err != nil {
		_ = rdb.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) ensureIndex(ctx context.Context, hnsw bool) error {
	algo := "FLAT"
	if hnsw {
		algo = "HNSW"
	}
	args := []any{
		"FT.CREATE", s.index, "ON", "HASH", "PREFIX", 1, s.prefix, "SCHEMA",
		"ns", "TAG",
		"epoch", "TAG",
		"neg", "TAG",
		"created", "NUMERIC",
		"vec", "VECTOR", algo, 6, "TYPE", "FLOAT32", "DIM", s.dim, "DISTANCE_METRIC", "COSINE",
	}
	if err := s.rdb.Do(ctx, args...).Err(); err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "already exists") {
			return nil
		}
		return fmt.Errorf("redis store: create index: %w", err)
	}
	return nil
}

func (s *Store) docID(namespace, key string) string {
	return s.prefix + namespace + ":" + key
}

// Set upserts a record as a Redis hash and applies a per-key TTL from
// rec.ExpiresAt so expiry is enforced server-side.
func (s *Store) Set(ctx context.Context, rec store.Record) error {
	vec := store.Normalize(rec.Vector)
	metaJSON, err := json.Marshal(rec.Meta)
	if err != nil {
		return fmt.Errorf("redis store: marshal meta: %w", err)
	}
	neg := "0"
	if rec.Negative {
		neg = "1"
	}
	id := s.docID(rec.Namespace, rec.Key)
	fields := []any{
		"ns", rec.Namespace,
		"epoch", rec.Epoch,
		"neg", neg,
		"created", rec.CreatedAt.UnixMilli(),
		"text", rec.Text,
		"payload", base64.StdEncoding.EncodeToString(rec.Payload),
		"meta", string(metaJSON),
		"vec", vecToBytes(vec),
	}

	pipe := s.rdb.TxPipeline()
	pipe.HSet(ctx, id, fields...)
	if !rec.ExpiresAt.IsZero() {
		pipe.PExpireAt(ctx, id, rec.ExpiresAt)
	} else {
		pipe.Persist(ctx, id)
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("redis store: set: %w", err)
	}
	return nil
}

// Nearest runs a KNN cosine search within a namespace and returns up to k
// matches ordered by descending similarity. Expired keys are dropped by Redis
// automatically, so they never appear.
func (s *Store) Nearest(ctx context.Context, namespace string, vec []float32, k int) ([]store.Match, error) {
	if k <= 0 {
		return nil, nil
	}
	blob := vecToBytes(store.Normalize(vec))
	query := fmt.Sprintf("(@ns:{%s})=>[KNN %d @vec $BLOB AS %s]", escapeTag(namespace), k, scoreField)
	args := []any{
		"FT.SEARCH", s.index, query,
		"PARAMS", 2, "BLOB", blob,
		"SORTBY", scoreField, "ASC",
		"RETURN", 6, scoreField, "text", "payload", "epoch", "neg", "meta",
		"DIALECT", 2,
		"LIMIT", 0, k,
	}
	raw, err := s.rdb.Do(ctx, args...).Slice()
	if err != nil {
		return nil, fmt.Errorf("redis store: search: %w", err)
	}
	return parseSearch(raw, namespace)
}

// Delete removes a record by namespace and key. A missing key is not an error.
func (s *Store) Delete(ctx context.Context, namespace, key string) error {
	if err := s.rdb.Del(ctx, s.docID(namespace, key)).Err(); err != nil {
		return fmt.Errorf("redis store: delete: %w", err)
	}
	return nil
}

// Len returns the number of indexed documents.
func (s *Store) Len() int {
	raw, err := s.rdb.Do(context.Background(),
		"FT.SEARCH", s.index, "*", "LIMIT", 0, 0, "DIALECT", 2).Slice()
	if err != nil || len(raw) == 0 {
		return 0
	}
	if n, ok := toInt(raw[0]); ok {
		return n
	}
	return 0
}

// Close releases the Redis client. It does not drop the index.
func (s *Store) Close() error { return s.rdb.Close() }

// DropIndex deletes the search index (keeping the documents). Useful in tests.
func (s *Store) DropIndex(ctx context.Context) error {
	return s.rdb.Do(ctx, "FT.DROPINDEX", s.index).Err()
}

var _ store.Store = (*Store)(nil)

// parseSearch turns a RESP2 FT.SEARCH reply [total, id, [f,v,...], id, [...]]
// into matches. Cosine similarity = 1 - cosine distance.
func parseSearch(raw []any, namespace string) ([]store.Match, error) {
	if len(raw) < 1 {
		return nil, nil
	}
	out := make([]store.Match, 0, (len(raw)-1)/2)
	for i := 1; i+1 < len(raw); i += 2 {
		id, _ := raw[i].(string)
		fieldsRaw, ok := raw[i+1].([]any)
		if !ok {
			continue
		}
		f := fieldsToMap(fieldsRaw)

		var sim float32
		if d, ok := parseFloat(f[scoreField]); ok {
			sim = float32(1 - d)
		}
		payload, _ := base64.StdEncoding.DecodeString(f["payload"])
		var meta map[string]string
		if f["meta"] != "" {
			_ = json.Unmarshal([]byte(f["meta"]), &meta)
		}
		out = append(out, store.Match{
			Score: sim,
			Record: store.Record{
				Key:       strings.TrimPrefix(id, prefixOf(id, namespace)),
				Namespace: namespace,
				Epoch:     f["epoch"],
				Text:      f["text"],
				Payload:   payload,
				Meta:      meta,
				Negative:  f["neg"] == "1",
			},
		})
	}
	return out, nil
}

// prefixOf returns the leading "<prefix><namespace>:" of a doc id so Key can be
// recovered. It is derived from the id itself to avoid threading the store
// prefix through parsing.
func prefixOf(id, namespace string) string {
	marker := namespace + ":"
	if idx := strings.Index(id, marker); idx >= 0 {
		return id[:idx+len(marker)]
	}
	return ""
}

func fieldsToMap(kv []any) map[string]string {
	m := make(map[string]string, len(kv)/2)
	for i := 0; i+1 < len(kv); i += 2 {
		name, _ := kv[i].(string)
		m[name] = toString(kv[i+1])
	}
	return m
}

func vecToBytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}

// escapeTag escapes RediSearch tag-query punctuation so an arbitrary namespace
// is matched literally.
func escapeTag(s string) string {
	var b strings.Builder
	for _, r := range s {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_') {
			b.WriteByte('\\')
		}
		b.WriteRune(r)
	}
	return b.String()
}

func toString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case []byte:
		return string(x)
	default:
		return fmt.Sprint(x)
	}
}

func toInt(v any) (int, bool) {
	switch x := v.(type) {
	case int64:
		return int(x), true
	case int:
		return x, true
	case string:
		var n int
		_, err := fmt.Sscan(x, &n)
		return n, err == nil
	default:
		return 0, false
	}
}

func parseFloat(s string) (float64, bool) {
	if s == "" {
		return 0, false
	}
	var f float64
	_, err := fmt.Sscan(s, &f)
	return f, err == nil
}
