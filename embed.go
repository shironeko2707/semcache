package semcache

import (
	"context"
	"hash/fnv"

	"github.com/shironeko2707/semcache/store"
)

// Embedder turns canonical query text into a vector. It is intentionally
// minimal and bring-your-own: wrap any provider (local model, hosted embedding
// API, ONNX runtime) behind this one method. semcache never locks you to a
// vendor and never calls a provider implicitly.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// EmbedderFunc adapts a plain function to the Embedder interface.
type EmbedderFunc func(ctx context.Context, text string) ([]float32, error)

// Embed calls the wrapped function.
func (f EmbedderFunc) Embed(ctx context.Context, text string) ([]float32, error) {
	return f(ctx, text)
}

// HashEmbedder is a deterministic, dependency-free feature-hashing embedder.
// Each token is hashed to a dimension and a sign, accumulated, then the vector
// is L2-normalized. Lexically similar text yields similar vectors, which makes
// it ideal for tests, benchmarks, and offline/air-gapped development without a
// model. It is NOT a semantic model — swap in a real Embedder for production
// semantics. Provided here so the default path stays zero-dependency.
type HashEmbedder struct {
	dims int
}

// NewHashEmbedder returns a HashEmbedder producing dims-dimensional vectors.
// dims <= 0 falls back to 256.
func NewHashEmbedder(dims int) *HashEmbedder {
	if dims <= 0 {
		dims = 256
	}
	return &HashEmbedder{dims: dims}
}

// Embed hashes the canonical text's tokens into a normalized vector.
func (h *HashEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	vec := make([]float32, h.dims)
	for _, tok := range tokenize(text) {
		sum := fnv.New32a()
		sum.Write([]byte(tok))
		hv := sum.Sum32()
		idx := hv % uint32(h.dims)
		// Use a separate bit for the sign to decorrelate it from the index.
		if hv&0x80000000 != 0 {
			vec[idx] -= 1
		} else {
			vec[idx] += 1
		}
	}
	return store.Normalize(vec), nil
}
