# semcache — a correctness-aware semantic cache for LLM responses

Most semantic caches optimize one number: hit rate. In a regulated workload — banking, healthcare, anything audited — that framing is dangerous, because a **wrong cache hit is a defect, not a missed optimization.** Serving a confidently-cached answer to a question that was *almost* the same one is worse than calling the model.

`semcache` is built for that setting. It treats a bad hit as a bug and makes correctness a first-class, measurable property:

- **Similarity floor + second-stage verification.** A high cosine score is *necessary but not sufficient*. Every floor-clearing candidate goes through a pluggable verifier (lexical-overlap by default, or your own LLM-judge) before it is served.
- **PII-aware keying.** Raw PII never enters the index. Query text is redacted (synthetic Vietnamese shapes: CCCD, mobile, card, account, email) and normalized *before* it is embedded, keyed, or stored — so two queries that differ only by an account number correctly collapse to one entry, and secrets are never persisted.
- **Deterministic staleness.** Per-entry TTL plus an optional **knowledge epoch**: when a fact changes, bump the epoch and every answer that depended on it deterministically expires — no guessing.
- **Negative cache.** A verified near-miss is remembered as a negative anchor, so a semantically-close-but-different query is short-circuited to a miss instead of risking the wrong answer again.
- **It measures the thing that matters.** Stats report not just hit rate but `FalseHitsAvoided` — candidates that cleared the floor and were rejected by verification. A generic cache never measures this.

The default path is **zero-dependency and in-memory**. Bring your own embedder; there is no provider lock-in and no implicit network call.

## Install

```bash
go get github.com/bbbb/semcache
```

Requires Go 1.26+.

## Quickstart

```go
c, _ := semcache.New(semcache.NewHashEmbedder(256)) // swap in a real Embedder for production semantics

q := semcache.Query{Text: "what is the daily transfer limit?", Namespace: "faq"}

if entry, found, _ := c.Lookup(ctx, q); found {
    return entry.Response
}
resp := callModel(ctx, q.Text)
_ = c.Store(ctx, q, semcache.Entry{Response: resp})
```

See [`examples/gateway_middleware.go`](examples/gateway_middleware.go) for the LLM-gateway wrapper pattern (no import cycle, cache stays a library).

## The design in one diagram

```
Lookup(query)
  │
  ├─ redact PII + normalize           ← raw PII never embedded/stored
  ├─ embed (your Embedder)
  ├─ Nearest(namespace, k)            ← pluggable Store (in-memory default)
  │
  └─ for each candidate, best-first:
       score < floor?      → stop, miss
       epoch mismatch?     → skip            (deterministic staleness)
       meta mismatch?      → skip            (tenant/filter isolation)
       negative anchor?    → miss            (known near-miss)
       verify(query,cand)? → no → false-hit avoided, record negative, skip
                             yes → HIT
```

## Bring your own embedder

`HashEmbedder` is a deterministic, dependency-free feature-hashing embedder — ideal for tests, benchmarks, and offline/air-gapped development. It is **not** a semantic model. For production semantics, implement the one-method interface around any model:

```go
type Embedder interface {
    Embed(ctx context.Context, text string) ([]float32, error)
}
```

## Verification strategies

| Verifier | When |
|----------|------|
| `LexicalVerifier{MinOverlap}` (default) | Cheap, no extra calls. Catches entity/amount swaps. Raise `MinOverlap` to be stricter. |
| `VerifierFunc` (LLM-judge) | Highest fidelity for cases lexical overlap misses (e.g. single-token negation). Off by default; opt in per deployment. |
| `FloorVerifier` | Floor-only. Only when false hits are genuinely cheap. |

**Honest limitation:** a single negation token (`"... allowed"` vs `"... not allowed"`) yields high lexical overlap. The lexical verifier alone will not catch it at a low threshold — raise `MinOverlap` or add an LLM-judge verifier for that class of query. The library makes this a deliberate, configurable choice rather than a silent one.

## Benchmark / exit criteria

```bash
make bench
```

On the synthetic repetitive banking workload (`bench/`), the reproducible targets are:

- **Hit rate > 40%** — current run: ~68% over a 60/40 repeat/long-tail mix.
- **Near-miss probes wrongly served: 0** — the correctness headline.
- **Lookup p99 < 5ms** in-memory — current run: ~1ms.

All workload data is synthetic. Numbers are reproducible from `make bench`.

## Status & roadmap

- [x] In-memory store, PII-aware keying, floor + verification, epoch/meta gating, negative cache, metrics, benchmark harness.
- [ ] Redis / RediSearch backend behind the same `store.Store` interface (kept out of the root module to preserve the zero-dependency default).
- [ ] ANN index for the in-memory store (today it is a flat cosine scan — exact, simple, O(n) per lookup; fine to the ~10k-entry range, swap in ANN beyond that).

## License

Apache-2.0. See [LICENSE](LICENSE). Authored independently on personal time and hardware; see [NOTICE](NOTICE).
