# Contributing to semcache

Thanks for your interest. semcache is a small, focused library; contributions
that keep it that way are very welcome.

## Principles

1. **Correctness over hit rate.** A change that raises hit rate but can serve a
   wrong answer is a regression. Every PR that touches lookup logic must keep
   `TestHitRateExitCriterion`'s "near-miss probes wrongly served: 0" intact.
2. **Zero-dependency default.** The root module must stay dependency-free. New
   backends (Redis, ANN, etc.) belong behind the `store.Store` interface and,
   if they need third-party deps, in their own nested module so the default
   install pulls nothing.
3. **No provider lock-in.** Never call an embedding/LLM provider implicitly.
   Everything provider-specific goes behind `Embedder` or `Verifier`.

## Before you open a PR

```bash
make vet      # go vet ./...
make test     # go test ./...
make bench    # exit-criterion + hot-lookup micro-bench
```

CI runs `go test -race ./...` and the benchmark smoke test on Go 1.26.

## Style

- Match the surrounding code: small interfaces, table-driven tests, doc comments
  that explain *why*, not *what*.
- New behaviour needs a test. New correctness knobs need a test that proves they
  prevent a wrong hit.

## Scope

Good first contributions: additional `Verifier` strategies, additional PII
redaction shapes (with synthetic test data only — never real PII), an ANN-backed
`store.Store`, a Redis backend in a nested module.
