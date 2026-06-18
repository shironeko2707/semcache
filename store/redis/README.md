# semcache Redis backend

A [RediSearch](https://redis.io/docs/latest/develop/interact/search-and-query/)-backed
`store.Store` for [semcache](../../). It lives in its own Go module so the
semcache root stays **zero-dependency** — you only pull in `go-redis` if you opt
into this backend.

## Why a separate module

The default in-memory store is a flat cosine scan: exact, simple, and fine into
the ~10k-entry range. When you need a shared cache across processes, persistence,
or a larger index, point semcache at Redis instead:

- **Vector KNN** via a RediSearch cosine index (FLAT or HNSW).
- **Server-side TTL** — entry expiry is a Redis key TTL, enforced by the server.
- **Namespace isolation** — every lookup is filtered by a namespace tag.

## Requirements

Redis with the **RediSearch** module — i.e. [Redis Stack](https://redis.io/docs/latest/operate/oss_and_stack/).

```bash
docker compose up -d   # see docker-compose.yml (starts redis-stack on :6379)
```

## Install

```bash
go get github.com/shironeko2707/semcache/store/redis
```

## Usage

```go
import (
    "github.com/shironeko2707/semcache"
    redisstore "github.com/shironeko2707/semcache/store/redis"
)

rs, err := redisstore.New(ctx, redisstore.Config{
    Addr: "localhost:6379",
    Dim:  256,        // MUST match your Embedder's output dimension
    // UseHNSW: true, // approximate index for large datasets; FLAT (exact) by default
})
if err != nil { /* ... */ }
defer rs.Close()

c, _ := semcache.New(embedder, semcache.WithStore(rs))
```

`Dim` is fixed at index creation. To change it, drop the index (`rs.DropIndex`)
and recreate the store.

## Tests

Integration tests run against a live Redis Stack and **skip** (not fail) when none
is reachable:

```bash
docker compose up -d
REDIS_ADDR=localhost:6379 go test ./...
```

## Notes

- Payloads are stored base64-encoded; the cache layer owns (de)serialization, so
  this backend never needs to know the cache's value type.
- Vectors are stored L2-normalized with the COSINE metric; similarity is
  `1 - distance`.
- Namespaces are RediSearch tag values; avoid commas in namespace names.
