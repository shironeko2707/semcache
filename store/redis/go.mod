module github.com/shironeko2707/semcache/store/redis

go 1.26

require (
	github.com/redis/go-redis/v9 v9.20.1
	github.com/shironeko2707/semcache v0.2.0
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	go.uber.org/atomic v1.11.0 // indirect
)

// Develop against the in-repo parent. This replace is ignored by consumers (Go
// only applies the main module's replaces), so `go get` of this nested module
// still resolves the parent at its required version.
replace github.com/shironeko2707/semcache => ../../
