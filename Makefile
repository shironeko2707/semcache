.PHONY: all test vet build bench bench-hit example tidy

all: vet test

test:
	go test ./...

vet:
	go vet ./...

build:
	go build ./...

# Full benchmark: hit-rate/correctness exit criterion (verbose) + hot-lookup micro-bench.
bench:
	go test ./bench/ -run TestHitRateExitCriterion -v
	go test ./bench/ -run x -bench BenchmarkLookupHot -benchmem

# Just the headline hit-rate / false-hit exit criterion.
bench-hit:
	go test ./bench/ -run TestHitRateExitCriterion -v

example:
	go run ./examples

tidy:
	go mod tidy
