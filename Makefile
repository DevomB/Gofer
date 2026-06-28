.PHONY: test bench race lint profile pgo-build build

test:
	go test ./...

bench:
	go run ./cmd/bench

race:
	go test -race ./...

lint:
	go vet ./...

profile:
	go test -cpuprofile=.tectonix/reports/cpu.prof -bench=BenchmarkLegalMoves -benchtime=3s ./cmd/gofer/

pgo-build:
	@test -f default.pgo || (echo "run: go test -cpuprofile=default.pgo -bench=BenchmarkPlayGame -benchtime=10s ./..." && exit 1)
	go build -pgo=default.pgo -o bin/gofer ./cmd/gofer

build:
	go build -o bin/gofer ./cmd/gofer
