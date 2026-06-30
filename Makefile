.PHONY: test bench race lint profile memprofile pgo-build pgo-profile build bench-check reproduce-9x9-baseline

test:
	go test ./...

bench:
	go run ./cmd/bench

bench-check:
	go run ./cmd/bench -baseline .tectonix/reports/bench-regression.json -check

race:
	go test -race ./...

lint:
	go vet ./...

profile:
	go test -cpuprofile=.tectonix/reports/legalmoves-cpu.prof -bench=BenchmarkLegalMoves -benchtime=3s ./cmd/gofer/

memprofile:
	go test -memprofile=.tectonix/reports/legalmoves-mem.prof -bench=BenchmarkLegalMoves -benchtime=3s ./cmd/gofer/

pgo-profile:
	go test -cpuprofile=default.pgo -bench=BenchmarkLegalMoves -benchtime=10s ./cmd/gofer/

pgo-build:
	@test -f default.pgo || (echo "run: make pgo-profile" && exit 1)
	go build -pgo=default.pgo -o bin/gofer ./cmd/gofer

build:
	go build -o bin/gofer ./cmd/gofer

play:
	go run ./cmd/gofer -play -size 9

analyze:
	go run ./cmd/gofer -analyze -size 9 -playouts 200

reproduce-9x9-baseline:
	go test ./... -count=1
	go run ./cmd/bench -baseline .tectonix/reports/bench-regression.json -check
	go run ./cmd/gofer -arena -games 200 -size 9 -playouts 400 -black-playouts 600 -white-playouts 200 -black-eval heuristic -white-eval heuristic -seed 42 -json .tectonix/reports/arena-9x9-baseline.json
