.PHONY: test bench race lint profile pgo-build pgo-profile build bench-check

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
	go test -cpuprofile=.tectonix/reports/cpu.prof -bench=BenchmarkLegalMoves -benchtime=3s ./cmd/gofer/

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
