.PHONY: test bench race lint profile memprofile pgo-build pgo-profile build build-onnx bench-check reproduce-9x9-baseline reproduce-9x9-onnx-gate sidecar train-bootstrap export-onnx

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

build-onnx:
	CGO_ENABLED=1 go build -tags=onnx -o bin/gofer ./cmd/gofer

build:
	go build -o bin/gofer ./cmd/gofer

play:
	go run ./cmd/gofer -play -size 9

analyze:
	go run ./cmd/gofer -analyze -size 9 -playouts 200

sidecar:
	python training/inference_server.py --model models/gofer-9x9-bootstrap.onnx --port 8080

export-onnx:
	python training/export_onnx.py --out models/gofer-9x9-bootstrap.onnx

train-bootstrap:
	go run ./cmd/gofer -selfplay -games 100 -size 9 -playouts 100 -eval heuristic -o training/data/samples.jsonl -seed 42
	python training/train_bootstrap.py --data training/data/samples.jsonl --epochs 20
	python training/export_onnx.py --checkpoint training/checkpoints/best.pt --out models/gofer-9x9-bootstrap.onnx

reproduce-9x9-baseline:
	go test ./... -count=1
	go run ./cmd/bench -baseline .tectonix/reports/bench-regression.json -check
	go run ./cmd/gofer -arena -games 200 -size 9 -playouts 400 -black-playouts 600 -white-playouts 200 -black-eval heuristic -white-eval heuristic -seed 42 -arena-enhanced baseline -json .tectonix/reports/arena-9x9-baseline.json

reproduce-9x9-onnx-gate:
	go test ./... -count=1
	go run ./cmd/bench -baseline .tectonix/reports/bench-regression.json -check
	@test -f models/gofer-9x9-bootstrap.onnx || (echo "run: make export-onnx" && exit 1)
	go run ./cmd/gofer -arena -games 200 -size 9 -playouts 400 -black-eval heuristic -white-eval onnx -seed 42 -arena-enhanced none -json .tectonix/reports/arena-9x9-onnx-v25.json
