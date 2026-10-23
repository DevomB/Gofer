PYTHON ?= .venv/bin/python
MODEL  ?= models/gofer-9x9.onnx
RUN    ?= runs/9x9

.PHONY: build build-nocgo test test-go test-py race lint bench gtp play train ort site

# Engine with ONNX Runtime (needs CGO and scripts/install-ort.sh).
build:
	CGO_ENABLED=1 go build -tags onnx -o bin/gofer ./cmd/gofer

# Pure-Go build: rules, search, GTP and arena work; network inference does not.
build-nocgo:
	CGO_ENABLED=0 go build -o bin/gofer ./cmd/gofer

test: test-go test-py

test-go:
	go test -short ./...

test-py:
	$(PYTHON) -m pytest -q training

race:
	go test -race -short ./...

lint:
	@test -z "$$(gofmt -l cmd internal)" || (gofmt -l cmd internal; exit 1)
	go vet ./...
	CGO_ENABLED=1 go vet -tags onnx ./...

bench:
	go test -run '^$$' -bench . -benchmem ./internal/game ./internal/mcts
	bin/gofer bench -model $(MODEL)

gtp: build
	bin/gofer gtp -model $(MODEL)

play: build
	bin/gofer play -model $(MODEL)

train: build
	$(PYTHON) -m training.loop --run $(RUN) --gofer bin/gofer

ort:
	scripts/install-ort.sh

# Static site in web/: compiled CSS + the engine as WebAssembly (outputs are committed).
site:
	scripts/build-site.sh
