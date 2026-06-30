# Gofer

Go engine inspired by serious computer-Go research ([Wu 2020](https://arxiv.org/abs/1902.10565)). Not KataGo.

- **Docs:** [`docs/`](docs/) — start with [`implementation-blueprint.md`](docs/implementation-blueprint.md)
- **Module:** `github.com/DevomB/gofer`
- **Rules:** Chinese (primary), Tromp-Taylor, positional superko wrapper

```bash
make test
make bench
make build
```

GTP mode:

```bash
go run ./cmd/gofer -gtp -gtp-playouts 400 -eval heuristic
```
