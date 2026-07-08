# Net-size ablation snapshot

`net-size-replay-snapshot.jsonl` is a **frozen copy** of production
`training/data/replay.jsonl` (50k rows, captured Jul 2026 pre–Piece-1-cap-randomization
cycles). **Do not append** new self-play data here.

Used only by `training/net_size_ablation.py`. Findings and decision:
`docs/decisions/0005-net-size-ablation.md`.

To refresh the snapshot for a future re-run (only when intentionally starting a
new ablation cohort):

```bash
cp training/data/replay.jsonl training/data/ablation/net-size-replay-snapshot.jsonl
```

The snapshot file is large (~130 MB) and is not committed to git.
