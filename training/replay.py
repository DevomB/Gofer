"""FIFO replay buffer for self-play JSONL samples."""

from __future__ import annotations

from pathlib import Path


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def append_jsonl(src: Path, dst: Path, *, skip_header_if_dst: bool = True) -> int:
    """Stream-copy lines from src into dst. Returns number of lines appended."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    dst_has_content = dst.exists() and dst.stat().st_size > 0
    with src.open(encoding="utf-8") as fin:
        mode = "a" if dst_has_content else "w"
        with dst.open(mode, encoding="utf-8") as fout:
            for line in fin:
                stripped = line.strip()
                if not stripped:
                    continue
                if skip_header_if_dst and dst_has_content and '"type":"header"' in stripped.replace(" ", ""):
                    continue
                if skip_header_if_dst and not dst_has_content and '"type":"header"' in stripped.replace(" ", ""):
                    fout.write(line if line.endswith("\n") else line + "\n")
                    appended += 1
                    dst_has_content = True
                    continue
                fout.write(line if line.endswith("\n") else line + "\n")
                appended += 1
    return appended


def trim(path: Path, max_lines: int = 50000) -> int:
    """Keep the tail of path (FIFO). Returns lines removed."""
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    if len(lines) <= max_lines:
        return 0
    removed = len(lines) - max_lines
    header = lines[0] if lines and '"type":"header"' in lines[0].replace(" ", "") else None
    tail = lines[-max_lines:]
    if header and (not tail or tail[0] != header):
        body = [ln for ln in tail if '"type":"header"' not in ln.replace(" ", "")]
        tail = [header] + body[-(max_lines - 1) :]
    with path.open("w", encoding="utf-8") as f:
        f.writelines(tail)
    return removed
