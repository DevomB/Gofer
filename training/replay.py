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


def _is_header(line: str) -> bool:
    return '"type":"header"' in line.replace(" ", "")


def _write_line(fout, line: str) -> None:
    fout.write(line if line.endswith("\n") else line + "\n")


def _iter_nonempty(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line


def _should_skip_header(line: str, dst_has_content: bool, skip_header: bool) -> bool:
    return skip_header and dst_has_content and _is_header(line)


def append_jsonl(src: Path, dst: Path, *, skip_header_if_dst: bool = True) -> int:
    """Stream-copy lines from src into dst. Returns number of lines appended."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    dst_has_content = dst.exists() and dst.stat().st_size > 0
    mode = "a" if dst_has_content else "w"
    with dst.open(mode, encoding="utf-8") as fout:
        for line in _iter_nonempty(src):
            if _should_skip_header(line, dst_has_content, skip_header_if_dst):
                continue
            if skip_header_if_dst and _is_header(line):
                dst_has_content = True
            _write_line(fout, line)
            appended += 1
    return appended


def trim(path: Path, max_lines: int = 50000) -> int:
    """Keep the tail of path (FIFO). Returns lines removed."""
    if not path.exists():
        return 0
    lines = list(_iter_nonempty(path))
    if len(lines) <= max_lines:
        return 0
    removed = len(lines) - max_lines
    header = lines[0] if lines and _is_header(lines[0]) else None
    tail = lines[-max_lines:]
    if header and (not tail or not _is_header(tail[0])):
        body = [ln for ln in tail if not _is_header(ln)]
        tail = [header] + body[-(max_lines - 1) :]
    with path.open("w", encoding="utf-8") as f:
        f.writelines(tail)
    return removed
