"""Subprocess layer: logged commands, background jobs, inference sidecars.

The runner talks to the outside world only through ``Executor`` so tests can
substitute a fake that writes the expected artifacts without Go or PyTorch.
"""

from __future__ import annotations

import os
import platform
import shlex
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

ORT_VERSION = "1.26.0"


class CommandError(RuntimeError):
    pass


class Job(Protocol):
    def wait(self) -> None: ...
    def poll(self) -> bool: ...  # True when finished
    def terminate(self) -> None: ...


class Executor(Protocol):
    def run(self, cmd: list[str], *, log: Path, env: dict[str, str] | None = None) -> None: ...
    def spawn(self, cmd: list[str], *, log: Path, env: dict[str, str] | None = None) -> Job: ...
    def sidecar(self, python: str, model: Path, port: int, *, log: Path) -> Job: ...


@dataclass
class _ProcJob:
    proc: subprocess.Popen
    cmd: list[str]
    log: Path
    cleanup: list[Job] = field(default_factory=list)

    def poll(self) -> bool:
        return self.proc.poll() is not None

    def wait(self) -> None:
        try:
            code = self.proc.wait()
        except BaseException:
            # Ctrl-C, SIGTERM, spot preemption: the child does not get the signal
            # when we are the ones being interrupted, so stop it explicitly
            # instead of leaving a trainer on the GPU or an arena still playing.
            self.terminate()
            raise
        finally:
            for c in self.cleanup:
                c.terminate()
        if code != 0:
            raise CommandError(f"exit {code}: {shlex.join(self.cmd)} (log: {self.log})\n{tail(self.log)}")

    def terminate(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


class SubprocessExecutor:
    """Runs commands from the repo root, appending stdout+stderr to a log file."""

    def __init__(self, cwd: Path, echo: bool = True) -> None:
        self.cwd = cwd
        self.echo = echo

    def _open(self, cmd: list[str], log: Path, env: dict[str, str] | None) -> _ProcJob:
        log.parent.mkdir(parents=True, exist_ok=True)
        if self.echo:
            print(f"  $ {shlex.join(cmd)}", flush=True)
        full_env = {**os.environ, **(env or {})}
        fh = log.open("a", encoding="utf-8")
        fh.write(f"\n$ {shlex.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=self.cwd, stdout=fh, stderr=subprocess.STDOUT, env=full_env)
        fh.close()  # the child holds its own handle
        return _ProcJob(proc, cmd, log)

    def run(self, cmd: list[str], *, log: Path, env: dict[str, str] | None = None) -> None:
        self._open(cmd, log, env).wait()

    def spawn(self, cmd: list[str], *, log: Path, env: dict[str, str] | None = None) -> Job:
        return self._open(cmd, log, env)

    def sidecar(self, python: str, model: Path, port: int, *, log: Path, timeout: float = 60) -> _ProcJob:
        job = self._open([python, "training/inference_server.py", "--model", str(model), "--port", str(port)], log, None)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if job.poll():
                raise CommandError(f"sidecar on :{port} exited early (log: {log})\n{tail(log)}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                    return job
            except OSError:
                time.sleep(0.5)
        job.terminate()
        raise CommandError(f"sidecar on :{port} not healthy after {timeout}s (log: {log})")


def ensure_ort_library(cache_dir: Path) -> Path:
    """Download the pinned ONNX Runtime shared library (Linux x64 / Windows x64)."""
    system = platform.system()
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        raise CommandError(f"no pinned ORT download for {system}/{machine}; set engine.ort_lib")
    if system == "Linux":
        name, lib_rel = f"onnxruntime-linux-x64-{ORT_VERSION}", f"lib/libonnxruntime.so.{ORT_VERSION}"
        url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{name}.tgz"
    elif system == "Windows":
        name, lib_rel = f"onnxruntime-win-x64-{ORT_VERSION}", "lib/onnxruntime.dll"
        url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{name}.zip"
    else:
        raise CommandError(f"no pinned ORT download for {system}; set engine.ort_lib")
    lib = cache_dir / name / lib_rel
    if lib.exists():
        return lib
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / url.rsplit("/", 1)[1]
    print(f"  downloading ONNX Runtime {ORT_VERSION}: {url}", flush=True)
    urllib.request.urlretrieve(url, archive)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(cache_dir)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(cache_dir, filter="data")
    if not lib.exists():
        raise CommandError(f"ORT archive did not contain {lib_rel}")
    return lib

