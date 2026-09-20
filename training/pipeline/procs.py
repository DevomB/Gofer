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
    # For short commands whose stdout IS the answer, rather than a log to tail.
    def capture(self, cmd: list[str], *, env: dict[str, str] | None = None) -> str: ...


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

    def capture(self, cmd: list[str], *, env: dict[str, str] | None = None) -> str:
        full_env = {**os.environ, **(env or {})}
        r = subprocess.run(cmd, cwd=self.cwd, capture_output=True, text=True, env=full_env, check=False)
        if r.returncode != 0:
            raise CommandError(f"{cmd[0]} exited {r.returncode}: {(r.stderr or r.stdout).strip()[:200]}")
        return r.stdout

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


# Release archives published for the pinned ONNX Runtime, by (system, architecture).
# ARM matters here: Apple Silicon is what most contributors have, and Graviton is
# the cheapest CPU capacity on AWS, so hard-coding x86-64 would rule out both.
ORT_BUILDS = {
    ("Linux", "x86_64"): (f"onnxruntime-linux-x64-{ORT_VERSION}", "tgz", f"lib/libonnxruntime.so.{ORT_VERSION}"),
    ("Linux", "aarch64"): (f"onnxruntime-linux-aarch64-{ORT_VERSION}", "tgz", f"lib/libonnxruntime.so.{ORT_VERSION}"),
    ("Darwin", "arm64"): (f"onnxruntime-osx-arm64-{ORT_VERSION}", "tgz", f"lib/libonnxruntime.{ORT_VERSION}.dylib"),
    ("Windows", "x86_64"): (f"onnxruntime-win-x64-{ORT_VERSION}", "zip", "lib/onnxruntime.dll"),
}
# platform.machine() spellings that mean the same architecture.
ORT_ARCH_ALIASES = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}


def ort_build_for(system: str, machine: str) -> tuple[str, str, str] | None:
    arch = ORT_ARCH_ALIASES.get(machine.lower(), machine.lower())
    if system == "Darwin" and arch == "aarch64":
        arch = "arm64"                      # the macOS archive is named arm64
    if system == "Windows":
        arch = "x86_64" if arch in ("x86_64", "aarch64") else arch
    return ORT_BUILDS.get((system, arch))


def ensure_ort_library(cache_dir: Path) -> Path:
    """Download the pinned ONNX Runtime shared library for this platform."""
    system = platform.system()
    machine = platform.machine()
    build = ort_build_for(system, machine)
    if build is None:
        raise CommandError(
            f"ONNX Runtime {ORT_VERSION} publishes no build for {system}/{machine}, so the "
            f"in-process backend cannot run here. Either set engine.ort_lib to a local "
            f"libonnxruntime, or use the sidecar backend, which runs anywhere Python "
            f"onnxruntime installs:\n"
            f"  python -m training.pipeline run --config <cfg> --set 'engine.backend=\"sidecar\"'"
        )
    name, kind, lib_rel = build
    url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{name}.{kind}"
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

