#!/usr/bin/env bash
# Copy a run dir between this machine and a remote box over SSH (rsync), then
# rebuild the dashboard locally. Works with any provider that gives you SSH
# (RunPod, Vast, Lambda, Lightsail, a home server).
#
#   bash infra/cloud/sync-run.sh pull user@host:/root/Gofer gpu     # remote -> local, then report
#   bash infra/cloud/sync-run.sh push user@host:/root/Gofer gpu     # local -> remote (resume elsewhere)
#   SSH_PORT=22022 bash infra/cloud/sync-run.sh pull root@1.2.3.4:/workspace/Gofer gpu
#
# Shards and champion files are immutable, so repeated pulls are incremental.
set -euo pipefail

dir="${1:?pull|push}"
remote="${2:?user@host:/path/to/repo}"
run="${3:-gpu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ssh_opts=(-e "ssh -p ${SSH_PORT:-22}")

case "$dir" in
  pull)
    mkdir -p "$ROOT/runs/$run"
    rsync -az --info=progress2 "${ssh_opts[@]}" "$remote/runs/$run/" "$ROOT/runs/$run/"
    (cd "$ROOT" && python -m training.pipeline report --set "run.name=\"$run\"")
    ;;
  push)
    rsync -az --info=progress2 "${ssh_opts[@]}" "$ROOT/runs/$run/" "$remote/runs/$run/"
    ;;
  *)
    echo "usage: $0 pull|push user@host:/path run-name" >&2
    exit 2
    ;;
esac
