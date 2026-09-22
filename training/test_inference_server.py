from __future__ import annotations

import re
from pathlib import Path

from inference_server import SCHEMA_VERSION


def test_sidecar_schema_matches_go_features():
    # The sidecar refuses any request whose schema_version differs from its own,
    # so a mismatch fails every evaluation and the engine falls back to the heuristic.
    go = (Path(__file__).resolve().parents[1] / "cmd" / "gofer" / "features.go").read_text()
    go_version = int(re.search(r"FeatureSchemaVersion\s*=\s*(\d+)", go).group(1))
    assert SCHEMA_VERSION == go_version
