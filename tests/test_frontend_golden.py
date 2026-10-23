"""Golden reconciliation: run the browser-tests golden test with node under pytest."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_JS = ROOT / "tests" / "golden_frontend.test.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_frontend_golden_matches_python() -> None:
    result = subprocess.run(
        ["node", str(TEST_JS)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        "Frontend inference deviates from Python golden values:\n"
        + result.stdout
        + result.stderr
    )
    assert "ALL GOLDEN MATCH" in result.stdout
