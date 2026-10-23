"""Pytest wrapper for the plain-Node docs/ page tests (see tests/frontend_pages.test.js)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_JS = ROOT / "tests" / "frontend_pages.test.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_docs_pages_behaviors() -> None:
    result = subprocess.run(
        ["node", str(TEST_JS)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, "docs/ 頁面行為測試失敗：" + result.stdout + result.stderr
    assert "ALL DOCS TESTS PASS" in result.stdout
