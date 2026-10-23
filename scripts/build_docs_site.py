"""Build the static documentation site in docs/ from existing outputs.

Reads only existing artifacts (figures/, results/, report/) and copies or
transcribes them into docs/. No numbers are invented and no model is trained.

Outputs:
    docs/figures/*.png                   Verbatim copies of figures/*.png
    docs/results.json                    Values read directly from results/
    docs/report/*.html, paper.css        Copied / converted reports
    docs/index.html                      Homepage rendered from results.json
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.md_to_html import render, strip_front_matter

FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"
DOCS_DIR = ROOT / "docs"
DOCS_FIGURES = DOCS_DIR / "figures"
DOCS_REPORT = DOCS_DIR / "report"

IRIS_METRICS = RESULTS_DIR / "metrics.json"
AI4I_METRICS = RESULTS_DIR / "ai4i_metrics.json"
IRIS_EDA = RESULTS_DIR / "eda_summary.md"
AI4I_EDA = RESULTS_DIR / "ai4i_eda_summary.md"
BASELINES_MD = REPORT_DIR / "ai4i_baselines.md"


def copy_figures() -> list[str]:
    DOCS_FIGURES.mkdir(parents=True, exist_ok=True)
    names = []
    for png in sorted(FIGURES_DIR.glob("*.png")):
        shutil.copy2(png, DOCS_FIGURES / png.name)
        names.append(png.name)
    print(f"Copied {len(names)} figures -> {DOCS_FIGURES}")
    return names


def build_results_json(figure_names: list[str]) -> dict:
    iris = json.loads(IRIS_METRICS.read_text(encoding="utf-8"))
    ai4i = json.loads(AI4I_METRICS.read_text(encoding="utf-8"))
    payload = {
        "iris": iris,
        "ai4i": ai4i,
        "iris_eda_markdown": IRIS_EDA.read_text(encoding="utf-8"),
        "ai4i_eda_markdown": AI4I_EDA.read_text(encoding="utf-8"),
        "figures": figure_names,
        "sources": {
            "iris": "results/metrics.json",
            "ai4i": "results/ai4i_metrics.json",
            "eda": ["results/eda_summary.md", "results/ai4i_eda_summary.md"],
        },
    }
    (DOCS_DIR / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {DOCS_DIR / 'results.json'}")
    return payload


def copy_reports() -> None:
    DOCS_REPORT.mkdir(parents=True, exist_ok=True)
    for html_file in sorted(REPORT_DIR.glob("*.html")):
        shutil.copy2(html_file, DOCS_REPORT / html_file.name)
        print(f"Copied {html_file.name} -> docs/report/")
    shutil.copy2(REPORT_DIR / "paper.css", DOCS_REPORT / "paper.css")

    md_text = BASELINES_MD.read_text(encoding="utf-8")
    body, meta = strip_front_matter(md_text)
    html_text = render(body, meta)
    (DOCS_REPORT / "ai4i_baselines.html").write_text(html_text, encoding="utf-8")
    print("Converted ai4i_baselines.md -> docs/report/ai4i_baselines.html")


def build_index() -> None:
    """Homepage (docs/index.html) and inference.html are hand-maintained; keep them."""


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_names = copy_figures()
    build_results_json(figure_names)
    copy_reports()


if __name__ == "__main__":
    main()
