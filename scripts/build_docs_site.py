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


INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML 分析結果（靜態網站）</title>
<style>
  body { font-family: "Noto Sans CJK TC", "PingFang TC", "Microsoft JhengHei", Arial, sans-serif;
         max-width: 900px; margin: 0 auto; padding: 2rem 1rem; line-height: 1.7; color: #222; }
  h1 { color: #1f3d55; border-bottom: 2px solid #315b7d; padding-bottom: .4rem; }
  h2 { color: #315b7d; margin-top: 2rem; }
  table { border-collapse: collapse; width: 100%; font-size: .92rem; margin: .8rem 0; }
  th, td { border: 1px solid #d7e1e8; padding: .45rem .6rem; text-align: left; }
  th { background: #edf3f7; color: #1f3d55; }
  .cards { display: flex; gap: 1rem; flex-wrap: wrap; }
  .card { border: 1px solid #d7e1e8; border-radius: 6px; padding: 1rem 1.2rem; flex: 1 1 200px; background: #fafcfd; }
  .card b { color: #1f3d55; font-size: 1.35rem; }
  .muted { color: #666; font-size: .88rem; }
  .gallery figure { margin: .8rem 0; }
  .gallery img { max-width: 100%; border: 1px solid #e3e8ec; border-radius: 4px; }
  pre.eda { background:#f5f7f9; border:1px solid #e3e8ec; padding: .8rem; white-space: pre-wrap; font-size:.85rem; }
  nav a { margin-right: 1rem; }
</style>
</head>
<body>
<h1>專案分析結果</h1>
<nav>
  <a href="inference.html">互動推論示範</a>
  <a href="report/ai4i2020_prediction_paper.html">AI4I 小論文</a>
  <a href="report/ai4i_baselines.html">AI4I 基準比較</a>
  <a href="report/iris_neural_network_paper.html">Iris 小論文</a>
  <a href="results.json">results.json</a>
</nav>
<p class="muted">本頁為純 HTML + JavaScript，資料全部取自 <code>results.json</code>，其內容逐值複製自 <code>results/</code> 現有產出，未重新訓練、未另行計算。</p>

<section id="ai4i"></section>
<section id="iris"></section>
<h2>資料探索摘要</h2>
<section>
  <h3>Iris</h3><pre class="eda" id="eda-iris">載入中…</pre>
  <h3>AI4I</h3><pre class="eda" id="eda-ai4i">載入中…</pre>
</section>
<h2>圖表</h2>
<div class="gallery" id="gallery"></div>
<p class="muted">以本機瀏覽需先起靜態伺服器：<code>python3 -m http.server -d docs 8000</code></p>

<script>
fetch("results.json")
  .then((r) => r.json())
  .then((data) => {
    const fmt = (v) => Number(v).toFixed(4);

    const ai = data.ai4i;
    const rows = Object.entries(ai.models).map(([name, m]) =>
      "<tr><td>" + name + "</td><td>" + fmt(m.recall) + "</td><td>" + fmt(m.f1) + "</td>" +
      "<td>" + fmt(m.pr_auc) + "</td><td>" + fmt(m.roc_auc) + "</td>" +
      "<td>" + m.confusion_matrix[1][1] + " / " + m.confusion_matrix[1][0] +
      " / " + m.confusion_matrix[0][1] + " / " + m.confusion_matrix[0][0] + "</td></tr>");
    const imp = Object.entries(ai.rf_feature_importance)
      .sort((a, b) => b[1] - a[1]).slice(0, 5)
      .map(([k, v]) => k + " (" + fmt(v) + ")").join("、");
    document.getElementById("ai4i").innerHTML =
      "<h2>AI4I 2020 預測性維護（二元分類基準比較）</h2>" +
      "<p>測試集 " + ai.test_rows.toLocaleString() + " 筆（訓練 " + ai.train_rows.toLocaleString() + "），分層 70/30、seed " +
      ai.random_seed + "，最佳模型：<b>" + ai.best_model_by_f1 + "</b></p>" +
      "<table><tr><th>模型</th><th>Recall</th><th>F1</th><th>PR-AUC</th><th>ROC-AUC</th><th>TP/FN/FP/TN</th></tr>" +
      rows.join("") + "</table>" +
      "<p>Random Forest 特徵重要度 Top 5：" + imp + "</p>" +
      "<div class='cards'>" +
        "<div class='card'>失效比例（訓練）<br><b>" + (ai.positive_class_rate_train * 100).toFixed(2) + "%</b></div>" +
      "</div>";

    const ir = data.iris;
    document.getElementById("iris").innerHTML =
      "<h2>Iris 神經網路分類</h2>" +
      "<div class='cards'>" +
        "<div class='card'>測試準確率<br><b>" + fmt(ir.test_accuracy) + "</b></div>" +
        "<div class='card'>測試損失<br><b>" + fmt(ir.test_loss) + "</b></div>" +
        "<div class='card'>類別分離 Macro AUC<br><b>" + fmt(ir.macro_auc) + "</b></div>" +
        "<div class='card'>5 折交叉驗證<br><b>" + fmt(ir.cv_mean_accuracy) + " ± " + fmt(ir.cv_std_accuracy) + "</b></div>" +
      "</div>" +
      "<p class='muted'>訓練 " + ir.epochs_trained + " epochs、best epoch " + ir.best_epoch +
      "（訓練/測試 = " + ir.train_size + "/" + ir.test_size + "）</p>";

    document.getElementById("eda-iris").textContent = data.iris_eda_markdown;
    document.getElementById("eda-ai4i").textContent = data.ai4i_eda_markdown;

    document.getElementById("gallery").innerHTML = data.figures
      .map((name) => "<figure><img src='figures/" + name + "' alt='" + name + "'><figcaption>" + name + "</figcaption></figure>")
      .join("");
  })
  .catch((e) => {
    document.getElementById("ai4i").innerHTML =
      "<p>results.json 載入失敗：" + e.message + "。請以靜態伺服器開啟：<code>python3 -m http.server -d docs 8000</code></p>";
  });
</script>
</body>
</html>
"""


def write_index() -> None:
    (DOCS_DIR / "index.html").write_text(INDEX_TEMPLATE, encoding="utf-8")
    print(f"Wrote {DOCS_DIR / 'index.html'}")


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_names = copy_figures()
    build_results_json(figure_names)
    copy_reports()
    write_index()


if __name__ == "__main__":
    main()
