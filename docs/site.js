/* 首頁資料渲染：全部數值取自 results.json（逐值複製自 results/ 產出），不做任何計算以外的轉換。 */
(function () {
  const fmt = (v) => Number(v).toFixed(4);
  const pct = (v) => (Number(v) * 100).toFixed(2) + "%";

  function renderIris(d) {
    const ir = d.iris;
    const grid = document.getElementById("iris-metrics");
    grid.innerHTML =
      "<dl>" +
      "<dt>測試準確率</dt><dd>" + fmt(ir.test_accuracy) + "</dd>" +
      "<dt>測試損失</dt><dd>" + fmt(ir.test_loss) + "</dd>" +
      "<dt>類別分離 Macro AUC</dt><dd>" + fmt(ir.macro_auc) + "</dd>" +
      "<dt>5 折交叉驗證</dt><dd>" + fmt(ir.cv_mean_accuracy) + " ± " + fmt(ir.cv_std_accuracy) + "</dd>" +
      "</dl>" +
      "<p class=\"source\">來源：results/metrics.json（訓練/測試 " + ir.train_size + "/" + ir.test_size +
      "、" + ir.epochs_trained + " epochs、best epoch " + ir.best_epoch + "）</p>";
    const cm = ir.confusion_matrix;
    document.getElementById("iris-cm").innerHTML =
      cm.map((row, i) => row.map((v, j) =>
        "<span class=\"cm-cell" + (i === j ? " hit" : "") + "\">" + v + "</span>").join("")
      ).join("");
  }

  function renderAI4I(d) {
    const ai = d.ai4i;
    document.getElementById("ai4i-metrics").innerHTML =
      "<dl>" +
      "<dt>測試規模</dt><dd>" + ai.test_rows.toLocaleString() + " 筆</dd>" +
      "<dt>失效比例（訓練）</dt><dd>" + pct(ai.positive_class_rate_train) + "</dd>" +
      "<dt>最佳模型（F1）</dt><dd>" + ai.best_model_by_f1 + "</dd>" +
      "<dt>最佳 F1 / PR-AUC</dt><dd>" +
        fmt(ai.models[ai.best_model_by_f1].f1) + " / " + fmt(ai.models[ai.best_model_by_f1].pr_auc) + "</dd>" +
      "</dl>" +
      "<p class=\"source\">來源：results/ai4i_metrics.json（分層 70/30、seed " + ai.random_seed + "）</p>";

    const rows = Object.entries(ai.models).map(([name, m]) => {
      const cm = m.confusion_matrix;
      return "<tr><td>" + name + "</td>" +
        "<td>" + fmt(m.recall) + "</td>" +
        "<td>" + fmt(m.f1) + "</td>" +
        "<td><span class=\"bar-cell\"><span class=\"bar\" style=\"width:" + (m.pr_auc * 100).toFixed(1) + "%\"></span></span>" +
          fmt(m.pr_auc) + "</td>" +
        "<td>" + fmt(m.roc_auc) + "</td>" +
        "<td>" + cm[1][1] + " / " + cm[1][0] + " / " + cm[0][1] + " / " + cm[0][0] + "</td></tr>";
    });
    document.getElementById("ai4i-table").innerHTML =
      "<table class=\"models\"><thead><tr><th>模型</th><th>Recall</th><th>F1</th>" +
      "<th>PR-AUC</th><th>ROC-AUC</th><th>TP / FN / FP / TN</th></tr></thead>" +
      "<tbody>" + rows.join("") + "</tbody></table>";

    const imp = Object.entries(ai.rf_feature_importance).sort((a, b) => b[1] - a[1]).slice(0, 6);
    const maxImp = imp[0][1];
    document.getElementById("ai4i-importance").innerHTML = imp.map(([k, v]) =>
      "<li><span class=\"impl-label\">" + k + "</span>" +
      "<span class=\"bar-cell\"><span class=\"bar bar--accent\" style=\"width:" +
      (v / maxImp * 100).toFixed(1) + "%\"></span></span>" + fmt(v) + "</li>").join("");
  }

  function renderEda(d) {
    document.getElementById("eda-iris").textContent = d.iris_eda_markdown;
    document.getElementById("eda-ai4i").textContent = d.ai4i_eda_markdown;
  }

  function renderGallery(d) {
    document.getElementById("gallery").innerHTML = d.figures
      .map((name) =>
        "<figure><a href=\"figures/" + name + "\" target=\"_blank\" rel=\"noopener\">" +
        "<img loading=\"lazy\" src=\"figures/" + name + "\" alt=\"" + name + "\"></a>" +
        "<figcaption>" + name + "</figcaption></figure>").join("");
  }

  function init() {
    return fetch("results.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status))))
      .then((d) => {
        renderIris(d);
        renderAI4I(d);
        renderEda(d);
        renderGallery(d);
        document.querySelectorAll("[data-loaded='later']").forEach((el) => el.classList.add("ready"));
      })
      .catch((e) => {
        const msg = "results.json 載入失敗（" + e.message + "）；請以靜態伺服器開啟：" +
          "<code>python3 -m http.server -d docs 8000</code>";
        document.getElementById("ai4i-metrics").textContent = "";
        document.getElementById("iris-metrics").textContent = "";
        ["ai4i-table", "ai4i-importance", "iris-cm", "gallery"].forEach((id) => {
          document.getElementById(id).innerHTML = "<p class=\"warning\">" + msg + "</p>";
        });
      });
  }

  window.site_init = init;
})();
