const path = require("path");

const ROOT = path.resolve(__dirname, "..");

function freshDom() {
  const els = new Map();
  const make = () => ({
    innerHTML: "",
    textContent: "",
    value: "",
    style: Object.create(null),
    listeners: {},
    classList: { add() {}, remove() {} },
    addEventListener(event, handler) {
      this.listeners[event] = handler;
    },
  });
  global.document = {
    getElementById(id) {
      if (!els.has(id)) els.set(id, make());
      return els.get(id);
    },
    querySelectorAll() {
      return [];
    },
  };
  global.window = {};
  return els;
}

function loadOnce(rel) {
  const abs = require.resolve(path.join(ROOT, rel));
  delete require.cache[abs];
  return require(abs);
}

const FIXED_RESULTS = {
  iris: {
    test_accuracy: 0.9555555555555556,
    test_loss: 0.24918960967,
    macro_auc: 0.9903846153846154,
    cv_mean_accuracy: 0.9666666666666667,
    cv_std_accuracy: 0.03333333333333333,
    train_size: 105,
    test_size: 45,
    epochs_trained: 600,
    best_epoch: 598,
    confusion_matrix: [[16, 0], [1, 29]],
  },
  ai4i: {
    test_rows: 3000,
    positive_class_rate_train: 0.034089046103896106,
    best_model_by_f1: "random_forest",
    random_seed: 42,
    models: {
      logistic_regression: {
        recall: 0.8235294117647058,
        f1: 0.301616073936215,
        pr_auc: 0.42423477190616834,
        roc_auc: 0.9170957394799424,
        confusion_matrix: [[100, 247], [132, 2721]],
      },
      random_forest: {
        recall: 0.7156862745098039,
        f1: 0.8021982537681584,
        pr_auc: 0.8495653145498467,
        roc_auc: 0.9770364608838641,
        confusion_matrix: [[100, 247], [54, 2799]],
      },
    },
    rf_feature_importance: { torque: 0.28, torque_speed_product: 0.2 },
  },
  iris_eda_markdown: "iris eda",
  ai4i_eda_markdown: "ai4i eda",
  figures: ["confusion_matrix.png"],
};

let failed = 0;
const fail = (msg) => {
  console.error("FAIL " + msg);
  failed = 1;
};
const expect = (cond, msg) => {
  if (!cond) fail(msg);
};

(async () => {
  let els = freshDom();
  global.fetch = () =>
    Promise.resolve({ ok: true, json: () => Promise.resolve(FIXED_RESULTS) });
  loadOnce("docs/site.js");
  await window.site_init();
  const metrics = els.get("iris-metrics").innerHTML;
  expect(metrics.includes("測試準確率"), "首頁 iris 面板缺『測試準確率』指標");
  expect(metrics.includes("測試損失"), "首頁 iris 面板缺『測試損失』指標");
  expect(metrics.includes("Macro AUC"), "首頁 iris 面板缺『Macro AUC』指標");
  expect(metrics.includes("0.9556"), "『測試準確率』未顯示 0.9556");
  expect(metrics.includes("0.2492"), "『測試損失』未顯示 0.2492");
  expect(metrics.includes("0.9904"), "『Macro AUC』未顯示 0.9904");

  els = freshDom();
  loadOnce("docs/app.js");
  for (const [id, value] of [
    ["i-sepal-length", "5.1"],
    ["i-sepal-width", "3.5"],
    ["i-petal-length", "1.4"],
    ["i-petal-width", "0.2"],
  ]) {
    document.getElementById(id).value = value;
  }
  const handler = els.get("i-run").listeners.click;
  expect(typeof handler === "function", "i-run 沒有綁 click 事件");
  handler();
  const pred = els.get("result-pred").textContent;
  expect(pred.includes("setosa"), "預測 5.1/3.5/1.4/0.2 應顯示 setosa，實際：" + pred);
  expect(els.get("result") !== undefined, "結果區不存在");
  expect(els.get("result").style.display === "block", "結果區未顯示");

  els = freshDom();
  global.fetch = () => Promise.resolve({ ok: false, status: 404 });
  loadOnce("docs/site.js");
  await window.site_init();
  const warn = els.get("ai4i-table").innerHTML;
  expect(warn.includes("results.json 載入失敗（HTTP 404）"), "fetch 失敗未顯示包含 HTTP 404 的錯誤訊息");
  expect(warn.includes("warning"), "錯誤訊息未使用 warning 樣式");

  if (failed) {
    console.error("DOCS TESTS FAILED");
    process.exit(1);
  }
  console.log("ALL DOCS TESTS PASS");
})();
