/* 前端推論黃金測試：與 tests/golden.json（Python 訓練產物計算）比對，
 * 機率誤差門檻 1e-6。若不一致，以三種候選原因的替身實作定位病因：
 * ① 前處理不同步（缺標準化）② 特徵／one-hot 順序錯置 ③ 權重或係數貼錯或貼漏 */
"use strict";
const fs = require("fs");
const path = require("path");
const assert = require("assert");

const ROOT = path.resolve(__dirname, "..");
const golden = JSON.parse(fs.readFileSync(path.join(ROOT, "tests", "golden.json"), "utf-8"));
const params = JSON.parse(fs.readFileSync(path.join(ROOT, "docs", "model_params.json"), "utf-8"));
const app = require(process.env.APP_PATH || path.join(ROOT, "docs", "app.js"));
assert(typeof app.predictIris === "function" && typeof app.predictAI4I === "function", "app.js 未輸出推論函式");

const TOL = golden.tolerance || 1e-6;
const { layers, classes, scaler_mean, scaler_scale } = params.iris_mlp;
const aiMean = params.ai4i_logreg.scaler_mean;
const aiScale = params.ai4i_logreg.scaler_scale;
const coef = params.ai4i_logreg.coef;
const intercept = params.ai4i_logreg.intercept;
const rawVec = (dict) => [dict.sepal_length, dict.sepal_width, dict.petal_length, dict.petal_width];

function forward(x, useScale, swapPairs) {
  const raw = swapPairs ? [x[1], x[0], x[3], x[2]] : x.slice();
  let h = useScale ? raw.map((v, i) => (v - scaler_mean[i]) / scaler_scale[i]) : raw;
  for (const layer of layers) {
    const out = layer.b.slice();
    for (let i = 0; i < h.length; i++)
      for (let j = 0; j < layer.b.length; j++) out[j] += h[i] * layer.W[i][j];
    h = layer === layers[layers.length - 1] ? out : out.map((v) => Math.max(v, 0));
  }
  const max = Math.max(...h);
  const exps = h.map((v) => Math.exp(v - max));
  return exps.map((e) => e / exps.reduce((a, b) => a + b, 0));
}

function ai4iZ(row, useScale, swapOneHot) {
  const nums = [row.air_temperature, row.process_temperature, row.rotational_speed, row.torque, row.tool_wear,
    row.torque * row.rotational_speed, row.process_temperature - row.air_temperature];
  let z = useScale ? nums.map((v, i) => (v - aiMean[i]) / aiScale[i]) : nums.slice();
  const m = row.type === "M" ? 1 : 0, h = row.type === "H" ? 1 : 0;
  return z.concat(swapOneHot ? [h, m] : [m, h]);
}

const logitLinear = (z, offset = 0) => z.reduce((s, v, i) => s + v * coef[i], intercept + offset);
const sigmoid = (s) => 1 / (1 + Math.exp(-s));
const maxDiff = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

function diagnoseIris(index, actual, expected) {
  const truth = classes.map((name) => expected[name]);
  const raw = rawVec(golden.iris[index].inputs);
  console.error(`\n[iris[${index}]] app.js 與 golden 不一致，maxDiff = ${Math.abs(maxDiff(actual, truth)).toExponential(2)}`);
  const control = forward(raw, true, false);
  console.error(`  控制組：以金樣（model_params.json）正確管線重算 → maxDiff vs golden = ${maxDiff(control, truth).toExponential(2)}` +
    (maxDiff(control, truth) < TOL ? "（金樣與權重自洽）" : "（金樣或 model_params.json 本身過期 → ③）"));
  const hypotheses = [
    { label: "① 前處理不同步（缺標準化）", value: forward(raw, false, false) },
    { label: "② 特徵順序錯置（成對交換）", value: forward(raw, true, true) },
    { label: "② 類別順序錯置（循環移位）", value: [control[1], control[2], control[0]] },
  ];
  for (const v of hypotheses) {
    const d = maxDiff(actual, v.value);
    console.error(`  ${v.label}: 預測輸出 maxDiff = ${d.toExponential(2)}` + (d < TOL ? "  ← 符合實作輸出，判定為此原因" : ""));
  }
  if (hypotheses.some((v) => maxDiff(actual, v.value) < TOL)) {
    console.error("  └ 瀏覽器側原因定位如上；依查明的假說修正 app.js 或 golden 之來源。");
  } else {
    console.error("  └ 判定：③ 權重或係數貼錯／貼漏（前處理與順序的替身皆不吻合，而正確管線重算等於 golden）");
  }
}

function diagnoseAI4I(index, actual, expected) {
  const row = golden.ai4i[index].inputs;
  const truth = expected.failure;
  console.error(`\n[ai4i[${index}]] app.js 與 golden 不一致，p = ${actual.toFixed(6)}，golden = ${truth}`);
  const control = logitLinear(ai4iZ(row, true, false));
  console.error(`  控制組：正確管線 → logit ${control.toFixed(6)}，p = ${sigmoid(control).toFixed(6)}` +
    (Math.abs(sigmoid(control) - truth) < TOL ? "（金色與係數自洽）" : "（金色或 model_params.json 過期 → ③）"));
  const actualLogit = Math.log(actual / (1 - actual));
  const candidates = [
    { label: "① 前處理不同步（缺標準化）", logit: logitLinear(ai4iZ(row, false, false)) },
    { label: "② one-hot 順序錯置（M/H 互換）", logit: logitLinear(ai4iZ(row, true, true)) },
  ];
  for (const c of candidates) {
    const d = Math.abs(actualLogit - c.logit);
    console.error(`  ${c.label}: logit = ${c.logit.toFixed(4)}（p = ${sigmoid(c.logit).toFixed(6)}），實作 logit 差 = ${d.toFixed(4)}` +
      (d < 1e-3 ? "  ← 接近，判定為此原因" : ""));
  }
  if (candidates.every((c) => Math.abs(actualLogit - c.logit) >= 1e-3)) {
    console.error("  └ 判定：③ 權重或係數貼錯／貼漏（前處理與順序假說皆不符；金樣與係數自洽）");
  } else {
    const best = candidates.reduce((a, b) => (Math.abs(actualLogit - b.logit) < Math.abs(actualLogit - a.logit) ? b : a));
    console.error(`  └ 判定：與實作 logit 最接近者為『${best.label}』（銜接建議：先確認 app.js 對應的標準化／one-hot 常數）`);
  }
}

let failed = 0;

golden.iris.forEach((g, i) => {
  const raw = rawVec(g.inputs);
  const actual = app.predictIris(...raw);
  const truth = classes.map((name) => g.probabilities[name]);
  const diff = maxDiff(actual, truth);
  if (diff < TOL) {
    console.log(`iris[${i}] OK  class=${classes[actual.indexOf(Math.max(...actual))]}  maxDiff=${diff.toExponential(2)}`);
  } else {
    failed += 1;
    diagnoseIris(i, actual, g.probabilities);
  }
});

golden.ai4i.forEach((g, i) => {
  const r = g.inputs;
  const actual = app.predictAI4I(r.air_temperature, r.process_temperature, r.rotational_speed, r.torque, r.tool_wear, r.type).failure;
  const diff = Math.abs(actual - g.probabilities.failure);
  if (diff < TOL) {
    console.log(`ai4i[${i}] OK  prob=${actual.toFixed(6)}  maxDiff=${diff.toExponential(2)}`);
  } else {
    failed += 1;
    diagnoseAI4I(i, app.predictAI4I(r.air_temperature, r.process_temperature, r.rotational_speed, r.torque, r.tool_wear, r.type).failure, g.probabilities);
  }
});

if (failed) {
  console.error(`\nFAILED: ${failed} 個黃金案例不一致（假說診斷如上：① 前處理 ② 順序 ③ 貼值）`);
  process.exit(1);
} else {
  console.log(`\nALL GOLDEN MATCH（機率誤差 < ${TOL}）`);
}
