# Code Smell 報告

- 產生時間：2026-10-23／產生指令：`npm run static-check`（本次為 docs/ 與 scripts/ 範圍重跑摘錄）
- 範圍：`docs/` 與 `scripts/`（依需求限定）；工具：ESLint 10.13.0 ＋ eslint-plugin-sonarjs 4.2.2、jscpd 5.4.2、knip 5.88.1
- **此報告僅陳述檢查結果，未修改任何程式碼**

## 1. 各項警告總數（依規則分組）

| 規則 | 門檻 | docs/ | scripts/ | 合計 |
|---|---|---:|---:|---:|
| complexity | max 10 | 0 | 0（無 JS 檔） | **0** |
| max-depth | max 3 | 0 | 0（無 JS 檔） | **0** |
| max-lines-per-function | max 50 | 0 | 0（無 JS 檔） | **0** |
| sonarjs/cognitive-complexity | max 15 | 0 | 0（無 JS 檔） | **0** |

- ESLint 總數：**0 警告／0 錯誤**（exit 0）。瀏覽器全域（window/document/fetch 等）與 Node 全域（require/module/process 等）分別宣告，無 no-undef 誤報。
- `scripts/` 目前沒有任何 JavaScript 檔（僅 Python），因此 Node 範圍為 0 檔掃描；一旦加入 JS 檔，`npm run static-check` 會自動納入（見 `scripts/static_check.sh` 的 find 判斷）。

## 2. 重複率與重複片段位置（jscpd，min-tokens 30，門檻 1%）

**範圍統計：21 clones / 175 重複行 / 3613 總行數 → 重複率 4.84%**（token 計 3.35%）——超出 1% 門檻，jscpd exit 1。

重複片段依類型分組：

### 2-1 手寫 JS（1 處）
| # | 片段 A | 片段 B | 對應 issue |
|---|---|---|---|
| 1 | `docs/app.js:103-108`（runIris 輸入驗證） | `docs/app.js:119-124`（runAI4I 輸入驗證） | #11 |

### 2-2 手寫 HTML 檔頭樣式（2 處）
| # | 片段 A | 片段 B | 對應 issue |
|---|---|---|---|
| 2 | `docs/index.html:1-6` | `docs/inference.html:1-6` | #12 |
| 3 | `docs/index.html:6-14` | `docs/inference.html:6-14` | #12 |

### 2-3 生成物：docs/report/ HTML（17 處）
| # | 片段 A | 片段 B | 備註 |
|---|---|---|---|
| 4-5 | `ai4i2020_prediction_paper.html:1-6`、`8-13` | `report/ai4i_baselines.html:1-6`、`8-13` | pandoc 風格 head / paper.css |
| 6-18 | `ai4i2020_prediction_paper.html` 自身 15 處（170-175↔393-398、259-264↔301-306/341-346、265-271↔309-315/351-357、301-306↔341-346、310-315↔352-357、318-323↔511-516、325-330↔516-521、393-398↔440-445） | 同檔 | markdown 表格/figure 章節樣板重複 |

### 2-4 生成物：docs/results.json 資料（2 處）
| # | 片段 A | 片段 B | 備註 |
|---|---|---|---|
| 19 | `results.json:82-109` | `results.json:129-156` | 三模型 metrics 同構 |
| 20 | `results.json:110-125` | `results.json:157-172` | 同上 |

### 2-5 手寫 Python（scripts/，3 處）
| # | 片段 A | 片段 B | 對應 issue |
|---|---|---|---|
| 21a | `scripts/run_ai4i_analysis.py:61-67`（save_figure） | `scripts/run_analysis.py:48-54`（同函式） | #15 |
| 21b | `scripts/run_ai4i_analysis.py:136-150` | `scripts/run_ai4i_analysis.py:360-374`（RF pipeline 重組裝） | #15 |
| 21c | `scripts/run_ai4i_analysis.py:301-307` | `scripts/run_ai4i_analysis.py:323-329`（曲線繪圖前重訓練片段） | #15 |

> 註：先前全 repo 掃描（含 tests/）另有 `test_ai4i_analysis.py:50-61 ↔ test_ai4i_features.py:95-106`（#16），本次範圍限定不含 tests/。

## 3. 未使用的檔案與函式（knip 5.88.1）

| 檢查項 | 結果 |
|---|---|
| 未使用檔案 | **0**（`eslint.config.mjs` 已於 knip 設定標記為工具設定檔，非應用檔） |
| 未使用 exports / 函式 | **0**（`docs/app.js` 的 `predictIris`、`predictAI4I` 經 `module.exports` 供測試使用） |
| 未使用依賴 | **0**（4 個 devDependencies 皆被 static-check 流程引用） |

## 4. 「前端推論」程式規模

前端推論＝`docs/app.js`（Iris MLP 前向傳播 ＋ AI4I LogReg sigmoid ＋ 表單接線）＋ `docs/inference.html`（表單 UI）。

| 檔案 | 總行數 | 函式數 | 函式清單 |
|---|---:|---:|---|
| `docs/app.js` | 142 | **6** | `predictIris`、`predictAI4I`、`show`、`percentFromProb`、`runIris`、`runAI4I` |
| `docs/inference.html` | 95 | 0 | 純 HTML/CSS＋引用 app.js |
| **前端推論合計** | **237** | **6** | 模型常數（Iris 權重 251 個、AI4I 係數 24 個）佔靜態資料，不含邏輯行 |

對照用：首頁渲染直式資料 `docs/site.js` 為 95 行／7 個函式（renderIris、renderAI4I、renderEda、renderGallery、init、fmt、pct）＋外層 IIFE，屬「首頁展示」非推論邏輯。

## 5. 建議（後續追蹤皆已有事件單）

1. #15（中）：Python 端抽出共用 helper——`save_figure`、RF pipeline factory、曲線資料產生器；需確保同 seed 行為不變（#7 reproducibility 測試為防護網）。
2. #11（低）：app.js 抽 `readInputs(ids)` 替換兩處輸入驗證。→ **已修復（見 §6 記錄 1）**
3. #12（低）：兩頁抽出共用 `static-style.css`；或接受單檔自足。
4. #13/#14（資訊性）：在 jscpd 設定中把生成物（docs/report/**、docs/results.json）加入 ignore，讓純手寫碼的重複率更能反應現實（當前 4.84% 中絕大部分來自生成物）。→ **已記錄（見 §6 記錄 3、4）**

工具輸出原文：`results/static/eslint.log`、`results/static/jscpd.log`、`results/static/knip.log`、HTML 報表 `results/static/jscpd/jscpd-report.html`。

## 6. Lab4 修復記錄（一次一項，每項全綠才下一項）

基線（前置：修 `scripts/static_check.sh` eslint node 範圍加 `--no-error-on-unmatched-pattern`，因 scripts/ 無 JS 時 eslint 10 直接報錯）：eslint 0／jscpd **36 clones、302 行、5.49%**／knip 0。

### 記錄 1 — issue #11：docs/app.js 輸入驗證重複
- 修了什麼：app.js 抽出共用 helper `readInputs(ids)`（parseFloat 逐欄讀值）與 `requireInputs(vals)`（NaN → 顯示「尚有欄位未填數值」並回傳 false），`runIris`／`runAI4I` 兩處改呼叫 helper；id 順序、錯誤訊息、Number.isNaN 行為不變。檔案 142 → 151 行。
- 警告數變化：jscpd **36 → 35 clones**、302 → 296 行、5.49% → **5.37%**（app.js 1 處重複消滅）；eslint 0（無新增違規、無 disable）。
- 測試數：pytest **95 passed**；`npm run frontend-test` golden 6 案例 OK；`npm run docs-test` 3 組行為 OK。全綠，無紅燈。

### 記錄 2 — issue #12：docs/index.html ↔ docs/inference.html 檔頭樣式重複
- 修了什麼：兩頁完全相同的 7 條規則（`:root`、`*`、`body`、`main`、`h1`、`.intro`、`a`）抽到新檔 `docs/static-style.css`（8 行），兩頁以 `<link rel="stylesheet" href="static-style.css">` 引用並維持原 cascade 順序；頁面專屬規則留在各自 `<style>`（index 49 行、inference 29 行）。index.html 132 → 126 行、inference.html 95 → 89 行。
- 警告數變化：jscpd **35 → 34 clones**、296 → 287 行、5.37% → **5.21%**（#12 兩處偏高重複消失）；eslint 0。
- 測試數：pytest **95 passed**；golden 6 案例 OK；docs-test 3 組行為 OK。全綠，無紅燈。

### 記錄 3 — issue #13：生成物 docs/report/ HTML 重複（17 處）
- 修了什麼：`scripts/static_check.sh` 的 jscpd 加 `--ignore "**/docs/report/**"`。這些 HTML 是 `scripts/md_to_html.py` 由 markdown 生成的生成物，不手改內容；排除後 jscpd 只反映手寫碼。**未動** docs/report/ 任何檔案。
- 警告數變化：jscpd **34 → 17 clones**、287 → 181 行、5.21% → **3.88%**；eslint 0；knip 0。jscpd 仍 exit 1（剩餘為 #14/#15/#16 範圍的手寫碼重複，門檻 1% 需後續項目逐步消化）。
- 測試數：pytest **95 passed**；golden 6 案例 OK；docs-test 3 組行為 OK。全綠，無紅燈。

### 記錄 4 — issue #14：生成物 docs/results.json 資料同構重複（2 處）
- 修了什麼：jscpd 追加 `--ignore "**/docs/results.json"`（與 #13 的 docs/report 排除並存）。results.json 是訓練管線產出的資料檔，內容一字未動；排除後剩餘重複全為手寫碼與頁面模板。
- 警告數變化：jscpd **17 → 15 clones**、181 → 137 行、3.88% → **3.31%**；eslint 0；knip 0。
- 測試數：pytest **95 passed**；golden 6 案例 OK；docs-test 3 組行為 OK。全綠，無紅燈。
