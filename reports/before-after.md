# 修復前後對照報告（Lab4 五項）

- 期間：2026-10-23，一次一項，每項修復後全測試綠燈才繼續
- 基線來源：`npm run static-check`（修復開跑前重跑）；逐項記錄：`reports/smell-report.md` §6
- 對照議題：#11、#12、#13、#14、#16（修復總覽留言於 #10）

## 1. 指標修正前後

| 指標 | 修復前（基線） | 修復後 | 變化 |
|---|---|---|---|
| eslint 警告／錯誤 | 0／0 | 0／0 | 不變（全程未使用 eslint-disable） |
| eslint complexity / max-depth / max-lines-per-function / cognitive | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 不變（重構未讓任何函式越過門檻） |
| jscpd clones | 36 | **15** | −21 |
| jscpd 重複行數 | 302 | **137** | −165 |
| jscpd 重複率 | 5.49% | **3.31%** | −2.18 個百分點 |
| knip 未使用檔案／exports／依賴 | 0／0／0 | 0／0／0 | 不變 |
| pytest 總數 | 95 passed | 95 passed | 不變（斷言零修改、未新增測試也未被逼紅） |
| golden 前端案例（`npm run frontend-test`） | 6 案例 OK | 6 案例 OK | 不變（機率誤差 < 1e-6） |
| docs 頁面行為（`npm run docs-test`） | 3 組 OK | 3 組 OK | 不變 |

逐項下降軌跡（jscpd clones／行數／%）：36/302/5.49% →（#11）35/296/5.37% →（#12）34/287/5.21% →（#13）17/181/3.88% →（#14）15/137/3.31%。

## 2. 各項修改 diff 摘要（改哪個檔案、幾行）

### #11 app.js 輸入驗證重複（commit 27428b8）
- `docs/app.js`：＋18／−9（合計 27 行異動）——新增 `readInputs(ids)` 3 行與 `requireInputs(vals)` 12 行，`runIris`／`runAI4I` 兩處的讀值＋驗證迴圈各縮為 2 行呼叫；id 順序、錯誤訊息、`Number.isNaN` 行為逐字不變。檔案 142 → 151 行。
- `scripts/static_check.sh`：＋1／−1——前置修復：node 範圍 eslint 加 `--no-error-on-unmatched-pattern`（scripts/ 無 JS 時 eslint 10 直接報錯的既有 bug）。
- `reports/smell-report.md`：＋10／−1（§6 記錄 1）＋結果 log 檔更新。

### #12 兩頁檔頭樣式重複（commit dff933a）
- `docs/index.html`：＋1／−7——刪 7 條與 inference.html 完全相同的規則，換成 1 行 `<link rel="stylesheet" href="static-style.css">`。
- `docs/inference.html`：＋1／−7——同上。兩檔各 132 → 126、95 → 89 行。
- `docs/static-style.css`：新檔 ＋8 行（`：root`、`*`、`body`、`main`、`h1`、`.intro`、`a` 共 7 條規則＋1 行檔頭註解）。
- `reports/smell-report.md`：＋6／−1（§6 記錄 2）。

### #13 生成物 docs/report/ 排除（commit 60883f0）
- `scripts/static_check.sh`：＋1／−1——jscpd 加 `--ignore "**/docs/report/**"`。**生成物 HTML 內容零改動**。

### #14 生成物 docs/results.json 排除（commit eba74d4）
- `scripts/static_check.sh`：＋1／−1——jscpd 追加 `--ignore "**/docs/results.json"`。**results.json 內容零改動**（完整性驗證 10,914 bytes）。

### #16 tests/ 重複片段（commit 29c6b58＋9237d8c：結案，零程式碼修改）
- 調查結論：test_ai4i_analysis.py:51-58 ↔ test_ai4i_features.py:97-103 的逐字重複**全部**是 `np.testing.assert_allclose(...)` 斷言本體（torque_speed_product、temperature_difference 公式驗證），不含任何準備程式碼。
- 依「不得修改任何測試的斷言」約束止步未改，僅在 `reports/smell-report.md` 新增 6 行記錄（§6 記錄 5），後 1 行註記裁定結案。

## 3. 最難修的一項：#16

**難的原因是「約束衝突」而非「技術難度」。**

- #11~#14 的難點是行為等價（id 順序、錯誤訊息、cascade 順序逐字保留），有 golden／docs 頁面測試把關，機械性強、風險可控。
- #16 的重複片段本身就是斷言本體：把它抽到 conftest.py 成共用 helper，必然要改寫兩個測試檔的 `np.testing.assert_allclose` 呼叫——這正是本次流程的紅線（「不要修改任何測試的斷言」）。修它唯一「安全」的方式就落在禁止區內。
- 折衷路徑（抽 `assert_derived_features_match(enriched)`，公式與期望值不變、只換呼叫處）在語義上零風險，但它在仍是斷言搬移，是否允許屬**裁定**而非工程判斷；遂記錄止步、提交裁定（維持現狀結案）。jscpd 因此維持 15 clones（若放寬可再降到 14）。
