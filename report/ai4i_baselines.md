# 基準模型比較：AI4I 2020 預測性維護

- 資料筆數：10000，欄位數：16
- 失效比例：3.39%（類別嚴重不平衡，不採用 Accuracy 做主指標）
- 切分：分層 70/30（seed = 42），以 recall / F1 / PR-AUC / ROC-AUC 評估
- 失效模式分布：twf 46, hdf 115, pwf 95, osf 98, rnf 19

## 特徵工程

- `type` one-hot 編碼（drop first）
- 派生特徵：`torque_speed_product`（扭矩 × 轉速）、`temperature_difference`（製程溫度 − 環境溫度）
- 資料外洩防護：移除 `udi`、`product_id`，並且**不使用** 5 個失效模式欄位作為特徵
- 不平衡處理：Logistic Regression / Decision Tree 使用 `class_weight=balanced`，Random Forest 使用 `balanced_subsample`

## Top 5 區辨特徵

```text
- rotational_speed
- tool_wear
- torque
```

## 模型比較（測試集）

|                     |   recall |       f1 |   pr_auc |   roc_auc |
|:--------------------|---------:|---------:|---------:|----------:|
| logistic_regression | 0.823529 | 0.301616 | 0.424235 |  0.917096 |
| decision_tree       | 0.813725 | 0.549669 | 0.758012 |  0.840658 |
| random_forest       | 0.715686 | 0.802198 | 0.849565 |  0.977036 |

## 混淆矩陣摘要

```text
- logistic_regression: TP=84, FN=18, FP=371, TN=2527
- decision_tree: TP=83, FN=19, FP=117, TN=2781
- random_forest: TP=73, FN=29, FP=7, TN=2891
```

## 特徵重要度（Random Forest）

```text
- torque: 0.2193
- rotational_speed: 0.2157
- tool_wear: 0.2052
- torque_speed_product: 0.1892
- temperature_difference: 0.0839
- air_temperature: 0.0472
- process_temperature: 0.0296
- type_M: 0.0066
```

## 結論

- 三組基準模型中，以 F1 最優者為 `random_forest`。
- 由於類別不平衡，PR-AUC 與 Recall 是本任務較具參考價值的指標；Precision-Recall 曲線見 `figures/ai4i_precision_recall_curves.png`。
- 後續可再比較：閾值調校、SMOTE 過採樣、以及 Gradient Boosting 系列模型。
