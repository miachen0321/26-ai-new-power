# AI4I 2020 資料探索摘要

- 資料筆數：10000
- 欄位數：16
- 缺失值：0
- 重複列：0
- 類別分布：正常 9661、失效 339
- 失效比例：3.39%
## 各失效模式出現次數

```text
twf: 46
hdf: 115
pwf: 95
osf: 98
rnf: 19
```
## 特徵與失效的相關係數

|                     |   machine_failure |
|:--------------------|------------------:|
| air_temperature     |            0.0826 |
| process_temperature |            0.0359 |
| rotational_speed    |           -0.0442 |
| torque              |            0.1913 |
| tool_wear           |            0.1054 |
| machine_failure     |            1      |

## 各類別特徵平均

|   machine_failure |   air_temperature |   process_temperature |   rotational_speed |   torque |   tool_wear |
|------------------:|------------------:|----------------------:|-------------------:|---------:|------------:|
|                 0 |           299.974 |               309.996 |            1540.26 |   39.63  |     106.694 |
|                 1 |           300.886 |               310.29  |            1496.49 |   50.168 |     143.782 |

## 區辨力最高的特徵（依類別平均差）

```text
- rotational_speed: -43.7733
- tool_wear: 37.0880
- torque: 10.5385
- air_temperature: 0.9124
- process_temperature: 0.2947
```