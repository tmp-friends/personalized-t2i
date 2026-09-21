# catalog-v2 文言パイロット（2026-09-21）

初期の軸文言で生成した64枚は、目視と凍結CLIPの補助指標の両方で4側面を描き分けられていなかった（光 20/64、dreamlike 0/16、4軸一致 5/64）。
文言を2回改訂し、被写体2種×16表現=32枚ずつ、同一seed・同一生成条件で試した。

| 候補 | color | lighting | texture | mood | 4軸一致 |
|---|---|---|---|---|---|
| 初期文言 | 20/32 | 9/32 | 15/32 | 17/32 | 2/32 |
| 改訂案1 | 24/32 | 10/32 | 18/32 | 22/32 | 2/32 |
| 改訂案2 | 25/32 | 8/32 | 19/32 | 27/32 | 4/32 |

数値は凍結した `openai/clip-vit-large-patch14` の top-1（各軸4水準のうち、主張する水準が最上位か）。
補助指標であり、確認の代わりではない。光の判定は目視とずれる。

## 目視の所見

- 改訂案2は雰囲気・描画・vivid が明確になった（`sheet-girl-c0.jpg`、`sheet-student-c3.jpg`）。
- 改訂案1の `colorful` は髪を虹色にして被写体を壊すため不採用（`rejected-candidate-1-student-c3.jpg`）。
  `sunset glow behind` は全パレットへ暖色を持ち込むため不採用。
- 残る問題は **光の軸**。`dramatic lighting` は寒色カードを夜景にし、`backlighting` は見えないことが多い。
  寒色パレットは肌を青く染める（`sheet-girl-c1.jpg`）。

## 判断

- 64枚すべてで4側面を確認済みにする条件は満たせないため、catalog-v2 は **未確認のまま**。
  `exhibit/configs/cards-v2-review.json` は空、展示は catalog-v1 を継続する。
- `exhibit/configs/catalog-v2.json` と heldout fixture は変更していない。heldout は未実施。

## 提案（未決定）

色・描画・雰囲気は `candidate-2-definition.json` の文言を採用する。
光は、パイロットで確実に描き分けられた条件（曇天／夜／夕景／日中）へ水準を再定義する。
これは軸の意味を変えるため、判断は保留している。
採用後の手順: 64枚を再生成 → 側面ごとに確認 → heldout fixture の参照文を更新 → `evaluate_fan.py heldout`。
