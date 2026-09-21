# FAN 個人化改善 · 検証レポート

生成日時: 2026-09-22 03:16 JST　入力hash: `1b5067e6a6b02df47842225a32e5cf70f8e61f4bad4e302582676dd3940f5e49`

`build_report.py` がディスク上の成果物だけから作ります。成果物がない項目は
**未実施**であり、成功として扱いません。再実行すると同じ入力からは同じ判定になります。

## 3つの完了状態

| 状態 | 判定 | 内訳 |
|---|---|---|
| 実装完了 | **不合格** | 自動テスト: 合格 / Preflight（資産・モデル）: 不合格 / サンプルと個人化ハッシュの不変: 合格 |
| 実機検証 | **不合格** | encoding 数値検査: 合格 / 実ブラウザーの一連操作: 合格 / 連続セッション実測: 合格 / catalog v2 の確認: 不合格 |
| 精度実証 | **未実施** | screen（8条件の絞り込み）: 合格 / refine（alpha 追加比較）: 合格 / heldout（事前基準の判定）: 未実施 / 本人によるブラインド評価: 未実施 |

## 現在の設定

- 既定 policy: `legacy_exhibit`（評価が揃うまで変更しない）
- 展示の catalog: `catalog-v2`
- エンコーダー設定の正: `exhibit/configs/fan-policies.json`。`configs/demo.json` はパス・pin・decoder hash のみ。
- FAN pin: `9d0b76843f6437718195accac9cf3f050a25d26b`

## 残っている作業

- 実装完了 / Preflight（資産・モデル）: 不合格（Unreviewed card: girl-c2-l2-t2-m1, Unreviewed card: student-c1-l0-t2-m3）
- 実機検証 / catalog v2 の確認: 不合格（not_reviewed: initial axis phrases do not render all four aspects distinguishably; revise phrases and regenerate before any review entry is written）
- 精度実証 / heldout（事前基準の判定）: 未実施（heldout_not_run）
- 精度実証 / 本人によるブラインド評価: 未実施（study_not_built）

## 再実行

```bash
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_report.py
```

人の回答は収集していません。代わりの回答を作ることはしません。
