# FAN 個人化改善 · 検証レポート

生成日時: 2026-09-22 12:11 JST　入力hash: `dc1b5739f6c8eb534e65c07404311f5912a8c23329acb92b3caee19b546e3cd5`

`build_report.py` がディスク上の成果物だけから作ります。成果物がない項目は
**未実施**であり、成功として扱いません。再実行すると同じ入力からは同じ判定になります。

## 3つの完了状態

| 状態 | 判定 | 内訳 |
|---|---|---|
| 実装完了 | **合格** | 自動テスト: 合格 / Preflight（資産・モデル）: 合格 / サンプルと個人化ハッシュの不変: 合格 |
| 実機検証 | **合格** | encoding 数値検査: 合格 / 実ブラウザーの一連操作: 合格 / 連続セッション実測: 合格 / catalog v2 の確認: 合格 |
| 精度実証 | **未実施** | screen（8条件の絞り込み）: 合格 / refine（alpha 追加比較）: 合格 / heldout（事前基準の判定）: 合格 / 本人によるブラインド評価: 未実施 |

## 現在の設定

- 既定 policy: `legacy_exhibit`（評価が揃うまで変更しない）
- 展示の catalog: `catalog-v2`
- エンコーダー設定の正: `exhibit/configs/fan-policies.json`。`configs/demo.json` はパス・pin・decoder hash のみ。
- FAN pin: `9d0b76843f6437718195accac9cf3f050a25d26b`

## 残っている作業

- 精度実証 / 本人によるブラインド評価: 未実施（study_not_built）

## 再実行

```bash
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_report.py
```

人の回答は収集していません。代わりの回答を作ることはしません。
