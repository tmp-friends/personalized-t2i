# Premier: 論文・公式実装との差分

## Upstream boundary

- 公式 repository: <https://github.com/120L020904/Premier>
- 固定 commit: `42473476a189b6b0127890a98e92b7edf49c0d59`
- clean source: `upstream/`
- patch 適用済み source: `.work/upstream/`

`upstream/` は直接変更しない。`scripts/prepare_upstream.py` が次の patch を適用する。

| Patch | Affected file | Reason |
|---|---|---|
| `0001-fix-single-block-checkpoint-kwarg.patch` | `scripts/pipeline/flux_adapter.py` | single-block checkpoint path が受け取らない `use_img_mod` keyword を渡す runtime error を修正 |

## Local integration differences

- `src/premier_repro/official.py` は wrapper が必要とする `EmbeddingLinearCombination` と
  `encode_images` の同一ロジックだけを保持し、CLI help 時の Lightning / dataset import を避ける。
- `scripts/run_official.py` と `scripts/train_official_user.py` は 24 GB GPU 向けに fp8 / int8、
  T5 offload、明示的な `artifacts/` / `outputs/` path を提供する。
- 独自の学習・評価 pipeline は `src/premier_repro/` と `scripts/00_*.py`〜`05_*.py` に分離する。
- 公開されていない test-user 元画像と 4×A800 前提の Stage 1 全学習は検証していない。

## Validation

```bash
python scripts/prepare_upstream.py
python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src scripts
.venv/bin/python scripts/run_official.py --help
git -C upstream status --short
```
