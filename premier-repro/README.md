# Premier — 公式実装のローカル実行と独自再現

## 1. 手法概要

**Premier: Personalized Preference Modulation with Learnable User Embedding in
Text-to-Image Generation**（Wang et al., CVPR 2026 Highlight,
[arXiv:2603.20725](https://arxiv.org/abs/2603.20725)）の公式実装と独自再現を扱う。

Premierは、ユーザーごとの30×1024の埋め込みをPreference AdapterでT5テキスト条件へ融合する。
block-shared／block-distinctの変調をFLUXのAdaLN入力へ加え、Stage 1でadapterと学習ユーザーを、
Stage 2で新規ユーザーの埋め込みまたは既存埋め込みの線形結合係数を学習する。

## 2. 現在の再現状況

| 実装 | パス | 状態 |
|---|---|---|
| 公式実装 | `upstream/` + `patches/` + `scripts/run_official.py` | 生成・新規ユーザー学習をRTX 4090で確認済み |
| 独自再現 | `src/premier_repro/`, `scripts/00_*.py`〜`05_*.py` | コア変調・学習stepをGPU確認済み。PrefBench取得は途中 |
| 詳細ノート | `docs/OFFICIAL_SETUP.md`, `docs/summary.html` | 公式との差分、メモリ対策、検証値を記録 |

通常の生成と新規ユーザー学習には公式実装のwrapperを使う。独自再現はStage 1を一から学習する場合や、論文の式と実装を追う場合に使う。

## 3. Quick start

```bash
git submodule update --init --recursive
cd premier-repro
uv sync --locked
uv run python scripts/prepare_upstream.py
```

公開重みはGitで管理しない。初回だけ次を実行する。

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('pino10010/Premier', local_dir='artifacts/weights/pino10010_Premier')"
```

軽量確認:

```bash
uv run python scripts/run_official.py --help
python -m unittest discover -s tests -v
```

## 4. ディレクトリ構成

```text
premier-repro/
├── upstream/                 # 公式Git submodule（直接編集しない）
├── patches/                  # 公式コードへ適用する互換性修正
├── .work/upstream/           # patch適用済み作業コピー（Git対象外）
├── src/premier_repro/        # 独自再現と公式integration helper
├── scripts/                  # 公式wrapperと独自再現pipeline
├── configs/                  # 独自再現の設定
├── docs/                     # setup、サマリー
├── data/examples/            # 小さな追跡対象fixture
├── data/raw/                 # 取得データ（Git対象外）
├── artifacts/                # 公開重み（Git対象外）
└── outputs/                  # 生成物・checkpoint・ログ（Git対象外）
```

## 5. 公式実装と固定コミット

- URL: <https://github.com/120L020904/Premier>
- 固定コミット: `42473476a189b6b0127890a98e92b7edf49c0d59`
- submodule: `upstream/`
- patch済み実行ソース: `.work/upstream/`

`scripts/prepare_upstream.py` はsubmoduleのSHAを検証し、`patches/series` を一時コピーへ順番に適用する。submodule自体は常にcleanに保つ。

現在のpatchは、公式のgradient-checkpointing分岐がsingle blockに存在しない `use_img_mod` 引数を渡す1行の不具合修正である。

## 6. 生成・学習・評価手順

公式重みで生成:

```bash
uv run python scripts/run_official.py \
  --users none train:0 train:1 test:3685 linear:3685 \
  --prompts "a cat sitting on a windowsill" "a city street at night" \
  --out outputs/official/demo --steps 28 --guidance 3.5 --size 512
```

新規ユーザーの学習:

```bash
uv run python scripts/train_official_user.py \
  --name alice --json data/examples/watercolor/items.json \
  --mode linear --steps 1000 --out outputs/official/users
```

独自再現pipeline:

```text
00_prepare_prefbench.py → 00b_download_parts.py → 01_cache_features.py
→ 02_train_stage1.py → 03_train_new_user.py → 04_generate.py → 05_evaluate.py
```

既定設定は `configs/default.yaml`、軽量確認は `configs/smoke.yaml` を使う。

## 7. 動作確認結果

| 項目 | RTX 4090での確認結果 |
|---|---|
| 公式生成（fp8、512px、20 step） | 5.4秒/枚、GPU peak 19.4GB |
| 新規ユーザー学習（linear、水彩8枚、300 step） | 0.8秒/step、約4分、GPU peak 14.2GB |
| 学習結果 | 未学習promptでも紙質感・平坦な塗り・イラスト調への変化を確認 |
| 独自変調 | Δ=0で公式FLUXと一致。int8 + checkpointingで512px batch 2を確認 |

公開されていないテストユーザー元画像や、4×A800前提の公式Stage 1全学習は未検証。

## 8. 論文・公式実装との差分

- 24GB GPU向けにFLUX/T5のfp8またはint8保存、T5 offloadを追加した。
- 著者環境の絶対パスを使わず、ローカルCLIから公式関数を呼ぶ。
- 公式Stage 2のうちwrapperが使う `EmbeddingLinearCombination` と `encode_images` は、不要なLightning／dataset依存を読み込まないよう `src/premier_repro/official.py` に同じロジックを保持する。
- 公式checkpointingの1行不具合は `patches/0001-fix-single-block-checkpoint-kwarg.patch` で管理する。
- 独自再現の評価指標は公式配布物に含まれないため、`src/premier_repro/eval/` に実装している。

詳細は [docs/OFFICIAL_SETUP.md](docs/OFFICIAL_SETUP.md) を参照。

## 9. データ、重み、生成物

| 種類 | パス | Git |
|---|---|---|
| 水彩fixture | `data/examples/watercolor/` | 追跡する |
| PrefBenchなどの取得データ | `data/raw/` | 追跡しない |
| 公開Premier重み | `artifacts/weights/pino10010_Premier/` | 追跡しない |
| 生成画像、学習結果、ログ | `outputs/` | 追跡しない |

旧checkoutに取得済みの `official/weights/` がある場合は、移行後に `artifacts/weights/` へ一度だけ移すか、上記コマンドで再取得する。

## 10. 引用

```bibtex
@inproceedings{wang2026premier,
  title     = {Premier: Personalized Preference Modulation with Learnable User Embedding in Text-to-Image Generation},
  author    = {Wang et al.},
  booktitle = {CVPR},
  year      = {2026}
}
```
