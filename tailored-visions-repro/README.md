# Tailored Visions — 公式実装と評価パイプライン

## 1. 手法概要

**Tailored Visions: Enhancing Text-to-Image Generation with Personalized Prompt Rewriting**
（Chen et al., CVPR 2024, [arXiv:2310.08129](https://arxiv.org/abs/2310.08129)）の
公式実装と独自評価パイプラインを扱う。

ユーザーの過去 prompt から現在の短い query に関連する履歴を検索し、履歴を嗜好の証拠として
LLM に渡す。LLM が prompt を書き換え、その結果を画像生成モデルへ入力する。生成モデル自体は変更しない。

## 2. 現在の再現状況

| 実装 | パス | 状態 |
|---|---|---|
| 公式実装 | `upstream/` + `patches/` | 固定 SHA に compatibility patches を適用して動作確認済み |
| 公式 runner | `scripts/run_official.sh` | local LLM、demo、bounded main、停止を一括管理 |
| 独自評価 | `src/tailored_visions_repro/`, `scripts/00_*.py`〜`08_*.py` | Table 2、ablations、PMS / Image-Align proxy / ROUGE-L、leakage analysis |
| 詳細ノート | `docs/OFFICIAL_SETUP.md`, `docs/DEVIATIONS.md`, `docs/summary.html` | setup、差分、論文サマリー |

## 3. Quick start

```bash
git submodule update --init --recursive
cd tailored-visions-repro
uv sync --locked
uv run python scripts/prepare_upstream.py
bash scripts/run_official.sh --help
```

公式 demo を local LLM で実行する場合:

```bash
scripts/run_official.sh demo 'a cat' --no-t2i
scripts/run_official.sh stop
```

## 4. ディレクトリ構成

```text
tailored-visions-repro/
├── upstream/                        # 公式 Git submodule（直接編集しない）
├── patches/                         # 公式実装への 3 compatibility patches
├── .work/upstream/                  # patch 適用済み実行 tree（Git 対象外）
├── src/tailored_visions_repro/      # 独自再現・評価 package
├── scripts/                         # 公式 runner と 00〜08 pipeline
├── docs/                            # setup、deviations、summary
├── data/raw/user_data/              # PIP dataset（Git 対象外）
├── outputs/official/                # 公式 runner の出力（Git 対象外）
└── outputs/                         # 独自 pipeline の出力（Git 対象外）
```

## 5. 公式実装と固定コミット

- URL: <https://github.com/zzjchen/Tailored-Visions>
- 固定コミット: `d0f4454ca08c68c5d30f08a01ff4a23a8b33b610`
- submodule: `upstream/`
- patch 適用済み source: `.work/upstream/`
- patch order: `patches/series`

`scripts/prepare_upstream.py` は SHA を検証し、3 patches を順番に一時 tree へ適用してから
`.work/upstream/` を原子的に更新する。詳細は [docs/DEVIATIONS.md](docs/DEVIATIONS.md) を参照。

## 6. 生成・学習・評価手順

公式実装:

```bash
scripts/run_official.sh demo 'a cat'              # rewrite + images
scripts/run_official.sh demo 'a cat' --no-t2i     # rewrite only
scripts/run_official.sh main --limit_users=5      # bounded PIP run
scripts/run_official.sh main --limit_users=5 --t2i
scripts/run_official.sh stop
```

独自評価 pipeline:

```bash
./run_all.sh                    # 全 stage
./run_all.sh --skip-ablations   # Table 2 + image metrics
```

```text
00 download → 01 prepare → 02 rewrite → 07 leakage → 03 subset
→ 04 preferences → 05 generate → 06 evaluate → 08 examples
```

各 stage は既存出力をスキップするため再開可能。公式 runner の詳細と OpenAI API の使い方は
[docs/OFFICIAL_SETUP.md](docs/OFFICIAL_SETUP.md) に記載する。

## 7. 動作確認結果

- 公式 `demo.py` / `main.py` は EBR・BM25、naive・ICL、T2I 有無の組合せで確認済み。
- 移行前の実行では Table 2 の `shortened_prompt`、`promptist`、`general_pr` が全 6,232 sample で完了。
- released dataset は 3,116 users / 300,255 prompts。論文記載より 1 user / 18 samples 多い。
- test ground truth prompt は履歴に完全一致で 35.6%、token-F1 ≥ 0.8 で 49.1% 存在し、ROUGE-L を押し上げる。
- query 自体が ground truth と同一の sample は 18.2%。

結果解釈と未完了 stage は [docs/DEVIATIONS.md](docs/DEVIATIONS.md) を参照。

## 8. 論文・公式実装との差分

- paper の `gpt-3.5-turbo` の代わりに、既定では local `Qwen/Qwen3.5-4B` を使う。絶対値は論文と一致しない。
- dataset の画像 URL が失効しているため、Image-Align は ground-truth prompt から生成した proxy image との比較になる。
- PMS 用 user preference summaries は未公開のため再構成する。
- 生成モデル既定は SDXL。paper-faithful な SD v1.5 は `--model stable-diffusion-v1-5/stable-diffusion-v1-5` で指定する。
- official compatibility fixes は source を直接変更せず、3 patches として管理する。

## 9. データ、重み、生成物

| 種類 | パス | Git |
|---|---|---|
| PIP dataset | `data/raw/user_data/` | 追跡しない |
| patch 適用済み公式 tree | `.work/upstream/` | 追跡しない |
| local LLM / diffusion model cache | Hugging Face cache | 追跡しない |
| 公式実装の出力 | `outputs/official/` | 追跡しない |
| 評価 pipeline の出力 | `outputs/` | 追跡しない |

移行前 checkout に `data/user_data/` や `results/` が残っている場合は、branch 統合前にそれぞれ
`data/raw/user_data/`、`outputs/` へ一度だけ移す。Git は再取得可能なデータと生成物を管理しない。

## 10. 引用

```bibtex
@inproceedings{chen2024tailored,
  title     = {Tailored Visions: Enhancing Text-to-Image Generation with Personalized Prompt Rewriting},
  author    = {Chen, Zijie and Zhang, Lichao and Weng, Fangsheng and Pan, Lili and Lan, Zhenzhong},
  booktitle = {CVPR},
  year      = {2024}
}
```
