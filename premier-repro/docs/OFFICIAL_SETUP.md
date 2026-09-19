# Premier 公式実装を RTX 4090 (24 GB) で動かす

- 論文: *Premier: Personalized Preference Modulation with Learnable User Embedding in Text-to-Image Generation* (CVPR 2026 Highlight, [arXiv:2603.20725](https://arxiv.org/abs/2603.20725))
- 公式コード: <https://github.com/120L020904/Premier> (`upstream/` submodule、固定 SHA は README を参照)
- 公開重み: <https://huggingface.co/pino10010/Premier> (`artifacts/weights/pino10010_Premier/` に配置)

## 構成と責務

```text
premier-repro/
  upstream/                   公式リポジトリの read-only submodule
  patches/                    公式実装へ適用する追跡可能な差分
  .work/upstream/             submodule と patch から生成する作業コピー（Git 管理外）
  src/premier_repro/          再現コードとローカル統合層
  scripts/prepare_upstream.py patch 適用済み作業コピーの生成
  scripts/run_official.py     画像生成 CLI
  scripts/train_official_user.py  新規ユーザー埋め込み学習 CLI
  data/examples/              小さな追跡可能サンプル
  artifacts/weights/          公開重み（Git 管理外）
  outputs/official/           生成・学習結果（Git 管理外）
```

`upstream/` は直接変更しない。`python scripts/prepare_upstream.py` が固定 SHA を検証し、
`patches/series` の順に差分を適用して `.work/upstream/` を原子的に生成する。

公式スクリプトには著者環境の絶対パスが含まれるため、アルゴリズムを呼び出す薄い CLI を
`scripts/` に置いた。24 GB GPU 向けの読み込み処理と、公式学習コードから必要な
`encode_images` / `EmbeddingLinearCombination` は `src/premier_repro/official.py` に隔離している。

## 24 GB GPU で動かすための変更点

| 問題 | 対策 |
|------|------|
| 公式コードは FLUX transformer と T5 を bf16 で GPU に載せる | `--memory fp8` は layerwise casting、`--memory int8` は optimum-quanto を使用 |
| `FLUX.1-dev` の一部コンポーネントがローカルキャッシュにない | VAE / CLIP-L / T5-XXL は同一重みの `FLUX.1-schnell` キャッシュから読み、scheduler は dev 設定を使用 |
| 公式の新規ユーザー学習は Lightning + bf16 全載せ | ローカル CLI は同じ学習ステップを PyTorch ループで実行し、int8 transformer + gradient checkpointing を使用 |
| 公式依存関係に実行時依存が不足 | `requirements-official.txt` と `pyproject.toml` で補完 |
| gradient checkpointing 分岐が single block に存在しない `use_img_mod` を渡す | `patches/0001-fix-single-block-checkpoint-kwarg.patch` で 1 行削除 |
| diffusers の既定除外では FLUX の AdaLN 射影が bf16 に残る | ローカル統合層で FLUX / T5 ごとの casting 対象を指定 |

## セットアップ

```bash
cd ~/stable-diffusion/personalized-t2i/premier-repro
git submodule update --init upstream
uv sync --locked
uv run python scripts/prepare_upstream.py

# --memory int8 / 学習で使用
uv pip install --python .venv/bin/python optimum-quanto

# 公開重み（必要な場合）
# uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('pino10010/Premier', local_dir='artifacts/weights/pino10010_Premier')"
```

## 画像生成

```bash
uv run python scripts/run_official.py \
  --users none train:0 train:1 test:3685 linear:3685 \
  --prompts "a cat sitting on a windowsill" "a city street at night" \
  --out outputs/official/demo --steps 28 --guidance 3.5 --size 512
```

`--users` の指定:

| 指定 | 内容 |
|------|------|
| `none` | 素の FLUX.1-dev |
| `train:<0-999>` | 学習ユーザーの embedding |
| `test:<id>` | 公開テストユーザーの直接学習 embedding |
| `linear:<id>` | 公開テストユーザーの線形結合版 |
| `file:<path>` | `train_official_user.py` が出力した safetensors |

## 新規ユーザーの学習

```bash
# items.json: [{"image": "img1.png", "caption": "a cat ..."}, ...]
uv run python scripts/train_official_user.py \
  --name alice --json items.json --mode linear --steps 1000 \
  --out outputs/official/users

uv run python scripts/run_official.py \
  --users none file:outputs/official/users/alice/user_combination_alice.safetensors \
  --prompts "a portrait of a woman" --out outputs/official/alice
```

- `--mode linear`: 学習済み 1000 ユーザーの埋め込みの線形結合係数を学習する（論文のデフォルト）
- `--mode direct`: 30×1024 の埋め込みを一から学習する。画像が 8 枚以上ある場合向け
- 公式設定は 5000 step。短い確認では `--steps` を下げられる

## 動作確認結果

2026-09-05、RTX 4090 24 GB、他プロセスなしで確認した結果:

| 項目 | 結果 |
|------|------|
| 生成 (`--memory fp8`, 512 px, 20 step) | 5.4 秒/枚、GPU ピーク 19.4 GB |
| 新規ユーザー学習 (`--mode linear`, `data/examples/watercolor/` 8 枚, 300 step) | 0.8 秒/step（約 4 分）、GPU ピーク 14.2 GB |
| 学習した embedding での生成 | 未学習プロンプトでも紙質感・平坦な塗り・イラスト調への変化を確認 |

GPU 空きが 20 GB 未満の場合は `--t5 offload`（自動判定あり）で T5 を CPU RAM から
ストリーミングする。学習では T5 を先に使い切って解放する。

## 未検証・注意

- Stage 1（アダプタ + 1000 ユーザー埋め込み学習）は 4×A800 と著者データ形式が前提。公開重みを使う通常の再現では不要。
- 公式の ViPer / CLIP / LPIPS 評価コードは含まれない。ローカル評価は `src/premier_repro/eval/metrics.py` を参照。
- テストユーザー 50 人の元画像は公開されていない。
- `artifacts/`、`outputs/`、`.work/` は再生成・再取得可能なため Git 管理外。
