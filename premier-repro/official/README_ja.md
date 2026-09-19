# Premier 公式実装を RTX 4090 (24 GB) で動かす

- 論文: *Premier: Personalized Preference Modulation with Learnable User Embedding in Text-to-Image Generation* (CVPR 2026 Highlight, [arXiv:2603.20725](https://arxiv.org/abs/2603.20725))
- 公式コード: <https://github.com/120L020904/Premier> (`Premier/` にクローン済み)
- 公開重み: <https://huggingface.co/pino10010/Premier> (`weights/pino10010_Premier/` にダウンロード済み)
  - `mod_adapter.safetensors` (1.8 GB): Preference Adapter (block-shared + block-distinct, 260k step 学習済み)
  - `user_embedding.safetensors` (61 MB): 学習ユーザー 1000 人分の user embedding (各 30×1024)
  - `users/user_embedding_<id>.safetensors`: テストユーザー 50 人分 (直接学習)
  - `users_linear/user_combination_<id>.safetensors`: 同 50 人分 (学習ユーザー埋め込みの線形結合係数 1000 個)
  - `adapter_config.yaml`: アダプタ構成 (T5 トークン=Q、ユーザー埋め込み=K/V、3 層、幅 3072、per-block は double block 19 個分)

## 構成

```
official/
  Premier/            公式リポジトリ (無改変)
  weights/            公開重み
  .venv/              公式 requirements.txt 通りの環境 (torch 2.6.0 / diffusers 0.33.0 / transformers 4.52.4, Python 3.12)
  requirements-local.txt   requirements.txt から社内 index 行を除き einops を追加したもの
  premier_local.py    ローカルの FLUX 重み読み込み・24 GB 向けメモリ対策・公開 user embedding の読み込み
  run_premier.py      画像生成 CLI (公式 generate_xverse をそのまま呼ぶ)
  train_new_user.py   新規ユーザーの embedding 学習 CLI (公式 training_step の移植 + 24 GB 対策)
```

公式スクリプト (`scripts/utils/generate_images_modulation.py`, `monitor_pid_*.sh`) は著者環境のパスがハードコードされていて
そのままでは動かないため、同じ関数 (`generate_xverse`, `load_modulation_adapter`, `transformer_forward_verse`,
`EmbeddingLinearCombination`) を CLI から呼ぶ薄いラッパを用意した。アルゴリズム部分は公式コードを一切変更していない。

## 24 GB GPU で動かすための変更点

| 問題 | 対策 |
|------|------|
| 公式コードは `FluxPipeline.from_pretrained(...).to("cuda")` を bf16 で読む (transformer 24 GB + T5 9.5 GB) | `--memory fp8`: diffusers の layerwise casting で transformer と T5 を float8 保存・bf16 計算 (推論用)。`--memory int8`: optimum-quanto の int8 重み (学習にも使える。nvcc 不要) |
| `FLUX.1-dev` はローカルに transformer しかキャッシュされていない | VAE / CLIP-L / T5-XXL は同一重みの `FLUX.1-schnell` キャッシュから読み、scheduler は dev の設定値をコードで指定 |
| 学習は Lightning + bf16 全載せ (A800 前提) | `train_new_user.py` は同じ学習ステップを素の PyTorch ループで実装し、int8 transformer + gradient checkpointing で動かす |
| `requirements.txt` に `einops`, `sentencepiece`, `protobuf` が無い | `requirements-local.txt` に追加 |
| 公式 `transformer_forward_verse` の gradient checkpointing 分岐が single block に存在しない引数 `use_img_mod` を渡していて落ちる (学習時のみ) | `patches/0001-fix-single-block-checkpoint-kwarg.patch` (1 行削除) を `Premier/` に適用済み |
| diffusers の layerwise casting は名前に `norm` を含む層を bf16 に残すため FLUX の AdaLN 射影 (3.2B params) が残り +3 GB になる。T5 は `T5LayerNorm` と `wo` が活性を重みの dtype に合わせるため fp8 化できない | `premier_local.fp8_cast`: FLUX は `pos_embed` 以外を fp8、T5 は norm / embedding / `wo` を bf16 のまま |

## セットアップ

```bash
cd ~/stable-diffusion/personalized-t2i/premier-repro/official
uv venv .venv --python /usr/bin/python3.12
uv pip install --python .venv/bin/python -r requirements-local.txt
uv pip install --python .venv/bin/python optimum-quanto   # --memory int8 / 学習に必要
# 重み (済み): python -c "from huggingface_hub import snapshot_download; snapshot_download('pino10010/Premier', local_dir='weights/pino10010_Premier')"
```

## 画像生成

```bash
cd ~/stable-diffusion/personalized-t2i/premier-repro/official
source .venv/bin/activate
# 行 = ユーザー、列 = プロンプト。同じ列は同じ seed なのでベースモデルと直接比較できる
python run_premier.py \
  --users none train:0 train:1 test:3685 linear:3685 \
  --prompts "a cat sitting on a windowsill" "a city street at night" \
  --out outputs/demo --steps 28 --guidance 3.5 --size 512
```

`--users` の指定:

| 指定 | 内容 |
|------|------|
| `none` | 素の FLUX.1-dev |
| `train:<0-999>` | 学習ユーザーの embedding |
| `test:<id>` | 公開テストユーザー (直接学習)。id は `weights/pino10010_Premier/users/` を参照 |
| `linear:<id>` | 同テストユーザーの線形結合版 |
| `file:<path>` | `train_new_user.py` の出力 (`user_combination_*.safetensors` / `user_embedding_*.safetensors`) |

## 新規ユーザーの学習 (数枚の「好きな画像」から)

```bash
# items.json: [{"image": "img1.png", "caption": "a cat ..."}, ...]  (公式の CSV 形式 positive_image,caption も可)
python train_new_user.py --name alice --json items.json --mode linear --steps 1000 --out outputs/users
python run_premier.py --users none file:outputs/users/alice/user_combination_alice.safetensors \
  --prompts "a portrait of a woman" --out outputs/alice
```

- `--mode linear` (論文のデフォルト): 学習ユーザー 1000 人の埋め込みの線形結合係数だけを学習 (AdamW lr 0.01, 公式設定)
- `--mode direct`: 30×1024 の埋め込みを一から学習 (Prodigy lr 1.0, 公式設定)。画像が 8 枚以上あるとき向け
- 公式設定は 5000 step (A800 で約 30 分)。4090 での速度は下記「動作確認結果」参照。`--steps` で調整する

## 動作確認結果 (2026-09-05, RTX 4090 24 GB, 他プロセス無し)

| 項目 | 結果 |
|------|------|
| 生成 (`run_premier.py --memory fp8`, 512px, 20 step) | 5.4 秒/枚、GPU ピーク 19.4 GB (transformer fp8 12 GB + T5 fp8 5.8 GB + アダプタ bf16 1.8 GB)。`outputs/check1/grid.jpg`: ベース / 学習ユーザー #0, #1 / テストユーザー 3685 (直接・線形結合) × 2 プロンプト。ユーザーごとに色調・画風が明確に変わり、直接学習と線形結合はほぼ同じ絵になる |
| 新規ユーザー学習 (`train_new_user.py --mode linear`, 水彩 8 枚, 300 step) | 0.8 秒/step (4 分)、GPU ピーク 14.2 GB。`test_data/watercolor/` は FLUX で生成した水彩画 (キャプションにはスタイル語なし) |
| 学習した embedding での生成 (`outputs/watercolor_gen/grid.jpg`) | 未学習の 4 プロンプトで、ベース (写真調) に対し紙質感・平坦な塗り・イラスト調に寄る。300 step の短い学習でも傾向は出る (公式は 5000 step) |

GPU に別プロセスが載っていて空きが 20 GB 未満のときは `--t5 offload` (自動判定あり) で T5 を CPU RAM からストリーミングする
(1 プロンプトあたり数秒の追加コスト)。学習は T5 を先に使い切って解放するので影響なし。

### 未検証・注意
- Stage 1 (アダプタ + 1000 ユーザー埋め込みの学習) は公式スクリプト `scripts/train_flux/train_premier.py` が 4×A800 + 著者のデータ形式前提。
  公開重みがあるので通常は不要。24 GB で回すなら `train_new_user.py` と同じ int8 + checkpointing 化が必要。
- 公式の ViPer / CLIP / LPIPS 評価コードは含まれていない (`../premier/eval/metrics.py` に独自実装あり)。
- テストユーザー 50 人 (id 3685〜4279) は著者の内部データ (pickapic_dxm_coco) の id で、その元画像は公開されていない。
