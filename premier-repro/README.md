# premier-repro — Premier (CVPR 2026) をローカルで動かす / 再現する

論文: **Premier: Personalized Preference Modulation with Learnable User Embedding in Text-to-Image Generation**
(Wang et al., CVPR 2026 Highlight, [arXiv:2603.20725](https://arxiv.org/abs/2603.20725))

このフォルダには 2 つの独立した部分がある。

| ディレクトリ | 内容 | 状態 |
|---|---|---|
| [`official/`](official/README_ja.md) | **公式実装** (github.com/120L020904/Premier) + 公開重み (hf.co/pino10010/Premier) を RTX 4090 (24 GB) で動かすためのラッパ | **生成・新規ユーザー学習とも動作確認済み** (`official/results/`)。`official/README_ja.md` 参照 |
| `premier/`, `scripts/`, `configs/` | 論文 (TeX ソース) からの **独自再現実装** (FLUX.1-dev + 自作 Preference Adapter / Dispersion Loss / 2 段階学習)。公式実装が見つかる前に書いたもの | コア部分 (トークン別変調・学習ステップ) は GPU で動作確認済み。学習データ (PrefBench) の取得が途中 |

公式コードで十分な用途 (生成・新規ユーザー学習) は `official/` を使う。独自実装はアダプタを一から学習したい場合や、
論文の式を追いたい場合の参考 (`docs/` に手法ノート)。

## 論文の要点 (実装との対応)

- **Learnable user embedding**: ユーザーごとに 30×1024 の学習可能テンソル。
- **Preference Adapter** ×2: T5 テキストトークンを Query、ユーザー埋め込みを Key/Value とする cross-attention 3 層。
  出力はテキストトークンごとの modulation 方向 Δ。block-shared (全 DiT block 共通) と block-distinct (block ごと) の 2 種。
  FLUX (MM-DiT) の AdaLN 入力ベクトル y に `y_i^j = y + Δ_shared_i + Δ_distinct_i^j` として加える (テキストトークンのみ)。
- **Dispersion loss**: 空プロンプトで計算した Δ をユーザー間で引き離す InfoNCE 型損失 (λ=0.1)。
- **2 段階学習**: (1) アダプタ + 学習ユーザー埋め込みを Flow matching + dispersion で学習 (2) 新規ユーザーは
  学習ユーザー埋め込みの線形結合係数のみを Flow matching で学習 (少数枚に強い)。
- 公開重みでは block-distinct アダプタは double block 19 個のみを変調し、その入力テキストは空プロンプト (`uncond: true`)。

## 独自再現実装の状態 (参考)

- `scripts/test_modulation.py` で、トークン別変調が Δ=0 のとき公式 FLUX と一致すること、int8 量子化 transformer + gradient checkpointing で
  512px・batch 2 の学習 step が 1.2 秒 / ピーク 17.7 GB で回ることを確認済み。
- 学習データ: `scripts/00_prepare_prefbench.py` で PrefBench (wenyii/PrefBench, diffusiondb split) から 1000+100 ユーザーの manifest を作成済み
  (`data/prefbench/manifest.json`)。画像本体 (tar 24 GB) は `scripts/00b_download_parts.py` で取得する (未完了、再開可能)。
- 以降の手順: `01_cache_features.py` → `02_train_stage1.py` → `03_train_new_user.py` → `04_generate.py` → `05_evaluate.py` (`configs/default.yaml`, `configs/smoke.yaml`)。
