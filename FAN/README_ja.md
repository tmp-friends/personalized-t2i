# FAN 公式実装 — 動作確認済みセットアップ (このマシン用)

論文: Kim, Ahn, Seo, **"Foundation Encoders Are All You Need for Preference-Aware Personalization"**, CVPR 2026
公式リポジトリ: https://github.com/Burf/FAN (MIT) — このフォルダはその clone に、環境定義 (`pyproject.toml`) と
起動オプションの小パッチ (`inference.py` に `--variant` / `--ref_weight_dir` を追加) を足したものです。
手法の説明・API は元の `readme.md` を参照してください。

## 何をするコード?

`prompt`(ターゲット) と `ref`(好みを表す参照プロンプト群) をテキストエンコーダに同時に入力し、
self-attention を「Personalized Attention」に差し替えて、参照の嗜好を α の強さで混ぜた条件ベクトルを作ります。
拡散モデル本体・エンコーダとも **追加学習なし**。唯一の学習済み重みは `weight/L.pth`, `weight/bigG.pth`
(OpenCLIP 用の class token 位置検出 MLP, 数百 KB) で、リポジトリに同梱済みです。

## 環境

```bash
cd ~/stable-diffusion/personalized-t2i/FAN
uv sync            # .venv を作成 (torch 2.14 / diffusers 0.39 / transformers 4.57.6)
```

`transformers<5` に固定しています。公式コードは self-attention の forward を monkey patch しており、
transformers 5.x では CLIP/T5 の attention の引数名 (`causal_attention_mask`, `position_bias` 等) が
変わっているため動きません。

## 動作確認済みコマンド

### 1. Stable Diffusion v1.5 (VRAM ~3GB, 約 10 秒/枚)

```bash
.venv/bin/python inference.py \
  --model stable-diffusion-v1-5/stable-diffusion-v1-5 --variant fp16 --dtype float16 \
  --prompt "A photograph of an astronaut riding a horse" \
  --ref "A retro-futuristic space exploration movie poster with bold, vibrant colors" \
  --alpha 0.4 --seed 42 --save_path ./image/sd15_alpha0.4
```

- `--alpha 0` で通常生成 (パーソナライズなし) と同じになります。比較用に α を振ってください。
- `--ref` は複数指定可 (`--ref "A" "B" "C"`)。`--weight` で参照ごとの嗜好強度 (省略時は全て 1)。
- 参照が多い場合 (ユーザー履歴など) は `--sample_size 1` を付けると論文の Tailored profiling
  (履歴の 10% を類似度+多様性で選別) が有効になります。

### 2. CLIP テキストエンコーダ単体 (拡散モデル不要)

```bash
.venv/bin/python smoke_clip.py
```

α=0 が元のエンコード結果と一致すること、α=0.4 で参照側に寄ることを数値で確認します。

## その他のモデル (未検証 / 要リソース)

| モデル | 必要な準備 | 備考 |
|---|---|---|
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` (fp16 ~7GB DL) | VRAM ~8GB。`--variant fp16 --dtype float16` |
| SD3.5 | HF で利用規約に同意 + `hf auth login` | VRAM 大。T5 も FAN 対象 |
| FLUX.1-dev | キャッシュ済み (`~/.cache/huggingface`) | bf16 で ~33GB。24GB の 4090 では `pipeline.to("cuda")` が載らないため、`enable_model_cpu_offload()` や量子化が必要 (公式スクリプトは未対応) |
| unCLIP / CLIP 検索 / LLaVA | `inference.ipynb` 参照 | 同じ `FAN` クラスを画像エンコーダに適用 |

## 展示 (好きな画像を数枚選ぶ → その人向け生成) への使い方メモ

- FAN の入力は **テキスト** の参照 (履歴プロンプト) が基本。画像から始める場合は
  (a) 画像をキャプション/タグ化して `--ref` に渡す、または
  (b) unCLIP 経路 (`FAN(pipeline.image_encoder, ...)`, `get_image_feature(target, refs)`) で画像埋め込みを直接混ぜる。
- 嫌いな画像の扱い: `weight` は正の強度として設計されているため、負の重みはそのままでは想定外。
- α の目安は論文で 0.3–0.4 (PIP), 0.2–0.3 (MovieLens)。

## 検証結果 (2026-09-05, RTX 4090, 他プロセスと VRAM 共有中)

| 確認項目 | 結果 |
|---|---|
| `smoke_clip.py` (CLIP-L 単体) | α=0 で元エンコードと一致 (max diff 1e-5)。α=0.4 で特徴が参照側に移動 (cos: target 0.90 / ref 0.60, 元は ref 0.34) |
| SD1.5 単一参照 α=0 / 0.4, seed 42 | `image/sd15_compare.png` (左: 通常, 右: パーソナライズ)。ポスター風の劇的な空・色調に変化、被写体は保持 |
| SD1.5 参照 5 件 + `--sample_size 1` (Tailored profiling) | `image/sd15_multiref_profiling/00000.png` 正常終了 |
