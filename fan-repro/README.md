# FAN — 公式実装のローカル実行

## 1. 手法概要

**Foundation Encoders Are All You Need for Preference-Aware Personalization**
（Kim, Ahn, Seo, CVPR 2026）の公式実装を、再現可能な形でローカル実行する。

FAN はターゲット prompt と好みを表す参照 prompt 群をテキストエンコーダへ同時に入力し、
self-attention を Personalized Attention に差し替えて、参照の嗜好を強度 α で混ぜる。
拡散モデル本体とエンコーダの追加学習は不要である。

## 2. 現在の再現状況

| 実装 | パス | 状態 |
|---|---|---|
| 公式実装 | `upstream/` | 固定 SHA の clean submodule |
| ローカル生成 wrapper | `scripts/generate.py` | SD1.5 で生成確認済み |
| CLIP smoke test | `scripts/smoke_clip.py` | α=0 の一致と α=0.4 の嗜好移動を確認済み |
| 詳細ノート | `docs/summary.html` | 論文、実装、検証値を記録 |

## 3. Quick start

```bash
git submodule update --init --recursive
cd fan-repro
uv sync --locked
uv run python scripts/prepare_upstream.py
uv run python scripts/generate.py --help
```

`prepare_upstream.py` は submodule の SHA を検証し、実行用コピーを `.work/upstream/` に生成する。

## 4. ディレクトリ構成

```text
fan-repro/
├── upstream/                 # 公式 Git submodule（直接編集しない）
├── patches/series            # 現在は patch なし
├── .work/upstream/           # 実行用コピー（Git 対象外）
├── scripts/generate.py       # ローカル生成 CLI
├── scripts/smoke_clip.py     # CLIP 単体 smoke test
├── scripts/prepare_upstream.py
├── docs/summary.html
├── tests/test_layout.py
├── pyproject.toml
└── outputs/                  # 生成物（Git 対象外）
```

## 5. 公式実装と固定コミット

- URL: <https://github.com/Burf/FAN>
- 固定コミット: `9d0b76843f6437718195accac9cf3f050a25d26b`
- submodule: `upstream/`
- 実行ソース: `.work/upstream/`
- source patch: なし

`weight/L.pth` と `weight/bigG.pth` は公式 submodule に同梱された OpenCLIP 用 class-token detector で、
ローカル CLI は `.work/upstream/weight/` から参照する。

## 6. 生成・確認手順

Stable Diffusion v1.5 の確認済み例:

```bash
uv run python scripts/generate.py   --model stable-diffusion-v1-5/stable-diffusion-v1-5   --variant fp16 --dtype float16   --prompt "A photograph of an astronaut riding a horse"   --ref "A retro-futuristic space exploration movie poster with bold, vibrant colors"   --alpha 0.4 --seed 42 --save_path outputs/sd15_alpha0.4
```

CLIP テキストエンコーダだけを確認する場合:

```bash
uv run python scripts/smoke_clip.py
```

`--ref` は複数指定でき、`--weight` で参照ごとの強度を指定する。参照が多い場合は
`--sample_size 1` で論文の Tailored profiling（履歴の 10% を選別）を有効にできる。

## 7. 動作確認結果

2026-09-05、RTX 4090 24 GB での結果:

| 確認項目 | 結果 |
|---|---|
| CLIP-L smoke test | α=0 で元エンコードと一致（max diff 1e-5） |
| α=0.4 | 特徴が参照側へ移動（target cos 0.90、ref cos 0.60、元の ref cos 0.34） |
| SD1.5、単一参照 | 約 10 秒/枚。被写体を保ち、ポスター風の空・色調へ変化 |
| SD1.5、参照 5 件 + profiling | 正常終了 |

SD1.5 の重みと当時の生成画像は 2026-09-19 にディスク整理で削除したため、再実行時は
重みの再取得が必要。ローカルにある SDXL base 1.0（fp16）は未検証である。

## 8. 論文・公式実装との差分

- 公式 source は変更せず、clean submodule として固定した。
- `scripts/generate.py` は公式 `inference.py` の引数を保ちつつ、`--variant`、`--dtype`、
  `--ref_weight_dir` と `outputs/` 既定値を外側の CLI として提供する。
- `transformers>=4.57,<5` に固定する。公式コードが monkey-patch する CLIP/T5 attention API は
  transformers 5 系と互換性がない。
- FLUX.1-dev は公式の `pipeline.to("cuda")` では 24 GB に収まらず、offload または量子化が別途必要。
- SD3.5、SDXL、unCLIP、CLIP 検索、LLaVA 経路は未検証。

patch 判断と未検証範囲は [docs/DEVIATIONS.md](docs/DEVIATIONS.md) にまとめる。

## 9. データ、重み、生成物

| 種類 | パス | Git |
|---|---|---|
| 公式 source / detector weight | `upstream/` | submodule の gitlink を追跡 |
| patch 適用済み実行 tree | `.work/upstream/` | 追跡しない |
| 生成画像 | `outputs/` | 追跡しない |
| Python 環境 | `.venv/` | 追跡しない |

画像から好みを与える場合は、画像を caption / tag 化して `--ref` に渡すか、公式 notebook の
unCLIP 経路を使う。負の `weight` は論文・公式実装の想定外である。

## 10. 引用

```bibtex
@inproceedings{kim2026foundation,
  title     = {Foundation Encoders Are All You Need for Preference-Aware Personalization},
  author    = {Kim, Hyungjin and Ahn, Seokho and Seo, Young-Duk},
  booktitle = {CVPR},
  year      = {2026}
}
```
