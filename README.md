# personalized-t2i — 嗜好パーソナライズ × 画像生成

数枚の評価やプロンプト履歴から**ユーザの美的嗜好**を推定し、生成に反映する手法
(preference personalization) の実装・検証。「生成AIなんでも展示会」向けの作業場所。

特定の人物や物を再現する subject personalization (DreamBooth / LoRA 系) とは別系統で、
ここにあるのは全て「好みを推定して反映する」側。

いずれも**公式実装を動く状態にしたもの**が本体で、独自の追加分はその上に載せている。

| フォルダ | 論文 | 手法の位置づけ | まとめ |
|---|---|---|---|
| [`tailored-visions-repro/`](tailored-visions-repro/) | Tailored Visions (CVPR 2024)<br>[arXiv:2310.08129](https://arxiv.org/abs/2310.08129) | **プロンプト書き換え型**。履歴を検索して LLM に渡し、プロンプトを書き換える。生成モデルは触らない | [`docs/tailored-visions.html`](tailored-visions-repro/docs/tailored-visions.html) |
| [`premier-repro/`](premier-repro/) | Premier (CVPR 2026 Highlight)<br>[arXiv:2603.20725](https://arxiv.org/abs/2603.20725) | **ユーザ埋め込み注入型**。学習可能な user embedding を Preference Adapter でテキスト条件と融合。新規ユーザは既存埋め込みの線形結合 | [`docs/summary.html`](premier-repro/docs/summary.html) |
| [`FAN/`](FAN/) | FAN (CVPR 2026)<br>Kim, Ahn, Seo | **エンコーダ改変型・学習不要**。テキストエンコーダの self-attention を Personalized Attention に差し替え、参照プロンプト群の嗜好を強度 α で混ぜる。拡散モデル本体もエンコーダも追加学習なし | [`FAN_summary.html`](FAN/FAN_summary.html) |

各フォルダの `README*.md` / `official/SETUP.md` に起動手順、`docs/DEVIATIONS.md` 等に
論文・公式実装との差分がある。

## 共通の注意

- **Python 環境はフォルダごとに独立**（それぞれ `.venv`）。`uv sync` 済みで、
  `<folder>/.venv/bin/python` を直接叩く運用。
- **GPU は 1 枚（RTX 4090 / 24GB）**。同時に 2 プロジェクト動かすと VRAM が足りない。
  特に `tailored-visions-repro/official/run.sh` が起動するローカル LLM サーバは
  約 9GB を掴んだまま常駐するので、**使い終わったら `./run.sh stop`**。
- このフォルダは 2026-09-05 に `~/stable-diffusion/` 直下の3フォルダをまとめたもの。
  公式実装は各手法の `upstream/` submodule に固定し、ローカル変更はその外側で管理する。

## 手法の対比（展示のネタとして）

同じ「パーソナライズ」でも介入する層が違う。

```
プロンプト層   Tailored Visions   履歴を検索 → LLM がプロンプトを書き換え
                                  学習不要・軽い・説明可能
条件付け層     Premier            学習可能な user embedding を adapter でテキスト条件に融合
                                  ユーザ埋め込みの学習が要る

エンコーダ層   FAN                text encoder の self-attention を差し替え、
                                  参照プロンプトの嗜好を混ぜる。追加学習なし
```

展示の差別化候補は「嗜好の言語化と可視化」「"嫌い" の活用」「デフォルト生成との比較提示」
「AI の解釈をユーザが修正できること」。詳細は各 `docs/`。
