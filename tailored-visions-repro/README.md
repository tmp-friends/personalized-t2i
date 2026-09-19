# Tailored Visions — 公式実装のセットアップと再現実装

**Tailored Visions: Enhancing Text-to-Image Generation with Personalized Prompt Rewriting**
(Chen et al., CVPR 2024, [arXiv:2310.08129](https://arxiv.org/abs/2310.08129)) を
このマシンで動く状態にしたものです。

手法の中身: あるユーザが過去に書いたプロンプト群から、いま入力された短いプロンプトに
関連するものを検索し、それを「このユーザの好み」の証拠として LLM に渡して
プロンプトを書き換えてから Stable Diffusion に投げます。生成モデル自体は触りません。

```
短いクエリ ──► retriever ──► このユーザの過去プロンプト top-k
                                        │
                指示文 + デモ例 + ────────┘ ──► LLM 書き換え ──► SD v1-5
```

---

## ディレクトリ構成

| | 中身 |
|---|---|
| **`official/`** | **公式実装 (zzjchen/Tailored-Visions) + 2026年に動かすためのパッチ。動作確認済み。** |
| `tv/`, `scripts/` | 自作の再現実装（評価パイプライン込み）。公式に足りない部分を埋めたもの |
| `data/user_data/` | PIP データセット (3,116 ユーザ / 300,255 プロンプト) |
| `docs/DEVIATIONS.md` | 論文・公式実装との差分と、その理由 |
| `.venv/` | 両方が共有する Python 環境 (`uv sync` 済み) |

---

## 1. 公式実装を動かす ← まずはこちら

詳細は **[`official/SETUP.md`](official/SETUP.md)**。

```bash
cd official
./run.sh demo 'a cat'              # 書き換え + 画像2枚 (元 / パーソナライズ後)
./run.sh demo 'a cat' --no-t2i     # 書き換えのみ、数秒
./run.sh main --limit_users=5      # PIP データセットで実験
./run.sh stop                      # ローカル LLM サーバを停止
```

`run.sh` が ChatGPT の代わりのローカル LLM サーバ (`Qwen3.5-4B`) を自動で立てます。
**OpenAI のキーがあるならそちらを使ってください** — 書き換えモデルは手法そのものなので、
論文の数値を再現したい場合はローカル LLM では合いません。

```bash
export TV_OPENAI_KEY=sk-... ; unset TV_OPENAI_BASE
python demo.py --input_prompt='a cat'
```

### 公式コードは配布状態では動きません

`git log`／`git diff` に全差分が残してあります（各所に `[2026 patch]` コメント）。
主なもの:

- `main.py` が `MY_BASE` を import しておらず、**全実行が `NameError`** で落ちる
- `main.py` が `retrieval=='full'` で分岐しているが CLI は `'ebr'`/`'bm25'` しか受け付けない。
  つまり **`--retrieval=ebr` は黙って BM25 で動いていた**（公開コードでは論文の EBR 行を再現できない）
- `language.py` が BM25 に生のクエリ文字列を渡しており、**1文字ずつ**のマッチに退化していた
- `demo.py` が参照する `demo_user.jsonl` がリポジトリに存在しない
- `runwayml/stable-diffusion-v1-5` は 2024年8月に Hub から削除済み
- `apiuse.py` が失敗時に無限リトライするため、キーが不正だと無言でハングする

全 17 項目の一覧は [`official/SETUP.md`](official/SETUP.md) §4。

---

## 2. 自作の再現実装（公式に足りない部分）

公式リポジトリは**書き換えと画像生成まで**で、論文の評価指標を回すスクリプトが
含まれていません。加えて PIP データセットの画像 URL は 2024年から死んでおり、
PMS が必要とするユーザ嗜好要約 `P_u` も同梱されていません。

そこを埋めたのが `tv/` + `scripts/` です。

```bash
./run_all.sh                    # 一式 (RTX 4090 で 3〜4時間)
./run_all.sh --skip-ablations   # Table 2 と画像指標のみ (約1.5時間)
```

各ステージは再開可能で、既に出力があるものはスキップします。

| ステージ | スクリプト | 出力 |
|---|---|---|
| 0 | `00_download_data.py` | `data/user_data/*.jsonl` |
| 1 | `01_prepare.py` | 履歴の重複除去 + CLIP ViT-L/14 埋め込み |
| 2 | `02_rewrite.py` | `results/rewrites/<method>.jsonl` |
| 3 | `03_subset.py` | 画像指標を計算する固定ユーザ集合 |
| 4 | `04_preferences.py` | PMS 用の嗜好要約 `P_u` |
| 5 | `05_generate.py` | 生成画像の CLIP 埋め込み |
| 6 | `06_evaluate.py` | `results/RESULTS.md`, `results/metrics.json` |
| 7 | `07_leakage.py` | `results/leakage.json` |
| 8 | `08_examples.py` | `results/EXAMPLES.md`（論文 Figure 8 相当） |

実装した手法は Table 2 の 7 行と、Table 4（retrieval top-k）・Table 5（ICL shot 数）の
アブレーション、Table 3 の入力長バリエーション（`--prompt-type`）です。

**現在の実行状態**: Table 2 の 7 手法のうち `shortened_prompt` / `promptist` /
`general_pr` が全 6,232 サンプルで完了済み（`results/rewrites/`）。残り 4 手法は
未実行です（`./run_all.sh` で続きから走ります）。

### 途中で分かったこと

いずれも `docs/DEVIATIONS.md` に根拠つきで記載しています。

1. **テストセットの約 1/3 が学習履歴にリークしています。** テストの正解プロンプトが
   同一ユーザの履歴に**完全一致で 35.6%**、token-F1 ≥ 0.8 で 49.1% 存在します。
   さらに 18.2% はクエリ自体が正解プロンプトと同一（データ生成時に 6 語以下の
   プロンプトは要約されないため）。ROUGE-L は履歴を参照する手法を過大評価します。
   `06_evaluate.py` はリーク除外後の列も併記します。
2. **モデルを使わない "Shortened Prompt" 行が論文と一致しません**（0.3964 vs 0.3268）。
   この行は入力をそのまま使うだけなので、データと指標設定だけで決まり、本来は完全一致する
   はずです。論文の ROUGE-L の前処理が公開物からは復元できないことを意味するので、
   絶対値の比較はできません。ベースラインからの差分で比較してください。
3. `rougeL` の実装が "." で文分割してから summary-level LCS を取るため、
   長い複数文の書き換えが +0.016 有利になります（4語のベースラインでは +0.0002）。
   `RESULTS.md` に 1 セグメント版も併記します。

---

## 引用

```bibtex
@inproceedings{chen2024tailored,
  title     = {Tailored Visions: Enhancing Text-to-Image Generation with Personalized Prompt Rewriting},
  author    = {Chen, Zijie and Zhang, Lichao and Weng, Fangsheng and Pan, Lili and Lan, Zhenzhong},
  booktitle = {CVPR},
  year      = {2024}
}
```

PIP データセットと 5 つの in-context デモ例は著者らのものです
（[公式リポジトリ](https://github.com/zzjchen/Tailored-Visions)）。
