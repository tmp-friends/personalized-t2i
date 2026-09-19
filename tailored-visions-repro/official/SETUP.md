# 公式実装 (zzjchen/Tailored-Visions) の動作確認済みセットアップ

CVPR 2024 "Tailored Visions" の**公式実装をそのまま clone し、2026 年の環境で動くように
最小限のパッチを当てた**ものです。`git log` / `git diff` で上流との差分がそのまま読めます。

- 上流: https://github.com/zzjchen/Tailored-Visions (commit `d0f4454`)
- 論文: https://arxiv.org/abs/2310.08129

---

## 1. すぐ動かす

Python 環境は親ディレクトリの `.venv` を共有しています（`uv sync` 済み）。

`run.sh` がローカル LLM サーバの起動込みで面倒を見ます（ターミナル1つで完結）。

```bash
cd ~/stable-diffusion/personalized-t2i/tailored-visions-repro/official

# デモ: 1ユーザの履歴からプロンプトを書き換えて画像を生成
./run.sh demo 'a cat'
#   -> ori_prompt.png (元プロンプト) / personalized_prompt.png (書き換え後)

# 書き換えだけ（画像生成なし、数秒）
./run.sh demo 'a cat' --no-t2i

# PIP データセットで実験。まず 5 ユーザで様子を見る
./run.sh main --limit_users=5
./run.sh main --limit_users=5 --t2i
#   -> image_result/<日時>/<user>-<query>-<n>/ に ori.png, generated.png, result.txt

# 全 3,116 ユーザ (--limit_users を省略)。--t2i 付きだと数日かかります
./run.sh main

./run.sh stop        # ローカル LLM サーバを止める
```

素の公式コードを直接叩く場合はこちら:

```bash
PY=../.venv/bin/python
$PY serve_local_llm.py &                        # ターミナル1
export TV_OPENAI_BASE=http://127.0.0.1:8000/v1  # ターミナル2
export TV_OPENAI_KEY=local
$PY demo.py --input_prompt='a cat'
$PY main.py --retrieval=ebr --num_retrieval=3 --rewrite_method=ICL --ICL_shot=1 --t2i
```

**本物の ChatGPT を使う場合**は、サーバを立てずに環境変数だけ差し替えてください。
コードの変更は不要です。

```bash
export TV_OPENAI_KEY=sk-...          # OPENAI_API_KEY でも可
export TV_OPENAI_MODEL=gpt-4o-mini   # 省略時 gpt-3.5-turbo
unset TV_OPENAI_BASE                 # 公式エンドポイントに戻す
```

---

## 2. 準備済みのもの

| 項目 | 状態 |
|---|---|
| PIP データセット (3,116 ユーザ / 30万プロンプト) | `user_data/` → `../data/user_data` のシンボリックリンク |
| CLIP ViT-L/14, ViT-B/32 | `~/.cache/clip/` にダウンロード済み |
| Stable Diffusion v1-5 | HuggingFace キャッシュに取得済み |
| `demo_user.jsonl` | `demo.py` 初回実行時に `user_data/87403.jsonl` から自動生成 |
| 動作確認 | `demo.py` / `main.py` を {ebr, bm25} × {naive, ICL} × {--t2i あり/なし} で実行済み。`error.jsonl` は全て 0 バイト |
| OpenAI SDK | 上流が想定する `openai==0.27.8` をそのまま使用（`apiuse.py` の呼び出しは無改造） |

---

## 3. `serve_local_llm.py` について

上流は書き換えを ChatGPT に投げます。API キーがない環境でも動作確認できるよう、
**同じ `/v1/chat/completions` を喋るローカルサーバ**を追加しました
（`Qwen/Qwen3.5-4B` をローカル実行）。`apiuse.py` の呼び出し経路は上流のままで、
向き先が変わるだけです。

- リクエストは 50ms 単位でまとめてバッチ推論します。`main.py` は 1 件ずつ投げてくるので、
  これがないと 6,232 サンプルの実行が現実的な時間で終わりません。
- Qwen3.5 は既定で `<think>` ブロックを出力します。これが書き換え結果に混入すると
  評価指標が静かに壊れるため、`enable_thinking=False` を指定しています。
- **論文の結果を再現したい場合はローカル LLM ではなく本物の ChatGPT を使ってください。**
  書き換えモデルは手法そのものなので、モデルを替えれば数値は変わります。

VRAM: サーバが約 9GB、`demo.py` 側が CLIP + SD で約 5GB。24GB あれば同時に動きます。
`run.sh` から起動した場合は `nohup` でバックグラウンドに残るので、**使い終わったら
`./run.sh stop` で止めてください**（放置すると 9GB 掴んだままになります）。

なお認証はありません。`127.0.0.1` のみに bind しているのでネットワークからは見えませんが、
同じマシンに他のユーザがいる場合はそのユーザからは叩けます。

---

## 4. 当てたパッチ（すべて上流の実バグ）

`git diff` で全文が読めます。各箇所に `[2026 patch]` コメントを付けています。

| ファイル | 内容 |
|---|---|
| `main.py` | `MY_BASE` を import していないため**全実行が `NameError`** で落ちる |
| `main.py` | `if self.retrieval=='full'` — CLI は `'ebr'`/`'bm25'` しか受け付けないので、**`--retrieval=ebr` が黙って BM25 で動いていた**（＝公開コードでは論文の EBR 行を再現できない） |
| `main.py` | `clip_model,=None` → `--retrieval=bm25` で `TypeError` |
| `main.py` | `--t2i` を付けない場合に `None.to(device)` で `AttributeError` |
| `main.py` | ICL テンプレートがクエリ非依存に作られていた。デモ例はクエリとの類似度順に並べる仕様（論文 4節）なのでクエリ毎に再構築 |
| `main.py` | `os.mkdir` が `image_result/` 未作成時に失敗 |
| `main.py`, `demo.py`, `SD.py` | `runwayml/stable-diffusion-v1-5` は 2024年8月に Hub から削除。同一重みの再ホスト先に変更 |
| `demo.py` | `StableDiffusionPipeline` を未 import のまま使用 |
| `demo.py` | README は `--input_prompt` だがコードは `--prompt` のみ受付。両方に対応 |
| `demo.py` | `--t2i` が `store_true` かつ `default=True` で**オフにできない**。`--no-t2i` を追加 |
| `demo.py` | リポジトリに存在しない `demo_user.jsonl` を参照。データセットから自動生成するように |
| `language.py` | BM25 に**生のクエリ文字列**を渡しており `for q in query` が**1文字ずつ**回っていた（＝文字単位マッチに退化）。文書と同じく単語分割するよう修正 |
| `language.py` | `sorted(..., key=lambda x: scores[sentences.index(x)])` — 同一プロンプトがあるとスコアを取り違え、かつ O(n² log n) |
| `prompts.py` | `rank_examples` がクエリ毎に CLIP を再ロード。1回だけロードするようキャッシュ |
| `apiuse.py` | 失敗時に**無限リトライ**するため、キーが不正だと無言でハングする。5回で打ち切り |
| `apiuse.py` | キー/エンドポイントを環境変数から読めるように（ソース編集不要） |
| `download.py` | `@set_timeout(5, after_timeout())` — デコレート時にコールバックを**呼んで**しまい、import するだけで `Time out!` を出力。かつ callback に `None` が入るため実際のタイムアウト時に `TypeError` |
| `main.py` | `--data_folder` / `--limit_users` を追加。上流は必ず全 3,116 ユーザを走査し、途中で止める手段がなかった |
| `requirements.txt` | 2023 年のピン（torch 1.13.1 / numpy 1.21.5 / spacy 3.6.1）が Python 3.12 で解決不能。`requirements-2026.txt` を追加 |

パッチを当てていない既知の問題（挙動を変えるため、あえて上流のまま）:

- `language.py:117` `list(set(sentences))` が `checklines` の後段で順序を壊し、
  プロンプトと `image_urls` の対応がズレる。画像 URL が既に死んでいるので実害なし。
- `prompts.py` の ZS テンプレートが指示文と履歴ブロックを改行なしで連結している。

---

## 5. 既知の制約

- **データセットの画像 URL は死んでいます。** `result_url` の `cdn1.printidea.art` は
  2024年から到達不能で、上流 README もそれを認めています。論文の `Image-Align`
  （生成画像 vs ユーザが実際に保存した画像）は**そのままでは計算できません**。
- `metrics.py` に `PMS` / `Image-Align` の関数はありますが、**評価を回す
  スクリプトは上流に含まれていません**。`main.py` は書き換えと生成までです。
- ユーザ嗜好 `P_u`（PMS が必要とする 5 フレーズ要約）もデータセットに同梱されていません。

このあたりを埋めた評価パイプライン（PMS / Image-Align 代替 / ROUGE-L β=5、リーク検査つき）を
親ディレクトリに別途実装してあります。`../README.md` と `../docs/DEVIATIONS.md` を参照してください。
