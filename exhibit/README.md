# FAN — 同じ一文から、あなたの一枚を

好きな画像を選び、その画像の**どこが好きか**（色・光・描画・雰囲気）を指定すると、指定した側面に付いた**確認済みの説明文**を参照にして、同じお題・同じseed・同じ生成設定のまま Illustrious XL v2.0 が描き直すローカル展示デモ。
個人化は [FAN](https://github.com/Burf/FAN)（Foundation Encoders Are All You Need for Preference-Aware Personalization, CVPR 2026）公式実装。設計は [FAN個人化改善設計書](../docs/superpowers/specs/2026-09-21-fan-personalization-improvement-design.md)（旧 [FAN展示デモ設計](../docs/superpowers/specs/2026-09-21-fan-exhibition-demo-design.md) と矛盾する箇所は新しい方を優先）。

## 現在の状態

| 項目 | 状態 |
|---|---|
| 既定 encoder policy | `legacy_exhibit`（評価が揃うまで変更しない。設計 §3） |
| 展示で使う catalog | `catalog-v2`（4被写体 × 16表現の64枚）。**所有者判断 2026-09-22** で切り替え、`catalog-v1` は互換を残さず削除 |
| `catalog-v2` の確認 | 生成済み 64 / 確認済みの枚数は `configs/cards-v2-review.json` が決めます。未確認のカードは `/api/config` に出ず、preflight も不合格になります |
| encoding 数値検査 | 実施済み。`legacy_exhibit` は合格、公式 pooled の条件は α=0 基準で不合格（検出結果として保存） |
| 画像評価（screen / refine / heldout） | 実行状況は [docs/reports/fan-personalization/](../docs/reports/fan-personalization/) を参照 |
| 本人によるブラインド評価 | **未実施**。回答は収集していない（代答もしない） |

判定の最新状態は `docs/reports/fan-personalization/index.html` と `evidence.json` にあります。成果物がない項目は「未実施」であり、成功として扱いません。

## 起動

リポジトリのルートから実行します。

```bash
uv sync --project exhibit --locked
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/preflight.py --models
uv run --project exhibit uvicorn exhibit.app:app --host 127.0.0.1 --port 7860
```

ブラウザーで **http://localhost:7860** を開きます。検証記録はサーバー経由では http://localhost:7860/report/ 、ファイルでは [docs/reports/fan-personalization/](../docs/reports/fan-personalization/)。旧構成の記録は [docs/reports/fan-demo/](../docs/reports/fan-demo/)、さらに旧い ZIPP 構成は [docs/reports/zipp-demo/](../docs/reports/zipp-demo/index.html) に履歴として残しています（どちらも旧設定の実測なので、現行設定の結果として引用しません）。

GPU推論は FAN 環境 `fan-repro/.venv/bin/python` を別プロセスで使います（FAN の attention monkey-patch は transformers 5 系と非互換）。初回は FAN 環境を用意します。

```bash
cd fan-repro && uv sync --locked && uv run python scripts/prepare_upstream.py
```

別環境を使う場合は `EXHIBIT_GPU_PYTHON=/absolute/path/to/python`。アプリは1プロセスで起動してください（`--workers` は指定しません）。モデル・固定画像が準備済みならインターネット不要で、推論時はHF/Transformersのoffline modeを強制します。

## 操作

1. **ラウンド1**: 12枚から好きな画像を選びます。同じ被写体のカードは並べて表示し、1ラウンドに同じ被写体は最大3枚までです。
2. **側面の指定**: 選んだカードごとに、好きな側面（色・光・描画・雰囲気）を1つ以上指定します。「全部好き」は明示的なボタンです。未回答のカードがあると決定できません。強さ `strength` は1か2です。
3. **ラウンド2・3（任意）**: 未提示の画像から12枚を追加します。最大3ラウンド、合計 **3〜10枚**。すでに提示した画像は二度出ません。
4. **お題と生成**: 6お題から1つ選ぶと、パーソナライズなし（参照なしの通常生成）の4枚と、パーソナライズありの4枚を、同じseedの順で並べます。
5. **好みを調整する**: 結果の下で、選択・側面・`strength`・側面ごとの倍率 `aspect_gain`（0.5 / 1 / 2 の絶対値指定）を変更し、同じお題・同じseedで描き直せます。
6. **任意の回答**: 完成した比較に「どちらが好きか」をラベル付きで回答できます。これは補助情報で、ブラインド評価の勝率には混ぜません。
7. **中止**は比較が成立しないので run ごと破棄します。終了／90秒の無操作で一時データを削除し、処理中は無操作リセットを止めて120秒の処理期限を適用します。

同じ内容に戻して描き直すと、その体験中に生成済みの画像をそのまま表示します（`run.mode` が `exact-cache`）。

## セッションで残るもの・消えるもの

- **消える**: 選択・ラウンド履歴・request台帳・キャッシュ・生成画像。終了、`DELETE /api/sessions/{sid}`、90秒の無操作のいずれでも `outputs/sessions/<sid>/` ごと削除します。生成中に終了した場合はワーカー終了後に削除します。
- **残る**: 配布用の固定資産（`assets/`）、準備の記録（`outputs/preparation/`）、研究用評価（`outputs/fan-evaluation/`）、`outputs/preflight.json`、レポート（`docs/reports/fan-personalization/`）。
- 通常セッションのデータを研究用評価へ自動コピーすることはありません（設計 §9.6）。研究用の回答は匿名 `participant_id` で別ディレクトリに保存します。

## 設定の正は1箇所

エンコーダーの振る舞いは **`configs/fan-policies.json`** だけが持ちます。

```json
{"schema_version": 1, "default_policy_id": "legacy_exhibit",
 "policies": {"legacy_exhibit": {"alpha": 0.5, "skip": -2, "skip_pa": [0,1,2,3,4,5,6,7],
   "use_attn_mask": false, "pooled_mode": "plain",
   "profiling": {"mode": "all"}, "reference_unit": "aspect_phrase"}, ...}}
```

`configs/demo.json` の `fan` にはFAN環境のパス・upstreamの固定commit・decoder重みのsha256だけが残り、`alpha` / `skip` / `sample_size` / `skip_pa` / `use_attn_mask` は持ちません。APIが返す `alpha` も解決済み policy の値です。通常APIはクライアントからencoder設定やファイルパスを受け取りません。サーバーが `default_policy_id` を解決します。

`profiling` は公式FANへ渡す型が変わります。`{"mode":"all"}` は整数 `0`（選別しない）、`{"mode":"ratio","value":0.1}` は float、`{"mode":"count","value":4}` は整数です。選別は公式 `sample_reference` に委譲し、呼び出しごとの入力順・選択indexを trace に記録します。

個人化の同一性（cache key）には、catalogの内容hash・選択カード・側面・強さ・`aspect_gain`・参照textとweight・実効policy・FAN pin・adapter/decoder/tokenizer hash・お題の文・生成設定・seedが入ります。表示用の `policy_id` だけでは同一と判定しません。旧形式のcacheは再利用しません。

**上流からの意図的な逸脱**: `legacy_exhibit` は個人化時の pooled 埋め込みに参照なし（plain）の値を使います（`pooled_mode: "plain"`）。公式の `ClassTokenDecoder` はpadding tokenを終端と誤検出するためで、`official_encoder` では公式 pooled（`"fan"`）をそのまま使い、その差を評価で比較します。画像イベントに `pooled` として記録します。

## カタログと確認手順

展示が使うカタログは `catalog-v2` ひとつだけです（**所有者判断 2026-09-22**。旧 `catalog-v1` は互換を残さず削除しました。設計 §6.2 の「確認完了後に切り替える」ゲートを上書きした判断です）。定義は `configs/catalog-v2.json`（4軸 × 4水準、明示列挙した16組 × 4被写体 = 64枚）。画像は `assets/cards-v2/`、生成後カタログは `assets/catalog-v2.json`、確認記録は `configs/cards-v2-review.json`、生成ログは `outputs/preparation/catalog-v2/`。カタログは1つなので `catalog_id` の設定項目・`--catalog` 引数はありません。id 文字列 `catalog-v2` だけが manifest・確認記録・評価の同一性のために残っています。

```bash
# 生成（トークン検査に通らなければ生成前に失敗します）
PYTHONPATH=exhibit/src fan-repro/.venv/bin/python exhibit/scripts/check_card_tokens.py
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/prepare.py cards
# 1枚だけ差し替えるときは configs/catalog-v2.json の seed_overrides と
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/prepare.py cards --only <card_id>
# 確認用ページを作り、実画像を見て4側面それぞれを確認してから export する
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_review_sheet.py
```

確認記録は `reviewed` だけでなく**4側面それぞれの確認結果**と、対象の画像sha256・説明文hashを持ちます。画像や説明文を変えると確認は自動的に無効になります。未確認のカードは `/api/config` に出ず、preflightも不合格になります。埋め合わせのために未確認画像を表示しません。64枚すべての確認は待たず、確認できたカードだけで展示を回します（設計 §6.1）。

`configs/cards-v2-review.json` は展示の所有者が確認ページから export したものだけを置きます。他の誰も（agentも）書きません。

## 固定assetの準備

重みは `configs/demo.json` の固定revisionを準備時に取得します。起動時downloadは行いません。

```bash
fan-repro/.venv/bin/python - <<'PY'
import json
from huggingface_hub import hf_hub_download, snapshot_download
settings = json.load(open("exhibit/configs/demo.json"))["generation"]
hf_hub_download(
    settings["model"], filename=settings["checkpoint"], revision=settings["revision"]
)
config = settings["pipeline_config"]
snapshot_download(
    config["model"], revision=config["revision"],
    allow_patterns=["model_index.json", "scheduler/*.json", "unet/config.json",
                    "vae/config.json", "text_encoder*/config.json", "tokenizer/*", "tokenizer_2/*"],
)
PY
```

```bash
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/prepare.py cards      # catalog-v2 の64枚
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/prepare.py generic
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/prepare.py samples
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_fallback.py
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/preflight.py --models
```

`prepare.py samples` は代表3選択 × 2お題の6サンプルを、確認済みカードと既定 policy の実際の参照から作ります（`configs/demo.json` の `sample_ids` が必須の6件を宣言します）。選択は `scripts/prepare.py` の `SAMPLE_SELECTIONS` にあり、いまは s1: 暖色 × 強い日差し（color+lighting）、s2: 寒色 × 油彩（color+texture）、s3: 逆光 × 線のない平塗り（lighting+texture, strength 2）です。サンプルは選択が参照しているカードが確認済みでないと作れず、参照・重み・hash が合わなければ preflight と `/api/config` から外れます。

`assets/fallback.html` は実画像を埋め込んだ単独HTMLで、サーバーが停止していても開けます。サンプルは代表的な選択の事前生成であり、来場者の選択を反映した結果としては表示しません。固定展示画像とサンプルは配布用assetとしてGit対象、モデルとセッション一時生成物は `outputs/` 配下でGit対象外です。

## 評価（設計 §10）

リポジトリのルートで実行します。準備以外はオフラインです。GPUを使うコマンド同士は同時実行しないでください（同じ lease を共有します）。

```bash
# 評価器（凍結CLIP）の準備と固定manifest
PYTHONPATH=exhibit/src fan-repro/.venv/bin/python exhibit/scripts/prepare_evaluation.py
# encoder設定の数値検査（α=0、重複参照、共通倍率、例外後の復元）
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py encoding --config exhibit/configs/fan-evaluation.json
# 8条件の絞り込み → alphaの追加比較 → heldout
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py screen  --config exhibit/configs/fan-evaluation.json --resume
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py refine  --config exhibit/configs/fan-evaluation.json --resume
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py heldout --config exhibit/configs/fan-evaluation.json --resume
```

- 固定fixture: `configs/fan-evaluation.json`、`configs/evaluation-cases/`。準備manifest: `outputs/fan-evaluation/preparation.json`。
- 実測: `outputs/fan-evaluation/<experiment_hash>/`（`records.json` / `metrics.json` / `decision.json` / `summary.json`）。
- `--resume` は同じ manifest hash の完了済み画像だけを再利用します。1 experiment の上限は512枚で、超えるmatrixは実行前に失敗します。
- 集計は成功例だけを抜き出しません。欠損率と理由を必ず出します。`heldout` は64枚全数の確認までは求めず、設計§9.2の3条件（参照カードが確認済み・各水準2枚以上・合計32枚以上）を満たせば実行できます。

### 人によるブラインド評価

```bash
# 1. 回答収集ページと実施説明を作る（participants.json が無ければ「比較画像未生成」と記録）
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_preference_study.py \
  --config exhibit/configs/fan-evaluation.json [--study-dir DIR]
# 2. 参加者のexportを検証してmergeする
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_preference_study.py \
  --config exhibit/configs/fan-evaluation.json --merge export1.json export2.json ...
# 3. participants.json ができたら、もう一度 build して manifest と比較ページを作る
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_preference_study.py \
  --config exhibit/configs/fan-evaluation.json
# 4. 比較画像を生成する（同じlease・同じmanifest契約）
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py study \
  --config exhibit/configs/fan-evaluation.json --resume
# 5. 回答JSONを study/answers/ に置いてから集計する
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/summarize_preference_study.py \
  --study exhibit/outputs/fan-evaluation/study
```

収集 → merge → manifest → 画像生成 → 回答 → 集計の順です。`participants.json` の必須項目は `participant_id, catalog_id, selection, aspect_gains` で、各参加者につき1件。旧16枚 pool と比較する `elicitation` study（設計 §9.7）は、比較対象だった catalog-v1 の削除にともない削除しました。回答は参加者内で平均してから参加者間で平均し、seed数を人数に加算しません。95%区間は参加者単位のbootstrap（2,000回、seed=0）です。回答が無ければ「未実施」と出力し、勝率は作りません。方式の対応表は集計器だけが読む別ファイルです。

既定値の変更は、heldoutの事前基準と本人評価の両方（候補vslegacy、本人vs別人のどちらも平均>0.5かつ95%区間下限>0.5、主評価は最低20人）を満たしたときに1回だけ行い、根拠の experiment / study hash を設定・README・レポートへ残します。満たさない間は `default_policy_id` を `legacy_exhibit` のままにします。

## 検証

```bash
PYTHONPATH=exhibit/src exhibit/.venv/bin/python -m pytest exhibit/tests -q
node --test exhibit/tests/test_browser_state.mjs
uv run --project exhibit ruff check exhibit && uv run --project exhibit ruff format --check exhibit
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/preflight.py --models
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/browser_check.py --report-dir docs/reports/fan-personalization
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/rehearsal.py --sessions 20 --url http://127.0.0.1:7860
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_report.py
```

- `rehearsal.py` は1セッションで「作成 → ラウンド1 → 側面つきの決定 → 生成 → gain変更で同一seed再生成 → 同じ内容へ戻して `exact-cache` → 生成中の中止 → 削除」を通し、成功/失敗・p50/p95・各runの120秒期限・peak VRAM（同じマシンのワーカーログから読めた場合）を記録します。`--url` を省くとプロセス内のアプリを直接駆動して実GPUを使います。
- `browser_check.py` は実Chromiumで同じ一連の操作を確認します。`--url` を省くと `exhibit/tests/mock_api.mjs` を自分で起動するためGPUを使いません（その場合は実機確認としては扱いません）。Chromiumは `EXHIBIT_CHROMIUM` で指定できます。
- `build_report.py` はディスク上の成果物だけからレポートを組み立て、実装完了 / 実機検証 / 精度実証を別々に判定します。何度実行しても、同じ入力からは同じ判定になります。

## 生成モデルとモチーフ

[Illustrious XL v2.0-STABLE](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0)（revision `69459c1fe6f46db41ab31e6114f05acc0e06bcaa`）を使用します。1024×1280（縦長）・30 steps・DPM++ 2M SDE Karras（`DPMSolverMultistepScheduler` + `algorithm_type: sde-dpmsolver++` / `use_karras_sigmas`）・CFG 5.0・fp16。デコーダーは固定revisionの [sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)（`207b116dae70ace3637169f1ddd2434b91b3a8cd`）に差し替えます（チェックポイント同梱のVAEをfp16で使うと白っぽく低コントラストになるため）。単一safetensorsを `from_single_file` で読み込み、構成ファイルとtokenizerだけを初期Illustriousの固定revisionから読みます。`from_single_file` は `name_or_path` を残さないため、FANのSDXLエンコーダーは `FAN(text_encoder, tokenizer, L.pth)` と `FAN(text_encoder_2, tokenizer_2, bigG.pth)` を `fan.wrapper.stable_diffusion_xl` で束ねて手動で組み立てます。

お題は「窓辺で猫と過ごす少女」「東京の夜景と青年」「森を旅する魔法使い」「海辺の灯台と船乗り」「雨の街角の少女」「カフェで迎える店員」の6件。カードの被写体（街角の少女・図書館の青年・草原の旅人・カフェの店員）はお題と重ねていないため、「被写体が好き」と「表現が好き」を切り分けられます。通常側と個人化側のモデル・設定・seedは一致させます。比較中は `generation` を固定し、変更した場合は別 experiment として扱い、過去の結果に追記しません。

## APIメモ

| API | 入力と動作 |
|---|---|
| `POST /api/sessions` | 空の draft、`revision=0`、ラウンド1を返す |
| `PUT /api/sessions/{sid}/selection` | `expected_revision, cards, aspect_gains, commit`。`cards` は `card_id, strength, aspects`。全置換で、内容が変わったときだけ revision を1増やす |
| `POST /api/sessions/{sid}/rounds` | `request_id, expected_revision`。未提示の次ラウンド。3ラウンド超過は409 |
| `POST /api/sessions/{sid}/runs` | `topic_id, request_id, expected_revision`。決定済みの選択からサーバーが policy を解決して開始 |
| `PUT /api/sessions/{sid}/runs/{rid}/feedback` | `expected_revision, preference`（`plain` / `personal` / `tie`）。完了済みの当該runだけ |
| `POST /api/sessions/{sid}/cancel` · `DELETE /api/sessions/{sid}` | 子プロセス終了待ち、lease解放、後処理 |

- 未知のカード・側面、未確認カード、不正な `strength`/`gain`、重複カード、未知フィールドは422。古い revision、生成中の変更、`request_id` の別内容への再使用は409。
- 同じ `request_id` の再送は同じ結果を返し、カードを二度見せません。
- 1 run = `run.plain`（なし4枚、事前生成キャッシュ）+ `run.personal`（あり4枚、できた順）。画像は `/api/sessions/{sid}/images/<relative_path>` で配信します。
- `run.mode` は `live | exact-cache | sample`。`/api/config` の `schema_version` が一致しない古い画面には再読み込みを案内します。

## 説明で守っていること

- 「来場者ごとの追加学習なし」と言います。「追加モデル・重みが一切ない」とは言いません（FAN公式の `ClassTokenDecoder` を使います）。
- 参照は「選んだ画像に付けた確認済みの説明文」です。画像そのものをエンコーダーへ入れているとは説明しません。参照に被写体語は含めません。
- 「外す」は重み0ではなく参照リストからの除去です。嫌い・未選択を負の重みには変換しません。
- 反映を強くするほど良いとは言いません。Attentionの値から「この色はこの画像由来」といった因果説明もしません。
- 論文の定量結果をこの展示の性能として扱いません。Illustriousと選んだカードで公式encoder処理を再現しても「論文の定量結果を再現した」とは書きません。
- 旧 `outputs/fan-probe/` と `docs/reports/fan-demo/evidence.json` は旧設定の実測です。現行設定の結果として引用しません。
- 削除した catalog-v1 の実測（旧16枚のカード・旧6サンプル）も、現行 catalog-v2 の結果として引用しません。
- 未実施の評価を成功として扱いません。人の回答が要る評価は代答しません。
