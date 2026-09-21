# FAN — 同じ一文から、あなたの一枚を

好きな画像を3〜5枚選ぶと、その画像に付けた**確認済みの説明文**を参照にして、同じお題・同じseed・同じ生成設定のまま Illustrious XL v2.0 が描き直すローカル展示デモ。
個人化は [FAN](https://github.com/Burf/FAN)（Foundation Encoders Are All You Need for Preference-Aware Personalization, CVPR 2026）公式実装。設計は [FAN展示デモ設計](../docs/superpowers/specs/2026-09-21-fan-exhibition-demo-design.md)。

## 起動

リポジトリのルートから実行します。

```bash
uv sync --project exhibit --locked
uv run --project exhibit python exhibit/scripts/preflight.py --models
uv run --project exhibit uvicorn exhibit.app:app --host 127.0.0.1 --port 7860
```

ブラウザーで **http://localhost:7860** を開きます。検証記録はサーバー経由では http://localhost:7860/report/ 、ファイルでは [docs/reports/fan-demo/](../docs/reports/fan-demo/)。旧 ZIPP 構成の記録は [docs/reports/zipp-demo/](../docs/reports/zipp-demo/index.html) に履歴として残しています。

GPU推論は FAN 環境 `fan-repro/.venv/bin/python` を別プロセスで使います（FAN の attention monkey-patch は transformers 5 系と非互換のため、`tailored-visions-repro/.venv` は使いません）。初回は FAN 環境を用意します。

```bash
cd fan-repro && uv sync --locked && uv run python scripts/prepare_upstream.py
```

別環境を使う場合は `EXHIBIT_GPU_PYTHON=/absolute/path/to/python`。FAN 本体の取得先とコミット、decoder 重みの sha256 は `configs/demo.json` の `fan` に固定しています。

アプリは1プロセスで起動してください。`--workers` は指定しません。モデル・固定画像が準備済みならインターネット不要です。推論時はHF/Transformersのoffline modeを強制します。

## 操作

1. 4被写体 × 4表現の16枚から好きなカードを3〜5枚選びます。任意で「どこが好き？」から色・光・描画・雰囲気の側面を外せます（外した句は参照から消えます）。
2. 6お題から1つ選び「描く」を押すと、通常生成（参照なし）と個人化生成（`alpha=0.5`）をseedごとに1対ずつ表示します。**どちらがどちらかは伏せたまま**、好きな方か「決められない」を選びます。
3. 4対に答えるか「答えを見る」で reveal。使った参照・説明文・`alpha`、そしてお題の文が変わっていないことを表示します。
4. reveal 後に `alpha`（弱0.35/中0.5/強0.6）と参照ごとの重み（通常1.0/重視2.0/外す）を変えて描き直せます。1 run につき variant は最大3、同じ設定は「同一条件のキャッシュ」として再利用します。
5. 終了／90秒の無操作で一時データを削除します。処理中は無操作リセットを停止し、120秒の処理期限を適用します。
6. 「中止」はワーカーを終了させます。ブラインド比較（v0）を答え合わせ前に中止した場合は、何も見せていないので run ごと破棄し、選び直しへ戻ります。reveal 後の描き直し（v1以降）を中止した場合は、その variant だけを中止扱いにし、run と完成済みの画像は残します。
7. 答え合わせ後（またはサンプル表示中）は、別のお題を選ぶと新しい run として最初から比較し直します。生成中のお題変更は受け付けません。

参照が0件になる操作（全部「外す」）は受け付けません。最低1件です。

## 現在の採用モード

**FAN実生成 + ブラインド比較**。来場者ごとの追加学習はしません。

- 参照は「選んだ画像に付けた確認済みの説明文」です。画像そのものをエンコーダーへ入れてはいません。参照には被写体語を含めず、表現の句だけを使います。
- FAN 公式実装の `ClassTokenDecoder`（`weight/L.pth` / `weight/bigG.pth`）を使います。「追加モデル・重みが一切ない」とは説明しません。
- 通常も個人化も同じ `prompt_embeds` 経路です。参照なしのFANエンコードは `pipe.encode_prompt` と一致することをG0スパイクで確認しました。負のプロンプトは常に参照なしで1回だけエンコードし、全画像で使い回します。
- **上流からの意図的な逸脱**: 個人化時の pooled 埋め込みは、`ClassTokenDecoder` がpadding tokenを終端と誤検出するため使わず、同じ文の参照なし pooled を使います（hidden states は個人化、pooled は plain）。画像イベントに `pooled: "plain"` として記録します。
- 「外す」は重み0ではなく参照リストからの除去として実装しています。
- 参照は**側面ごとの短い句**を重複排除して渡します。同じ句が複数のカードから来た場合は1件にまとめ、weightを合算します（例: `calm atmosphere` が2枚から選ばれれば weight 2.0、片方が「重視」なら 3.0）。

### 検証で決めた設定

G0の追試で次を確定しました。`skip_pa=[0,1,2,3,4,5,6,7]`（personalized attentionを前半8層で行わない）にすると、参照全体に掛かっていた一般的な「もや」が消え、α=0.6まで被写体・構図・衣装などの指定が保たれます。`use_attn_mask=true` はα=0（参照の影響ゼロのはず）でも target のエンコードを変えてしまい（plainとのcos類似 0.59）、無効のまま固定します。参照は1枚1文の長い束ね方をやめ、側面ごとの短い句を重複排除・weight合算で渡します（長い束ね方は構図が大きく振られました）。pooled は個人化せず plain を使います（上流のClassTokenDecoderがpadding tokenを終端と誤検出するため）。追試では α=0.7 で「東京の夜景と青年」の人物指定（1boy）が崩れ、0.8 では緑の瞳が失われたため、強は 0.6 を上限にしています。warm_soft の光の句は `warm golden hour light, gentle shadows` にすると暖色と線の鮮明さが保たれたため採用しました。これらは `alphas`（弱0.35/中0.5/強0.6）とともに `configs/demo.json` の `fan` に固定し、personalization hash に含めています。

実測は `exhibit/outputs/fan-probe/followup/followup.json`（`outputs/` はGit管理外）。確定した数値とサンプル画像はHTMLレポート `docs/reports/fan-demo/` に転記します。
- Attentionの値から「この色はこの画像由来」といった因果説明はしません。表示するのは参照画像・説明文・強度だけです。
- ブラインド比較の集計は少人数の記録であり、性能主張には使いません。論文の定量結果をこの展示の性能として扱いません。

## APIメモ

- `run.mode` は `live | sample` で、`run.blind.revealed` とは独立です。`variants[].mode` は `live | exact-cache | sample`。サンプルは `revealed: true` で始まりますが、`mode` はいずれも `sample` のままです。実生成の run は reveal の前後で `mode` が変わりません。
- reveal 前のスナップショットは `blind.mapping` / `blind.score` / `plain` / `variants[0].images` / `variants[0].personalization` を含みません。ブラインド画像は `/api/sessions/{sid}/images/blind/<token>.png` だけで配信し、同じ run の `…/plain-N.png` や `…/v0/v0-N.png` を直接叩くと reveal 前は404です（バイト比較での種明かしを防ぐため）。
- `variants[].mode` が `exact-cache` の場合、同じ体験中に同一条件で生成済みの画像を再利用しています。

## 生成モデルとモチーフ

[Illustrious XL v2.0-STABLE](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0)（revision `69459c1fe6f46db41ab31e6114f05acc0e06bcaa`）を使用します。1024×1280（縦長）・30 steps・DPM++ 2M SDE Karras（`DPMSolverMultistepScheduler` + `algorithm_type: sde-dpmsolver++` / `use_karras_sigmas`）・CFG 5.0・fp16。サンプラーは実画像比較でEuler ancestralより明確に高精細・高彩度だったため採用しました（2026-09-21計測、固定VAE・同一seed: 彩度 60.8→87.3、laplacian 243→462、1枚あたり +0.24 秒）。縦長 1024×1280 は upper body のお題の構図が安定し、同一条件で暖/寒の分離幅が正方形より広かったため採用しました（cat 30.0→32.8、rain 26.3→43.4、α=0.6 で被写体指定は維持、1枚あたり 3.7→4.7 秒）。単一safetensorsを `from_single_file` で読み込み、構成ファイルとtokenizerだけを初期Illustriousの固定revisionから読みます。`from_single_file` は `name_or_path` を残さないため、FANのSDXLエンコーダーは `FAN(text_encoder, tokenizer, L.pth)` と `FAN(text_encoder_2, tokenizer_2, bigG.pth)` を `fan.wrapper.stable_diffusion_xl` で束ねて手動で組み立てます。

デコーダーだけは固定revisionの [sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)（`207b116dae70ace3637169f1ddd2434b91b3a8cd`）に差し替えます。チェックポイント同梱のVAEをfp16で使うと全体が白っぽく低コントラストになるためです（2026-09-21計測、同一seed・同一構図でデコーダーのみ変更: 彩度 42.6→60.8、コントラスト 46.7→62.1、laplacian 84→179）。

お題は「窓辺で猫と過ごす少女」「東京の夜景と青年」「森を旅する魔法使い」「海辺の灯台と船乗り」「雨の街角の少女」「カフェで迎える店員」の6件。カードの被写体（街角の少女・図書館の青年・草原の旅人・カフェの店員）はお題と重ねていないため、「被写体が好き」と「表現が好き」を切り分けられます。通常側と個人化側のモデル・設定・seedは一致させます。

## 固定assetの準備

重みは `configs/demo.json` の固定revisionを準備時に取得します。起動時downloadは行いません。画像は `assets/manifest.json` に出典・hash・生成条件・参照文を保持します。Illustriousの取得例（リポジトリルートで一度だけ）:

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
uv run --project exhibit python exhibit/scripts/prepare.py cards
# 16枚を目視し、説明文と合うものだけ configs/cards-review.json で reviewed: true にする
uv run --project exhibit python exhibit/scripts/prepare.py generic
uv run --project exhibit python exhibit/scripts/prepare.py samples
uv run --project exhibit python exhibit/scripts/build_fallback.py
uv run --project exhibit python exhibit/scripts/preflight.py --models
```

未確認のカードは `/api/config` に出ず、preflightも不合格になります。`assets/fallback.html` は実画像を埋め込んだ単独HTMLで、サーバーが停止していても開けます。サンプルは代表的な選択の事前生成であり、来場者の選択を反映した結果としては表示しません。

固定展示画像とサンプルは配布用assetとしてGit対象。モデルとセッション一時生成物は `outputs/` 配下でGit対象外です。

## 検証

```bash
uv run --project exhibit pytest exhibit/tests -q
uv run --project exhibit ruff check exhibit && uv run --project exhibit ruff format --check exhibit
node --test exhibit/tests/test_browser_state.mjs
PYTHONPATH=fan-repro/.work/upstream:exhibit/src fan-repro/.venv/bin/python exhibit/scripts/fan_probe.py
uv run --project exhibit python exhibit/scripts/rehearsal.py --sessions 20
uv run --project exhibit python exhibit/scripts/browser_check.py --report-dir docs/reports/fan-demo
```

`fan_probe.py`（G0）は参照なしFANエンコードの一致・速度・VRAMを、`rehearsal.py` は連続セッションのwall/p95を、`browser_check.py` は実Chromiumで「3枚選択→お題→ブラインド4対→reveal→調整→再生成→終了」を確認します。ブラウザー試験のChromiumは `EXHIBIT_CHROMIUM` で指定できます。GPUを使うコマンド同士は同時実行しないでください。

第三者5人での理解度確認と2時間連続稼働は別の展示受入作業です。実施済みの内容・計測範囲・残項目はHTMLレポートに記載します。
