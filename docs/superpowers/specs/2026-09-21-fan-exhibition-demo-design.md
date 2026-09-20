# FAN 展示デモ設計：同じ一文から、あなたの一枚を

作成日: 2026-09-21 / 展示日: 2026-09-23 / 状態: 実装中（feat/fan-exhibit）

[ZIPP-style persona × PIGReward 案](2026-09-19-zipp-pigreward-exhibition-demo-design.md) を置き換える。
主役を **FAN**（Foundation Encoders Are All You Need for Preference-Aware Personalization, CVPR 2026）にし、
LLM によるプロンプト書き換えと VLM 解析、PIGReward 推薦を展示の主経路から外す。ZIPP との比較は行わない。

## 1. 展示で見せること

> **入力文も生成モデルも変えていません。変えたのは、あなたが選んだ好みと、その反映の強さだけです。**

- 来場者が「好きな画像」を数枚選ぶ。選んだ画像に対応する **確認済みの表現説明文** を FAN の参照プロンプトにする。
- 同じお題・同じ seed で、通常生成（参照なし）と FAN 個人化生成を並べる。**方式を伏せたまま** 来場者に好きな方を選んでもらい、その後で答えを見せる。
- 反映強度 `alpha`（弱・中・強）と、参照ごとの `weight`（重視・通常・外す）を来場者が操作し、「比較する」を押して描き直す。
- 来場者ごとのモデル学習はしない。ただし FAN 公式実装の `ClassTokenDecoder`（`weight/L.pth`, `weight/bigG.pth`）を使うため、「追加部品が一切ない」とは説明しない。
- Attention の値から「この画像がこの色を生んだ」といった因果説明はしない。参照に使った画像と説明文、強度だけを表示する。

## 2. 技術構成

| 層 | 採用 | 備考 |
|---|---|---|
| 生成器 | Illustrious XL v2.0-STABLE（SDXL 系、単一 safetensors、固定 revision） | 現行 `configs/demo.json` の `generation` を維持（1024², 28 steps, Euler a, CFG 5.0, fp16） |
| 個人化 | FAN 公式実装 `fan-repro/.work/upstream/fan`（固定 SHA）。`FAN(text_encoder, tokenizer, decoder=L.pth)` と `FAN(text_encoder_2, tokenizer_2, decoder=bigG.pth)` を `fan.wrapper.stable_diffusion_xl` で束ね、`prompt_embeds` / `pooled_prompt_embeds` を pipeline へ渡す | `personalized_t2i_encoder` の `name_or_path` 文字列判定には依存しない |
| GPU 環境 | `fan-repro/.venv/bin/python`（transformers 4.57 系, diffusers 0.39 系, torch 2.14） | FAN の attention monkey-patch は transformers 5 系と非互換のため、既存 `tailored-visions-repro/.venv` は使わない |
| 履歴選択 | `sample_size=0`（参照 3〜5 件は全部使う） | 参照が増えた将来に `sample_reference` を有効化する |
| Web | FastAPI + vanilla JS（既存の単一セッション・GPU lease・poller 構造を継承） | 推論はすべて子プロセス。Web プロセスはモデルを import しない |

負のプロンプトは参照なしで plain にエンコードし、`negative_prompt_embeds` / `negative_pooled_prompt_embeds` として渡す。
通常生成と個人化生成で **同じ target prompt・同じ negative・同じ seed・同じ設定** を使い、差は `ref`/`weight`/`alpha` のみとする。
G0 スパイク（`exhibit/scripts/fan_probe.py`, 結果は `exhibit/outputs/fan-probe/report.json`）で、
参照なし FAN エンコードが pipeline の `encode_prompt` と一致すること、および速度・VRAM を確認したうえで採用する。

### 2.1 G0 で確定した設定

G0 と追試（`exhibit/scripts/fan_probe.py`、`exhibit/outputs/fan-probe/followup/followup.json`）で次を確定した。数値は HTML レポートに転記する。

- `skip_pa = [0,1,2,3,4,5,6,7]`（personalized attention を行わない encoder 層の**リスト**。CLIP-L と bigG の両方に適用）。前半 8 層を外すと参照由来の一般的な「もや」が消え、α=0.6 まで被写体・構図・衣装の指定が保たれる。
- `use_attn_mask = false`（明示）。true にすると α=0（参照の影響が無いはず）でも target のエンコードが変わる（plain との cos 類似 0.59）。
- `alpha` は 弱 0.35 / 中 0.5 / 強 0.6。
- 参照は 1 枚 1 文の長い束ね方をやめ、**側面ごとの短い句**を重複排除して渡す（長い束ね方は構図を大きく振った）。同じ句は 1 件にまとめ weight を合算する。
- pooled は個人化しない。上流の `ClassTokenDecoder` が padding token を終端と誤検出するため、同じ文の参照なし pooled を使う（hidden states は個人化、pooled は plain）。image event に `pooled: "plain"` として記録する。

これらは `configs/demo.json` の `fan` / `alphas` に固定し、`Personalization.hash` に含める。

## 3. 来場者体験（約 3 分、1 端末 1 セッション）

```text
01 好きな画像を選ぶ  →  02 お題を選ぶ  →  03 どちらが好き？（方式を伏せて比較）
                                          →  04 答えを見て、強さと参照を調整して描き直す  →  終了
```

### 3.1 好きな画像を選ぶ（3〜5 枚）

- **4 被写体 × 4 表現プロファイル = 16 枚** のカードを 4×4 で表示する（並び順はセッションごとにシャッフル）。
- 被写体は 6 お題と重ならないものを使う（街角の少女・図書館の青年・草原の旅人・カフェの店員）。同じ被写体が 4 種の表現で並ぶため、「被写体が好き」と「表現が好き」を切り分けられる。
- 各カードは `subject`（被写体プロンプト、参照には使わない）と `profile`（表現）を持つ。**参照プロンプト `ref_en` は表現部分だけ**（例: `cool color palette, blue and teal tones, soft diffused light, cel shading, clean lineart, calm atmosphere`）。被写体語を参照に含めない。
- 表現プロファイルは `color / lighting / texture / mood` の 4 側面の句で構成する。カード選択後、任意で「この画像のどこが好き？」から側面を外せる（既定は全部 ON）。外した側面の句は `ref_en` から除く。
- 3 枚未満では進めない。5 枚を超えては選べない。
- 表現プロファイル（初期案・実画像で調整）:

| id | 表示名 | color | lighting | texture | mood |
|---|---|---|---|---|---|
| warm_soft | あたたかく、やわらかく | warm color palette, amber and orange tones | warm golden hour light, gentle shadows | watercolor painting, soft brushwork, painterly texture | calm atmosphere |
| cool_clean | 涼しく、くっきり | cool color palette, blue and teal tones | soft diffused light | cel shading, clean lineart, anime coloring | calm atmosphere |
| dramatic | 劇的な光と影 | muted colors, dark background | dramatic lighting, strong shadows, rim light | detailed shading, painterly texture | serious atmosphere |
| vivid_lively | 鮮やかで、にぎやか | vivid colors, colorful, high saturation | bright lighting, sparkle | cel shading, flat color | cheerful expression, lively atmosphere |

- カード画像と説明文の対応は、生成後にスタッフが目視で確認し `reviewed: true` を付ける。未確認カードは preflight で不合格。

### 3.2 お題を選ぶ

6 お題（現行と同じ）。通常生成 4 枚（seed 固定）は事前生成キャッシュ。target prompt は `basic_prompt_en + positive_prompt_tail` の固定文で、LLM による補足はしない。

### 3.3 どちらが好き？（ブラインド比較）

- 「描く」を押すと、FAN 個人化 4 枚（`alpha=0.5`、全参照 `weight=1.0`）を 1 枚ずつ生成する。
- 画面は **seed ごとの 4 対**。各対は通常 / 個人化を左右ランダムに並べ、ラベルを付けない。画像は 1 対ずつ完成した順に表示し、来場者は各対で好きな方（または「決められない」）を選ぶ。
- サーバーは対ごとの並びを不透明 ID で返し、`revealed` になるまで対応表と個人化画像の実 URL を返さない。通常画像も `/assets/generic/...` ではなくセッション経由の URL で配信し、URL から方式が分からないようにする。
- 4 対すべてに答えるか「答えを見る」を押すと reveal。「あなたが選んだ 4 枚のうち n 枚が、好みを反映した画像でした」と、使った参照（画像・説明文）・`alpha`・target prompt が変わっていないことを表示する。
- 全参照が外されて（3.4 で）参照が 0 件になる操作は許さない。最低 1 件。

### 3.4 強さと参照を調整する

- reveal 後、`alpha` を弱 0.35 / 中 0.5 / 強 0.6 から選び、参照ごとに 通常 1.0 / 重視 2.0 / 外す を選び、「この設定で描く」を押すと別 variant として 4 枚を生成する。連続生成はせず、押したときだけ生成する。
- 結果画面は通常 4 枚の行と、各 variant の行（設定ラベル付き、最新を強調）。variant は 1 run につき最大 3。同じ設定は「同一条件のキャッシュ」として再利用する。
- 「強くすればよいわけではない」ことを伝える一文を添える。判断は来場者に委ねる。
- P2（余力があれば）: 「別の人の好みで描く」プリセット（固定の反対方向参照セット）。FAN の画像が一般的にきれいなのか、本人向けだから好まれるのかを切り分ける実験用。

### 3.5 終了・リセット

終了ボタン、または 90 秒無操作で一時データを削除する。処理中は無操作リセットを止め、120 秒の処理期限を適用する。中止はワーカーを終了させる。

## 4. データ契約

```jsonc
// configs/demo.json（抜粋）
{
  "version": "exhibit-fan-illustrious-v2",
  "generation": { /* 現行と同一 */ },
  "seeds": [230923, 230924, 230925, 230926],
  "alphas": {"weak": 0.35, "mid": 0.5, "strong": 0.6},
  "weights": {"exclude": 0, "normal": 1.0, "emphasis": 2.0},
  "selection": {"min": 3, "max": 5},
  "fan": {
    "python": "fan-repro/.venv/bin/python",
    "upstream": "fan-repro/.work/upstream",
    "commit": "9d0b76843f6437718195accac9cf3f050a25d26b",
    "decoders": {"L.pth": "<sha256>", "bigG.pth": "<sha256>"},
    "skip": -2, "sample_size": 0,
    "skip_pa": [0,1,2,3,4,5,6,7], "use_attn_mask": false
  },
  "topics": [ {"id","label","basic_prompt_en"} ],
  "card_subjects": [ {"id","label","basic_prompt_en","seed"} ],
  "card_profiles": [ {"id","label","aspects": {"color","lighting","texture","mood"}} ]
}
```

- `Card`: `id = "<subject>-<profile>"`, `subject_id`, `profile_id`, `label`, `aspects`, `ref_en`（= aspects を `, ` で結合）, `prompt`（= subject basic prompt + aspects + tail、生成に使った実プロンプト）, `path`, `sha256`, `seed`, `reviewed`。カード画像は `assets/cards/<id>.png`、manifest は `assets/manifest.json`。
- `Selection`: `[{card_id, aspects_off: []}]`（3〜5 件、順序保持）。
- `Personalization`: `{refs: [{text, weight, aspect, card_ids: []}], alpha, sample_size: 0, hash}`。`refs` は選択順・側面順に並べた**側面ごとの句**で、同じ句は 1 件にまとめ weight を合算し `card_ids` を連ねる（`exclude` のカードは何も寄与しない）。`hash` は `refs`（順序込み）・`alpha`・`fan.commit`・`fan` の符号化設定（`skip` / `sample_size` / `skip_pa` / `use_attn_mask`）・`generation`・`seeds` の digest。生成画像 event は `personalization_hash` を持ち、一致しない結果は表示しない。
- `Run`: `{id, topic_id, status, plain: [image×4], variants: [{id, request_id, alpha_key, weights: {card_id: key}, personalization, images: [...], status, mode: live|exact-cache|sample, timings, metrics, error?}], blind: {pairs: [{index, seed, items: [{token}], pick: token|null|"tie"}], revealed: bool, personal_tokens_hidden_until_reveal}, message, elapsed_seconds}`。
- 画像 event: `{type:"image", id, path, sha256, seed, prompt, negative_prompt, settings, personalization_hash|null, seconds}`。

### 4.1 スナップショットの形（フロントエンドとバックエンドの共通契約）

`GET /api/config`:

```jsonc
{
  "cards": [{"id": "girl-warm_soft", "subject_id": "girl", "profile_id": "warm_soft",
             "label": "街角の少女 · あたたかく、やわらかく", "subject_label": "街角の少女", "profile_label": "あたたかく、やわらかく",
             "aspects": {"color": "warm color palette, amber and orange tones", "lighting": "...", "texture": "...", "mood": "..."},
             "aspects_ja": {"color": "暖かなオレンジ系の色", "lighting": "...", "texture": "...", "mood": "..."},
             "url": "/assets/cards/girl-warm_soft.png"}],
  "aspects": {"color": "色", "lighting": "光", "texture": "描画", "mood": "雰囲気"},
  "topics": [{"id": "cat", "label": "窓辺で猫と過ごす少女", "preview_url": "/assets/generic/cat-0.png"}],
  "alphas": {"weak": 0.35, "mid": 0.5, "strong": 0.6},
  "alpha_labels": {"weak": "弱", "mid": "中", "strong": "強"},
  "weights": {"exclude": 0, "normal": 1.0, "emphasis": 2.0},
  "weight_labels": {"exclude": "外す", "normal": "通常", "emphasis": "重視"},
  "selection": {"min": 3, "max": 5},
  "max_variants": 3,
  "idle_seconds": 90, "timeout_seconds": 120,
  "samples": [{"id": "s1-cat", "topic_id": "cat", "label": "..."}],
  "ready": true
}
```

`GET /api/sessions/{sid}`（reveal 前）:

```jsonc
{
  "id": "…",
  "card_order": ["girl-dramatic", "..."],            // 16 件のシャッフル順
  "selection": [{"card_id": "girl-warm_soft", "aspects_off": ["mood"]}],
  "run": {
    "id": "…", "topic_id": "cat", "status": "generating",   // queued | generating | done
    "message": "2 / 4枚ができました。", "elapsed_seconds": 0, "error": null,
    "blind": {
      "revealed": false,
      "pairs": [
        {"index": 0, "seed": 230923, "ready": true,
         "items": [{"token": "k7f…", "url": "/api/sessions/{sid}/images/blind/k7f….png"},
                   {"token": "q2a…", "url": "/api/sessions/{sid}/images/blind/q2a….png"}],
         "pick": null},                                        // token | "tie" | null
        {"index": 1, "seed": 230924, "ready": false, "items": [], "pick": null}
      ],
      "answered": 0, "mapping": null, "score": null
    },
    "plain": [],                                             // reveal 後に 4 件
    "variants": [{"id": "v0", "request_id": "…", "alpha_key": "mid",
                  "weights": {"girl-warm_soft": "normal", "...": "normal"},
                  "status": "generating", "mode": "live", "done_count": 2,
                  "images": [], "personalization": null, "timings": {}, "error": null}]
  }
}
```

reveal 後（および 2 つ目以降の variant）:

```jsonc
"blind": {"revealed": true, "pairs": [...同じ...], "answered": 4,
          "mapping": {"k7f…": "plain", "q2a…": "personal"},
          "score": {"personal": 3, "plain": 1, "tie": 0, "answered": 4}},
"plain": [{"id": "plain-0", "seed": 230923, "url": "/api/sessions/{sid}/images/plain-0.png", "prompt": "…"}],
"variants": [{"id": "v0", "alpha_key": "mid", "weights": {...}, "status": "done", "mode": "live", "done_count": 4,
              "images": [{"id": "v0-0", "seed": 230923, "url": "/api/sessions/{sid}/images/v0/v0-0.png", "sha256": "…"}],
              "personalization": {"alpha": 0.5, "sample_size": 0, "hash": "…",
                                  "refs": [{"text": "warm color palette, amber and orange tones", "weight": 1.0, "aspect": "color", "card_ids": ["girl-warm_soft"]}]},
              "prompt": "masterpiece, … absurdres, highres.", "timings": {"generation": {"wall_seconds": 25.1}}, "error": null}]
```

- reveal 前の snapshot に `mapping`・`plain`・`variants[0].images` を含めない。blind 画像はセッション経由の不透明 URL で配信し、通常画像も同じ形式にする。
- `mode` は `live | exact-cache | sample`。sample は `blind.revealed = true` で始まる。
- 2 つ目以降の variant はブラインドにしない（`blind` は variant 0 のみ）。

## 5. API（`/api`）

| Method | Path | Body | 備考 |
|---|---|---|---|
| GET | `/config` | – | cards（reviewed のみ）, topics, alphas, weights, selection, idle/timeout, samples, ready |
| GET | `/health` | – | `{status, gpu_busy, mode: "fan-live", offline: true}` |
| POST | `/sessions` | – | 単一セッション。既存があれば 409 |
| GET | `/sessions/{sid}` | – | snapshot（reveal 前は方式・実 URL を隠す） |
| POST | `/sessions/{sid}/touch` | – | |
| PUT | `/sessions/{sid}/selection` | `{cards: [{card_id, aspects_off}]}` | run 中は 409。3〜5 件 |
| POST | `/sessions/{sid}/runs` | `{topic_id, alpha: "mid", weights: {card_id: "normal"|"emphasis"|"exclude"}, request_id}` | 初回は run を作る。2 回目以降は同じ topic に variant 追加（run が done のときのみ、最大 3）。request_id で冪等 |
| POST | `/sessions/{sid}/blind` | `{pair_index, pick: token|"tie"}` | variant 0 の対。reveal 後は 409 |
| POST | `/sessions/{sid}/reveal` | – | |
| POST | `/sessions/{sid}/cancel` | – | 実行中 variant を中止（run の完成済み部分は保持） |
| POST | `/sessions/{sid}/sample` | `{sample_id}` | 事前生成サンプル（mode: sample、reveal 済み扱い） |
| DELETE | `/sessions/{sid}` | – | |
| GET | `/sessions/{sid}/images/{name}` | – | セッション画像（個人化・通常とも同じ形式の URL） |

ローカル Origin 制約・`no-store`・`X-Content-Type-Options` は現行を継承する。

## 6. ワーカー（`exhibit/workers.py`、`fan-repro/.venv` で実行）

- 1 プロセス 1 リクエスト。`stage: "generate"`, `items: [{id, prompt, negative_prompt, seed, path, personalization|null}]`。
- 起動時に pipeline を `from_single_file` で読み、FAN encoder（L, bigG）を 1 回だけ構築。`loaded` event に versions と load 秒を含める。
- 各 item: `personalization` が null なら参照なし（plain）でエンコード、あれば `encoder(prompt, refs, weight=weights, alpha=alpha, skip=-2, sample_size=0, skip_pa=[0..7], use_attn_mask=False)`。negative は常に plain。すべて fp16 で pipeline に渡す。
- 画像を保存し `image` event を emit。最後に `metrics`（wall 秒、peak VRAM/RSS）。
- `PYTHONPATH` に `fan-repro/.work/upstream` と `exhibit/src` を含め、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。
- スパイクの結果、plain エンコードが `encode_prompt` と一致しない場合は、通常画像を「文字列プロンプト経路」ではなく「参照なし FAN 経路」で生成し、両者を同じ経路にそろえる。

## 7. 事前準備（`scripts/prepare.py`）

| step | 生成物 |
|---|---|
| `cards` | 16 枚 `assets/cards/*.png`（各 subject の seed 固定、profile ごとに同 seed）。manifest に prompt・ref_en・aspects・hash。`reviewed` は初期 false、目視後にスタッフが `configs/cards-review.json` で true にする |
| `generic` | 6 お題 × 4 seeds `assets/generic/*.png`（plain） |
| `samples` | 代表 3 選択 × 2 お題、`alpha=0.4` の 4 枚 `assets/samples/*.png` と `assets/samples.json`（選択、personalization、画像） |
| `fallback` | `assets/fallback.html`（サーバー不要の単独 HTML） |

`preflight.py`: manifest の generation 一致、全画像 hash、カード reviewed、samples 整合、`fan-repro/.venv` と `.work/upstream` の存在、decoder 重みの sha256、Illustrious の固定 checkpoint と pipeline config、モデル取得なし。LLM のチェックは削除。

## 8. 検証

- 単体: personalization hash が refs 順・weight・alpha・設定で変わる。aspects_off で ref_en から句が消える。除外で参照 0 件は拒否。blind の対応表が reveal 前の snapshot に含まれない。stale job/session/personalization の結果を表示しない。variant 上限、exact cache、cancel、idle。
- ワーカー: 疑似ワーカーでプロセス死・timeout・部分完成の保持（既存 `test_worker_recovery.py` 相当）。
- GPU 実測: `scripts/fan_probe.py`（G0）、`scripts/rehearsal.py`（連続 session の wall/p95）、`scripts/browser_check.py`（実 Chromium で 3 枚選択→お題→ブラインド 4 対→reveal→調整→再生成→終了、外部通信 0・JS 例外 0）。
- レポート: `docs/reports/fan-demo/index.html` に実測（生成秒、VRAM、equivalence diff、スクリーンショット、限界）。旧 `docs/reports/zipp-demo/` は履歴として残す。

## 9. 説明で守ること

- 「来場者ごとの追加学習なし」と言う。「追加モデル・重みが一切ない」とは言わない。
- 参照は「選んだ画像に付けた確認済みの説明文」。画像そのものをエンコーダへ入れているとは言わない。
- 反映を強くするほど良いとは言わない。target とのバランスは来場者が判断する。
- 論文の定量結果をこの展示の性能として使わない。ブラインド比較の集計は少人数の記録であり、性能主張にしない。
