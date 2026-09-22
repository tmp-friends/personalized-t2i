# FAN 個人化の効きを強くするための実験計画

作成: 2026-09-22　対象: `exhibit/`(Illustrious-XL + FAN)　状態: 2026-09-22 に P0〜P4・A2・A3・A4・A5・A6・B1〜B4・C1・C2 を実施（D 群と C3/C4 は未実施）。結果は `docs/reports/fan-personalization/strength/SUMMARY.md`（結論）と同 `README.md`（自動生成の表・画像・AI 判定）。人による評価は未実施。

## 0. 要約

- 現在の既定 policy `legacy_exhibit` は FAN を設計上の強度の数分の一でしか動かしていない。
  効きが弱い主因は alpha の値ではなく、`skip_pa=[0..7]`、`pooled_mode=plain`、
  `use_attn_mask=false`、`profiling=all` の組み合わせが信号を削っていること。
- 既存の screen / refine / heldout は「alpha 0.5 + skip_pa [0] + plain + ratio 0.1」を
  勝者として選んだ(heldout: Δhistory +0.0131、Δtarget −0.0076)。ただし policy 未登録、
  人による目視なし、alpha 0.5 超は未測定。
- 本計画は (A) 論文・公式実装に準拠したパターン、(B) 設定のみのスイープ、
  (C) アダプタ側の小改修、(D) 踏み込んだ案、の 4 群に分け、A→B→C→D の順で回す。

## 1. 前提知識(コードで確認済み)

### 1.1 alpha の意味

- alpha は埋め込みの補間係数ではない。CLIP text encoder の各 self-attention 層で
  「自プロンプトへの注意質量 ×(1−α)」と「参照への注意質量 ×α」を混ぜる係数
  (`fan-repro/.work/upstream/fan/model.py` `personalized_attention`)。
- 自プロンプト側の key/value はプレーンなコピーから取り、残差ストリームは無改変。
  そのため **alpha=1.0 でも崩壊しない**(attention 分岐が完全に参照駆動になるだけ)。
- alpha の [0,1] 制限は `exhibit/src/exhibit/fan_adapter.py:77` の validator のみ。
  upstream は範囲を強制しない。
- 参照の weight は Σw で正規化される。**UI の strength / aspect_gain を全部「強」にしても
  画像は変わらない**(相対配分しか変えられない)。

### 1.2 論文・公式実装の既定(準拠パターンの基準)

| 項目 | 公式既定 | 出典 |
|---|---|---|
| alpha | 0.4 | `inference.py:16`, README |
| skip_pa | 0(層 0 のみ PA を外す。CLIP-L 11/12 層、bigG 31/32 層で PA) | `wrapper.py` 既定引数 |
| use_attn_mask | False | `wrapper.py` 既定引数 |
| pooled | FAN 経路(`ClassTokenDecoder` で pool 位置を推定、bigG を `skip=-1` で再エンコード) | `wrapper.py` `stable_diffusion_xl` |
| profiling(sample_size) | 0(全参照)または 0.1(tailored profiling) | `inference.py:11,25` |
| weight | 参照ごとの float、既定 1.0 | `inference.py:15` |
| 参照の形式 | 自然文の記述("A retro-futuristic space exploration movie poster with bold, vibrant colors") | README Usage |
| プロンプト | 自然文("A photograph of an astronaut riding a horse") | `inference.py:14` |
| SDXL 設定 | skip −2、1024×1024、50 steps、pipeline 既定 scheduler、negative prompt なし | `inference.py:44-51,66-75` |
| モデル | 汎用 SDXL / SD3 / FLUX(anime 特化モデルは対象外) | README |

### 1.3 現行 exhibit がこの既定から外れている点

| 項目 | exhibit(`legacy_exhibit`) | 影響 |
|---|---|---|
| skip_pa | [0..7]。CLIP-L は 4/12 層、`skip=-2` で最終層の PA も捨てるため実効 3 層 | 大 |
| pooled_mode | plain。SDXL の pooled 条件付け(色・光・全体トーン)が完全に非個人化 | 大 |
| use_attn_mask | false。参照は 77 トークンに padding され、6〜10 トークンの語句に対し注意質量の 8〜9 割が padding に落ちる | 中 |
| profiling | all。最大 16 語句の重み付き平均(セントロイド)で個性が相殺 | 中 |
| 参照の形式 | Danbooru タグ風の短句(`catalog-v2.json` の axes) | 中 |
| プロンプト | タグ列 + quality tail、約 69 トークン | 中 |
| negative prompt | 長いタグ列をプレーンに encode。CFG は「personalized − negative」を増幅し、見せたい「personalized − plain」差分は増幅しない | 中 |
| pooled FAN 経路の不整合 | 長いタグ列で `ClassTokenDecoder` が padding を終端と誤検出(cos 0.78 vs 基準 ≥0.999) | pooled=fan が使えない原因 |

`fan-repro/README.md` の「α=0.4 で効く」は upstream 既定(skip_pa=0、FAN pooled)で測った値で、
配備中 policy の挙動ではない。

### 1.4 既存エビデンス

| 段階 | 条件 | Δhistory vs legacy | Δtarget vs legacy | 判定 |
|---|---|---|---|---|
| screen | skip_pa [0] / plain / all / α0.4 | +0.0068 | −0.0004 | 合格 |
| screen | skip_pa [0] / plain / ratio0.1 / α0.4 | +0.0058 | −0.0017 | 合格 |
| screen | pooled=fan の 4 条件 | — | — | numerical_failure |
| refine | skip_pa [0] / plain / all / α0.5 | +0.0145 | −0.0141 | 不合格(床 −0.01) |
| refine | skip_pa [0] / plain / ratio0.1 / α0.5 | +0.0170 | −0.0096 | 合格・選出 |
| heldout | 同上 | +0.0131 | −0.0076 | 合格 |

注意点: screen の alpha は 0.4、比較対象の legacy は 0.5 で変数が混ざっている。
勝者は policy として未登録で、目視評価なし。

### 1.5 制約(変更しない)

- 既定 policy は heldout と 20 名以上のブラインド評価を通すまで `legacy_exhibit` のまま。
  強化案は **別 policy_id として登録**して比較する。
- target 床 `mean Δtarget ≥ −0.01`、history 改善 `≥ +0.005`(`fan-evaluation.json` rules)。
- 比較中は生成設定(model / VAE / sampler / steps / CFG / 解像度)を固定。変える実験は
  別 experiment として記録する。
- upstream は固定 SHA、source patch なし。層別 alpha 等は adapter 側 monkeypatch で行う。
- `exhibit/configs/cards-v2-review.json` は agent が書かない。
- GPU は RTX 4090 一枚、lease は一度に一つ。

## 2. 共通プロトコル

### 2.1 事前修正(実験の前に必ず)

| ID | 内容 | 理由 |
|---|---|---|
| P0 | `fan-evaluation.json` の `screen.alpha` を 0.5 に揃える(または legacy を 0.4 で再測) | 変数の混在をなくす |
| P1 | refine/heldout 勝者を `fan-policies.json` に `strong_v1` 等の名前で登録 | 以後の比較基準にする |
| P2 | `fan_probe.py` に `skip_pa` / `pooled_mode` / `use_attn_mask` を pass-through | 現状は upstream 既定で測っており policy を反映しない |
| P3 | 勝者と legacy の heldout 画像を並べて目視(topic × history × seed) | CLIP delta と体感の対応を先に把握 |
| P4 | `fan_block` の provenance に alpha を含める | どの強度で作った画像か一目で分かるように |

### 2.2 固定条件

- 生成: `demo.json` の現行(Illustrious-XL v2.0、DPM++ SDE Karras、30 steps、CFG 5.0、1024×1280、fp16-fix VAE)。
- 評価: `fan-evaluation.json` の topics × histories × seeds(screen 24 件、heldout 156 件)。
- 指標: `target_score`(画像 vs 目標プロンプトの CLIP cos)、`history_score`(画像 vs 参照の重み付き CLIP cos)。
  合否は既存 rules。
- 追加指標(任意): plain との pixel MAE(`fan_probe.py`)、embedding cos(plain vs personalized)、
  却下レベル語句に対する CLIP cos(anti-preference 実験用)。
- 目視: 各条件で topic 2 × history 2 × seed 2 の 8 枚を plain と並べて保存。

### 2.3 実行コマンド

```bash
# encoder レベルの速い確認(数分)
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/fan_probe.py \
  --topic cat --alpha-sweep 0.4 0.6 0.8 1.0 --no-generate

# 画像評価(policy grid は fan-evaluation.json の screen / refine ブロックで宣言)
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py encoding --config exhibit/configs/fan-evaluation.json
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py screen   --config exhibit/configs/fan-evaluation.json --resume
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py refine   --config exhibit/configs/fan-evaluation.json --resume
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py heldout  --config exhibit/configs/fan-evaluation.json --resume
```

## 3. 実験群 A: 論文・公式実装に準拠したパターン

目的: 「FAN 本来の使い方」で効きがどれだけ出るかを先に確かめ、exhibit 固有の削減要因を切り分ける。

### A1. 公式既定を Illustrious でそのまま(`official_encoder` 相当)

- 設定: alpha 0.4、skip_pa [0]、use_attn_mask false、pooled fan、profiling all、weight 1。
- 現状: pooled=fan が α=0 整合検査(cos ≥0.999)で落ちる。
- 手順: `fan_probe.py class_token_report` で `ClassTokenDecoder` が選ぶ位置を確認し、
  (a) 短い自然文プロンプト、(b) タグ列プロンプト、で pool 位置が EOS に一致するかを比較。
- 仮説: タグ列 + 69 トークンで decoder が padding を指す。短い自然文なら一致し、pooled=fan が使える。
- 判定: (a) で cos ≥0.999 なら A3/A4 へ。(b) のみ失敗なら原因はプロンプト形式で、C2(EOS pooled)が根本対処。
- コスト: 設定のみ、encoder のみ。

### A2. 論文レジームの再現(制御実験、base SDXL 1.0)

- 設定: `stabilityai/stable-diffusion-xl-base-1.0`、自然文プロンプト、自然文参照、alpha 0.4、
  skip_pa [0]、pooled fan、50 steps、1024×1024、negative なし、pipeline 既定 scheduler。
  `fan-repro/scripts/generate.py` で実行できる。
- 目的: FAN が「自分の土俵」でどの程度見える効きを出すかの上限を知る。exhibit 側の
  効きの弱さが、モデル(anime 特化)・プロンプト形式・policy のどれ由来かを切り分ける。
- 比較: 同じ参照語句(catalog-v2 の axes を自然文化したもの)で alpha 0 / 0.4 / 0.7 の 3 点。
- 判定: 目視で明確な差があれば「FAN 自体は効く」。差が小さければ参照語句の設計(A3)が先。
- コスト: 別 experiment として記録(生成設定が違うため exhibit の rules は適用しない)。

### A3. 参照を自然文にする(`reference_unit` の拡張)

- 現状: `aspect_phrase` は "warm color palette, amber tones" のようなタグ句。
  `card_description` はそれを ", " で連結。どちらも公式 README の例(完全な文)とは形式が違う。
- 案: 各軸レベルに自然文の説明を持たせる(例: "an illustration with a warm amber color palette
  and golden tones")。新 `reference_unit=aspect_sentence` を `domain.py` に追加。
- 仮説: CLIP は自然文で事前学習されており、padding を除く実トークンが増えて注意質量が語句側に乗る。
- 手順: screen grid に `reference_unit` を次元として追加(現状は grid にない)。
- 判定: 既存 rules。目視で「色・光」が動いたかを確認。
- コスト: 文言設計(16 レベル分)+ domain 側 20 行程度。

### A4. tailored profiling を公式どおりに使う(sample_size 0.1 vs 0)

- 現状の勝者は ratio 0.1(12〜16 参照 → 1 本)。公式 CLI も `--sample_size` を渡すと 0.1 に固定する。
- 案: `{"mode":"count","value":k}` で k = 1 / 2 / 4 / 全部 を比較。
- 仮説: 少数参照はセントロイド平均を避けて鋭くなるが、履歴の多様性を落とす。k=2〜4 に最適点がある。
- 手順: screen の `profiling` に count 2 / 4 を追加。
- 判定: 既存 rules + mixed / sparse history での安定性(seed 間の分散)。
- コスト: 設定のみ。

### A5. weight を公式のように「参照ごとの意図」で使う

- 現状: weight は strength × aspect_gain の合算で、Σw 正規化により全体を上げても無効。
- 案: 公式の "Per-reference preference intensity" として、選択回数の多いレベルに集中させる
  (例: 最頻レベルに 2.0、それ以外 0.5)。alpha とは独立に「どこに効くか」を鋭くする。
- 判定: history_score の内訳(軸別)で最頻軸が伸びるか。
- コスト: `domain.py` の重み集計ロジックのみ。

### A6. 公式の生成設定に寄せた対照(sampler / negative)

- 案: negative prompt なし、`dpmsolver++`(非 SDE)、50 steps で legacy と strong_v1 を比較。
- 目的: SDE の毎 step ノイズと長い negative が差分を洗い流している度合いを測る。
- 注意: 生成設定が変わるため別 experiment。exhibit 本番設定を変える提案ではなく、効きの可視性の診断。

## 4. 実験群 B: 設定のみのスイープ(コード変更なし)

| ID | 変更 | 仮説 | リスク | 判定 |
|---|---|---|---|---|
| B1 | `skip_pa` を [0] / [0..3] / [0..7] で比較(alpha 0.5 固定) | PA 層数が最大の単独要因。[0] が最強、[0..3] が中庸 | 前半層は語彙的なので主題へのブリード | screen rules + 目視 |
| B2 | alpha を 0.5 / 0.6 / 0.7 / 0.85 / 1.0(skip_pa [0]、ratio 0.1) | 0.8 付近まで target 床内。1.0 でも崩壊しない | 主題の細部消失が先に来る | `fan_probe --alpha-sweep` → refine `added_alpha` |
| B3 | alpha 同上 × profiling all | all は α0.5 で床割れ。history 最大だが target 犠牲 | 床割れ | rules の床の妥当性検討も兼ねる |
| B4 | `use_attn_mask=true` | padding 希釈を止めて同 alpha で鋭くなる | upstream の癖で PA 経路の causal mask が padding mask に上書きされ、target ブロックが双方向になる。プロンプト意味が変わり得る | まず `fan_probe --no-generate` で plain との cos を確認、次に screen 次元に追加 |
| B5 | B1 × B2 × B4 の小さな full factorial(2×3×2=12 条件 × 24 件) | 相互作用の把握 | GPU 時間 | rules |

## 5. 実験群 C: アダプタ側の小改修(upstream 無改変)

### C1. 埋め込み空間のゲイン `hidden = plain + w·(pers − plain)`

- 変更: `fan_adapter.py` の encode 末尾。policy に `embed_gain` を追加(validator も)。
- 根拠: pooled=plain の経路では既に plain と personalized の両方を計算しているため追加コストゼロ。
- スイープ: w ∈ {1.0, 1.5, 2.0, 2.5}(alpha 0.5、skip_pa [0])。
- リスク: 高 w で encoder の manifold から外れる(バンディング、ハロー、質感ノイズ)。
- 判定: rules。w=2.0 で target 床内なら採用候補。

### C2. pooled を EOS 位置で個人化する新モード `pooled_mode=fan_eos`

- 変更: bigG を `pooling=False, skip=-1, normalize=False` で個人化エンコードし、
  `pool_text_hidden_state`(EOS 位置)→ `normalize_text_hidden_state` → `projection_text_hidden_state`。
  `POOLED_MODES` に追加。
- 根拠: `ClassTokenDecoder` の padding 誤検出を回避しつつ、SDXL の pooled チャネル
  (色・光・全体トーン)を復活させる。
- 検査: α=0 で `pipe.encode_prompt` の pooled と cos ≥0.999(既存の整合検査を流用)。
- 判定: screen の `pooled_mode` 次元に追加。目視で全体トーンの変化を確認。
- リスク: token 埋め込みと pooled が食い違うと全体トーンだけ動く。

### C3. anti-preference から negative prompt を作る

- 変更: `workers.py` の negative encode 時に、各軸の非選択レベルの語句を negative に追加
  (または参照として encode)。`fan_probe.py` の `OPPOSITE_REFS` が前例。
- 根拠: CFG に「却下スタイルから離れる」方向を与える。UNet 追加パスなし。
- リスク: 相関する軸(`forbidden_level_pairs`)で過抑制。negative が 77 トークン上限に近い。
- 判定: history_score 上昇 + 却下語句に対する CLIP cos の低下 + target 床。
- 注意: 負の weight は `fan_adapter.py:163` と `domain.py:237` で遮断されており、
  upstream の Σw 正規化とも相性が悪い。負 weight ではなくこの方式で行う。

### C4. UI の strength を alpha(または embed_gain)に接続

- 変更: `app.py` の aspect_gains / strength から alpha を導出(例: 平均 gain で 0.4〜0.8 に写像)。
- 根拠: 現状はダイヤルが切れており、来場者の「弱い」体感の一部はここ。
- 副作用: personalization identity hash が変わりキャッシュ無効化。
- 判定: 画像指標ではなく、全 0.5 と全 2.0 の gain で hidden が一致する(現状)/しない(修正後)を assert。

## 6. 実験群 D: 踏み込んだ案(B・C の結果を見てから)

| ID | 内容 | 期待 | リスク・コスト |
|---|---|---|---|
| D1 | 層別 alpha(後段ほど大、例 0.3→0.9 を層 4〜10 で ramp)。wrapper は層 index ごとに構築されるので adapter 側 monkeypatch で可能 | 前半層の忠実度を保ちつつ後段でスタイルを載せる。品質あたり強度は最良の見込み | upstream 無改変方針との整合(monkeypatch の前例は `fan_adapter.py` の `sample_reference` 差し替え) |
| D2 | denoising step のスケジュール(`callback_on_step_end` で `prompt_embeds` / `add_text_embeds` を差し替え、前半強・後半弱またはその逆) | 一様には耐えられない強度を、対象要素が決まる step に限定して使える | 切替 step の継ぎ目。SDE sampler と干渉 |
| D3 | noise 空間の三方向ガイダンス `eps_neg + g·(eps_plain − eps_neg) + w·(eps_pers − eps_plain)` | 個人化差分だけを増幅する理論的に正しい方法 | UNet が毎 step 3 回で生成時間 1.5 倍、120 秒 timeout と衝突。独自ループが必要 |
| D4 | alpha > 1(1.25 / 1.5)を opt-in policy で | 外挿。数値的には安全 | 主題・構図が先に崩れる。validator 緩和は専用 policy に限定 |

非推奨: guidance_scale の引き上げ(プロンプトもアーティファクトも等しく増幅、Illustrious は 7 前後で飽和)。

## 7. 実行順序

1. P0〜P4(事前修正と目視)。
2. A1(pooled=fan が使えるかの切り分け)と B2 の encoder 部分(`fan_probe --no-generate`)。半日。
3. A2(base SDXL での上限確認)を 1 回だけ回し、目視で「FAN が効く絵」の基準を持つ。
4. B1 / B2 / B4 を screen → refine で。ここで policy `strong_v1`〜`strong_v3` が決まる見込み。
5. A3 / A4 / A5(参照設計)を B の勝者の上で。
6. C1 / C2 を実装し、B の勝者に重ねて refine。C3 / C4 は並行可。
7. D は B+C で target 床が拘束になってから。
8. 最終候補 2 つで heldout → ブラインド評価(20 名)。既定 policy の切替はそこまで行わない。

## 8. 記録

- 各実験は `exhibit/outputs/fan-evaluation/<hash>/` に既存の manifest / decision で残す。
  生成設定を変えた実験(A2, A6)は `experiment_kind` を分けて rules を適用しない。
- 目視画像は `docs/reports/fan-personalization/strength/<experiment>/` に plain と並べて保存。
- 判定が rules に反した場合(例: history は大きく伸びたが target が −0.012)は床の見直しを
  提案として記録し、黙って床を動かさない。
