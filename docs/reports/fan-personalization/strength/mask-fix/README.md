# FAN attention mask 修正 · 検証

対象プロンプト: topic `cat`（`configs/demo.json`）

参照は `warm color palette, amber tones`（重み 3.0）、`harsh sunlight, hard cast shadow, high contrast`（重み 3.0）。CPU・fp32・実チェックポイントの text encoder 2 本のみで計測。

再現: `exhibit/scripts/fan_mask_check.py`（使い方は docstring）。

## `use_attn_mask=true` の意味

**参照側の pad トークンを attention の key から外す、それだけ。**
target プロンプトの encode は素の pipeline と 1 bit も変えません（pad 位置も含む）。

upstream には 2 つの問題があります。1 つ目は `fan.model.wrapper_forward` が
causal mask と padding mask のうち最後の 1 本しか個人化 attention に渡さないこと。
transformers 4.57 の CLIP は 2 本を別々に attention へ渡し `CLIPAttention` の中で
加算するので（FAN が差し替える層より 1 段下）、mask ありでは causal 構造が消えます。
2 つ目は `FAN.__call__` が target 自身の 2 行にも padding mask を配ること。
target の実トークンは causal のおかげで無傷ですが pad 27 個が動き、
UNet は 77 位置すべてを読むため参照ゼロでも絵が変わります（e10-mask 参照）。

修正は upstream の wrapper が見る前に padding mask の target 2 行を 0 にし、
causal mask と加算して 1 本の `attention_mask` に畳み込みます
（`exhibit/src/exhibit/fan_mask.py`）。参照なしの encode は
`fan_adapter._encode_once` と `evaluation_worker._raw_official` が
そもそも mask を渡しません。

transformers 4.57 が実際に渡す形（実測）: `causal_attention_mask` [4, 1, 77, 77]、`attention_mask` [4, 1, 77, 77]、`weight` [1, 79]、`n_token` 77 で、1 件あたり 4 行（target 2 + 参照 2）。

## a. 参照なしの encode（素の pipeline と同じであること）

CLIP-L は実トークン 50 個・pad 27 個。基準は mask なしの参照なし encode（e10-mask で pipeline と完全一致を確認済み）。

| encode | 最大絶対差 | cosine | 実トークン最大 L2 | pad 平均 L2 | pad 最大 L2 | pooled 最大絶対差 |
|---|---|---|---|---|---|---|
| 修正後（adapter 経由） | 0.0 | 0.99999988 | 0.0 | 0.0 | 0.0 | 0.0 |
| upstream の mask 意味 | 22.42300415 | 0.71854699 | 0.0 | 23.988726 | 51.558697 | 0.0 |

## b. alpha=0 と plain の一致（全 77 位置）

合否は `evaluation_worker` の PERSONAL_LIMITS（normalized_rmse ≤ 5e-3、mean_cosine ≥ 0.999）と同じ。出力は fp16。

| use_attn_mask | 版 | hidden 最大絶対差 | hidden rmse | hidden cosine | pooled 最大絶対差 |
|---|---|---|---|---|---|
| false | upstream | 0.00048828 | 6.51e-06 | 0.99999988 | 0.0 |
| false | 修正後 | 0.00048828 | 6.51e-06 | 0.99999988 | 0.0 |
| true | upstream | 22.42300415 | 0.64964867 | 0.59051532 | 0.0 |
| true | 修正後 | 0.00048828 | 6.51e-06 | 0.99999988 | 0.0 |

upstream の行は、素の pipeline と同じ plain を基準にしているので a の参照なしのずれをそのまま含みます。e10 の診断 `alpha_zero_vs_no_reference` は mask ありの plain を基準にしていたため 0.99999 で通っていました。

`use_attn_mask=false`・alpha=0.5 での修正前後の差: hidden 0.0 / pooled 0.0（完全一致）。

## c. 参照 pad へ流れる attention（alpha=0.5, skip_pa=[0]）

clip_l layer 1 (first personalized layer with skip_pa=[0])。head 平均、target 位置ごとの重み合計。

| target 位置 | use_attn_mask | target 側 | 参照の実トークン | 参照の pad |
|---|---|---|---|---|
| 5 | false | 0.5 | 0.5 | 0.0 |
| 20 | false | 0.5 | 0.37561 | 0.12439 |
| 45 | false | 0.5 | 0.434991 | 0.065009 |
| 5 | true | 0.5 | 0.5 | 0.0 |
| 20 | true | 0.5 | 0.5 | 0.0 |
| 45 | true | 0.5 | 0.5 | 0.0 |

参照の実トークン数: 8 / 11。causal mask があるため、target 位置が参照長より手前なら pad は見えません。

## d. plain との cosine（alpha=0.5, hidden 2048 次元, token 平均）

plain は mask の有無によらず同一（a のとおり）なので、この列は個人化の強さそのものです。

| skip_pa | use_attn_mask | cosine | 最大絶対差 |
|---|---|---|---|
| [0] | false | 0.766487 | 17.40197754 |
| [0] | true | 0.796556 | 18.8984375 |
| 0-7 | false | 0.957523 | 17.125 |
| 0-7 | true | 0.944982 | 14.24902344 |

## 留意

- alpha=0 に残る差は fp16 の丸め。plain は 1 本、個人化は 4 本まとめて encode するため fp32 の結果がわずかに異なります。mask の有無で同じ値です。
- 参照側の encode には padding mask が入るので、参照の hidden state は mask なしの版と変わります。ただし参照の pad は key から外れるため、個人化 attention が読むのは実トークンだけです。
- 強度の向きは skip_pa によって変わります（d の表）。skip_pa [0] は cosine 0.766487 → 0.796556 で弱まり、skip_pa 0-7 は 0.957523 → 0.944982 で強まります。採用するなら alpha は取り直しになります。

## 以前の版

最初の修正は causal mask を戻すだけで、target 自身の pad も mask していました。
そのときの値（この README の前版と e10-mask）:

- 参照なしの encode が pipeline と一致せず、`fan_no_reference_vs_pipeline` が cosine 0.718。実トークンと pooled は一致、pad 27 個だけが平均 L2 24 で動いていた。
- 参照ゼロの生成画像が平均絶対画素差 40.9/255（topic `cat`）で別物になった。
- plain との cosine は skip_pa [0] で 0.931、skip_pa 0-7 で 0.968。
  いまの値と違うのは、当時の plain 基準が mask ありだったため。

