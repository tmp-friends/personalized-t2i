# e12-hidden-norm · plain の pad を差し戻すと何が起きるか（2026-09-23）

参照だけを mask すると target の pad 27 個が plain の 0.58 倍に縮み、UNet はそこも読む。
`hidden_norm` で plain の pad を差し戻し、白茶け・線の消失が消えて steering が残るかを見た。

- 6 policy + legacy 暗黙、history warm / cool / t2-flat-graphic / warm-sun、topic 3、seed 2、**174 枚**
- 生成設定は現行のまま（既定 policy との比較基準を e10 と揃えるため。e11 の非 SDE は未適用）
- `exhibit/outputs/fan-evaluation/strength/24b50523…`

**結論: `plain_pad` はアーティファクトを消し、しかも初めて両側 steering を有意にした。**

## 主要な数値（Δ は plain 比、n=24）

| policy | Δlum | Δsat | laplacian_ratio | Δblown | Δdarkline | Δtarget | warm−cool | 95% CI | 両側 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy_exhibit` | −7.62 | −1.40 | 0.741 | +0.73 | −1.10 | +0.0001 | +4.38 | [−4.32, +12.94] | 両側 |
| `mask-skip1-alpha0.5`（対照） | +13.05 | −18.65 | 0.391 | −1.75 | −2.90 | +0.0040 | +6.16 | [−2.06, +13.78] | − |
| **`…alpha0.5-plainpad`** | **+4.27** | **−11.20** | **0.846** | **+0.93** | **−0.82** | **+0.0027** | **+8.07** | **[+1.04, +14.46]** | **両側** |
| **`…alpha0.5-plainpadtoken`** | **+4.21** | **−16.90** | **0.866** | **+1.22** | **−1.05** | **+0.0059** | **+12.54** | **[+7.01, +17.67]** | **両側** |
| `…alpha0.7-plainpad` | +13.77 | −15.17 | 0.681 | +0.75 | −1.85 | −0.0130 | +6.91 | [−6.64, +19.84] | − |
| `…alpha0.7-plainpadtoken` | −23.30 | −15.50 | 0.491 | −0.27 | −4.16 | −0.0476 | −1.40 | [−18.11, +15.39] | − |
| `…alpha0.7-plaintoken` | +60.83 | −57.44 | 0.030 | −7.41 | −6.87 | −0.1516 | +3.80 | [−15.68, +23.08] | 両側 |

## アーティファクトは消えたか — 消えた

`crop-focus-cat.jpg` / `crop-focus-cafe.jpg`（原寸 512 px 切り出し、
列は plain / legacy / alpha0.5 / 0.5+plain_pad / 0.5+plain_pad_token）。

- `alpha0.5` は白茶けて、髪の束の暗い境目が消え、目の輪郭も弱る。報告どおりの症状。
- **`0.5+plain_pad` と `0.5+plain_pad_token` は plain に近い。** 髪の暗い筋が戻り、
  目の輪郭が戻り、コントラストが戻る。
- 数値も一致する。ラプラシアン比 0.391 → 0.846 / 0.866、
  線の消失 −2.90 → −0.82 / −1.05、白飛びは負から正に戻る（−1.75 → +0.93 / +1.22）。
  **lum の持ち上がりも 13.05 → 4.3 と 1/3 になる。**

## steering — 初めて有意かつ両側

これまでのどの policy も、warm−cool が有意になるときは cool 側も暖色に寄る片側反応だった。

- **`0.5+plain_pad`**: warm +4.23 / cool −3.84、contrast +8.07、CI [+1.04, +14.46]。**有意かつ両側。**
- **`0.5+plain_pad_token`**: warm +9.93 / cool −2.61、contrast +12.54、CI [+7.01, +17.67]。
  **有意かつ両側で、contrast は最大。**

対照（`alpha0.5`）は cool 側が +5.75 で両側にならない。pad を戻すことで
「寒色を選んだら寒くなる」が初めて成立した。

## alpha 0.7 と plain_token は使えない

- `0.7-plainpadtoken` は lum −23.30、Δtarget −0.0476。暗く沈み被写体も崩れる。
- `0.7-plaintoken` は lum +60.83、sat −57.44、ラプラシアン比 0.030、Δtarget −0.1516。破綻。
- **`plain_token` は診断も落ちる。** `invariants.common_weight_scale` と
  `invariants.duplicate_split` に失敗した（`adapter_vs_direct_official` は cosine 1.000000 で通過、
  alpha=0 も 0.999985 で通過）。全参照の重みを一律に倍にしても、参照を複製に分割しても
  出力が変わってはいけない、という不変条件を破る。token 単位で plain のノルムに揃えるため、
  全体の大きさに結果が依存してしまう。**採ってはいけない。**
- 他の 5 policy は診断を全通過（`adapter_vs_direct_official` cosine 1.000000、
  alpha=0 は plain_pad 系で 0.999996 と対照より良い）。

## 推奨

既定を `mask_skip1_v1`（= `mask-skip1-alpha0.5`）に据えたうえで、
**`hidden_norm: "plain_pad"` を足すことを勧める。**

- アーティファクトが消える（ラプラシアン比 0.85、線と白飛びが plain 並みに戻る）。
- 霞が薄まる（lum +13.05 → +4.27、sat −18.65 → −11.20）。
- 被写体を保つ（Δtarget +0.0027）。
- **両側 steering が有意になる**（+8.07、CI が 0 を跨がない）。
- 既定 policy に 1 フィールド足すだけで、alpha も skip_pa も変えない。

`plain_pad_token` は contrast が +12.54 とさらに強く Δtarget も +0.0059 で最良だが、
彩度の落ち込みが −16.90 と `plain_pad` の −11.20 より大きい。
**色の自然さを優先するなら `plain_pad`、効きを優先するなら `plain_pad_token`。**
展示の狙い（好みの方向へ穏やかに動かす）には `plain_pad` が合う。

なお本実験は現行 sampler で回した。e11 で非 SDE sampler も同じアーティファクトを減らすと
分かっている。両者は独立な軸なので、併用すればさらに改善する可能性があるが未検証。

## 残っているもの

- 宣言は `exhibit/configs/fan-strength.json` の `e12-hidden-norm`。
- 人による評価はしていない。画素指標と CLIP target_score と原寸目視のみ。
