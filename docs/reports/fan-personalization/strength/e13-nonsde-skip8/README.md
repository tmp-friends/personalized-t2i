# e13-nonsde-skip8 · 非 SDE sampler での legacy と mask-skip8（2026-09-23）

展示の既定を `legacy_exhibit` と `mask-skip8` のどちらにするかの判断材料。
生成設定は e11b と同じ非 SDE（DPM++ 2M・Karras・30 steps）。

- policy は `mask-skip8-alpha0.5` / `0.7`、`legacy_exhibit` は暗黙、**78 枚**
- history warm / cool / t2-flat-graphic / warm-sun、topic cat / forest / cafe、seed 2 本
- 生成設定を上書きしているので rules は参考値

**結論: `mask-skip8-alpha0.5` を推す。** 非 SDE のもとでアーティファクトは 3 policy とも出ず、
ドリフトはほぼゼロ、warm−cool が唯一有意かつ両側。alpha 0.7 は猫耳が出るので採らない。

## 数値（Δ は plain 比、n=24）

| policy | Δlum | Δsat | laplacian_ratio | Δblown | Δdarkline | Δedge | Δtarget |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy_exhibit` | −1.74 | +4.42 | 0.865 | −0.11 | −0.64 | −0.98 | −0.0017 |
| **`mask-skip8-alpha0.5`** | **+0.10** | **+1.80** | **1.096** | **+0.30** | **+0.99** | **+3.43** | **−0.0038** |
| `mask-skip8-alpha0.7` | +1.94 | −5.43 | 0.973 | −0.49 | +0.44 | +3.10 | −0.0054 |

**アーティファクトは 3 policy とも出ていない。** ラプラシアン比は 0.865〜1.096 で、
`mask-skip8-alpha0.5` は 1 を超える（plain より細部が多い）。
e10 の現行 sampler では `mask-skip1-alpha0.5` が 0.391 だったので、
**sampler の変更だけで白茶け・線の消失は解消している。**

ドリフトも小さい。`mask-skip8-alpha0.5` は lum +0.10 / sat +1.80 とほぼ無変化で、
legacy（lum −1.74 / sat +4.42）と同程度。

## warm − cool

| policy | warm | cool | contrast | 95% CI | 有意 | 両側 |
| --- | --- | --- | --- | --- | --- | --- |
| `legacy_exhibit` | +2.88 | −1.30 | +4.18 | [−0.34, +8.39] | − | 両側 |
| **`mask-skip8-alpha0.5`** | **+2.19** | **−1.85** | **+4.04** | **[+1.38, +6.38]** | **有意** | **両側** |
| `mask-skip8-alpha0.7` | +1.84 | −3.11 | +4.95 | [−1.43, +11.26] | − | 両側 |

3 policy とも両側（warm で暖色へ、cool で寒色へ）。
**CI が 0 を跨がないのは `mask-skip8-alpha0.5` だけ。** legacy は +4.18 とほぼ同じ大きさだが
下限が −0.34 で有意に届かない。mask を足すと効きの大きさは変わらず**ばらつきが小さくなる**。

history ごとの Δwarmth（`metrics.md` に全表）:

| policy | warm | cool | t2-flat-graphic | warm-sun |
| --- | --- | --- | --- | --- |
| `legacy_exhibit` | +2.88 | −1.30 | +0.98 | +3.27 |
| `mask-skip8-alpha0.5` | +2.19 | −1.85 | −0.47 | +1.15 |
| `mask-skip8-alpha0.7` | +1.84 | −3.11 | −1.72 | +1.51 |

## 被写体のアーティファクト — alpha 0.7 で猫耳

`sheet-cat-230923.jpg` の 3 行目（`t2-flat-graphic`）の最終列、
**`mask-skip8-alpha0.7` で少女に茶色の猫耳が生えている。**
owner が `mask_skip1_v1` を取り下げた理由と同じ症状が、skip8 でも alpha 0.7 では出る。

`mask-skip8-alpha0.5` と `legacy_exhibit` では、
cat topic の 4 history すべてで猫耳は出ていない。

なお個人化列はいずれも plain の黒猫を白猫に、服装も変える。
これは legacy でも skip8 でも同じに起きるので mask 固有ではなく、
参照を混ぜたことによる場面の変化。

## 目視

`crop-cat-warm.jpg` / `crop-cafe-warm.jpg`（原寸 512 px、列 plain / legacy / skip8 0.5 / skip8 0.7）。
4 列とも線がはっきり残り、目の輪郭も崩れず、白茶けもない。
**skip8 と legacy の見分けは難しい。** 差は色味の微妙な寄り方だけで、描画品質の差はない。

## 推奨

**`mask-skip8-alpha0.5` + 非 SDE sampler。**

- warm−cool が有意かつ両側になる唯一の policy（+4.04、CI [+1.38, +6.38]）。
  legacy は同じ大きさ（+4.18）だがばらつきが大きく有意に届かない。
- ドリフトは実質ゼロ（lum +0.10 / sat +1.80）。
- アーティファクトなし（ラプラシアン比 1.096）。猫耳も出ない。
- Δtarget −0.0038 は legacy の −0.0017 よりわずかに悪いが小さい差。

legacy を選んでも見た目の破綻はなく、Δtarget は僅かに良い。
**「効きの一貫性」を取るなら skip8 0.5、「現状維持の安全」を取るなら legacy。**
両者の絵の差は目視では小さいので、有意な両側反応が得られる skip8 0.5 を勧める。

`mask-skip8-alpha0.7` は contrast が最大（+4.95）だが有意でなく、猫耳が出るので採らない。

## 未検証

e12 で `hidden_norm: "plain_pad"` が skip_pa [0] のアーティファクトを消し、
warm−cool を有意な両側（+8.07）にすることが分かっている。
**skip8 + plain_pad + 非 SDE の組み合わせは試していない。**
contrast は plain_pad 系（+8.07〜+12.54）が skip8（+4.04）の 2〜3 倍なので、
効きを求めるならそちらを次に確かめる価値がある。

## 残っているもの

- 宣言は `exhibit/configs/fan-strength.json` の `e13-nonsde-skip8`。
- 人による評価はしていない。画素指標と CLIP target_score と原寸目視のみ。
