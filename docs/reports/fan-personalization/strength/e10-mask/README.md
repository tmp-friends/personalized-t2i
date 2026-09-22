# e10-mask · 参照だけを mask したあとの alpha 再掃引（2026-09-23）

`use_attn_mask=true` が「参照の pad トークンを attention の key から外す、それだけ」になった後の
alpha 掃引。pooled を EOS で個人化する fan_eos も同じ枠に入れた。

- 実験 `e10-mask`、10 policy + legacy 暗黙、7 history × topic 3 × seed 2、**468 枚**
- `exhibit/outputs/fan-evaluation/strength/0bf81bf739ec880cd083aaaa1cd1e222cc3bfe8600765ef4d9914e03878de6b5`
- 診断は **11 policy すべて通過**（後述）

結論を先に書く。

- **mask は被写体の保持だけを大きく改善する。** 同 alpha で Δtarget が −0.028 → −0.006。
  だが彩度・コントラスト・エッジのドリフトは縮まず、edge はむしろ悪化する。
- **mask は強い暖色バイアスを足す。** skip_pa [0] の mask ありは warmth +7〜+10。
  参照が warm でも cool でもまとめて暖色へ寄る。
- **両側に振れてかつ CI が 0 を跨がない policy は 1 つもない。**
  alpha 0.85 以上で warmth の contrast は有意になるが、cool 側も plain より暖色のままで片側だけ。
- **自然文はタグより強いまま。** pad を塞いでも逆転しなかった。
- **fan_eos は cool 側を改善しない。** むしろ僅かに悪化させる。

## (a) 汎用ドリフトと (c) 被写体保持

全 7 history の平均Δ（n=42）。全表は `metrics.md`。

| policy | warmth | lum | sat | contrast | edge | darkline | Δtarget |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy_exhibit` | −0.74 | −6.21 | −2.19 | +1.47 | +0.68 | −1.07 | −0.0000 |
| `nomask-skip1-alpha0.7` | −0.43 | +19.22 | −21.78 | −7.85 | −3.98 | −4.40 | −0.0280 |
| `nomask-skip1-alpha0.7-faneos` | −0.28 | +18.07 | −22.59 | −7.66 | −3.28 | −4.36 | −0.0260 |
| `mask-skip1-alpha0.5` | +9.58 | +12.22 | −19.01 | −6.46 | −1.49 | −2.77 | **+0.0030** |
| `mask-skip1-alpha0.7` | +10.01 | +16.68 | −22.52 | −8.40 | −6.09 | −3.98 | −0.0060 |
| `mask-skip1-alpha0.85` | +7.16 | +18.76 | −22.61 | −8.33 | −7.07 | −3.97 | −0.0200 |
| `mask-skip1-alpha1.0` | +8.10 | +23.28 | −24.16 | −9.26 | −7.35 | −4.26 | −0.0260 |
| `mask-skip1-alpha0.7-faneos` | +9.78 | +16.53 | −22.59 | −8.30 | −6.08 | −3.89 | −0.0070 |
| `mask-skip1-alpha0.85-faneos` | +7.62 | +19.19 | −23.00 | −8.31 | −6.99 | −3.93 | −0.0200 |
| `mask-skip8-alpha0.5` | −0.79 | −3.16 | −10.94 | −0.14 | +3.79 | −0.21 | −0.0060 |
| `mask-skip8-alpha0.7` | −0.21 | −7.85 | −16.33 | −2.33 | +2.73 | −1.46 | ±0.0000 |

### 同 alpha で mask の有無（skip_pa [0]・alpha 0.7・plain pooled）

| 指標 | no-mask | mask | 変化 |
| --- | --- | --- | --- |
| Δtarget | −0.028 | −0.006 | **8 割改善** |
| lum | +19.22 | +16.68 | 13% 縮小 |
| darkline | −4.40 | −3.98 | 10% 縮小 |
| sat | −21.78 | −22.52 | ほぼ同じ |
| contrast | −7.85 | −8.40 | やや悪化 |
| edge | −3.98 | −6.09 | **5 割悪化** |
| warmth | −0.43 | +10.01 | 暖色バイアスが出現 |

**ドリフトは全体としては縮まない。** 縮んだのは被写体保持と明度・線だけで、
エッジは逆に増える。mask の主な効果は「被写体を守ること」と「暖色へ寄せること」。

### `mask-skip8-alpha0.5` と `legacy_exhibit`

| 指標 | legacy | mask-skip8-alpha0.5 |
| --- | --- | --- |
| warmth | −0.74 | −0.79 |
| lum | −6.21 | −3.16 |
| sat | −2.19 | **−10.94** |
| contrast | +1.47 | −0.14 |
| edge | +0.68 | **+3.79** |
| darkline | −1.07 | −0.21 |
| Δtarget | −0.0000 | −0.0060 |

こちらも縮んでいない。明度の落ち込みは半分になるが、彩度は 5 倍、エッジは 5 倍に増える。

## (b) history contrast

CI はブートストラップ 2000 回、上下の群を独立にリサンプル。全表は `metrics.md`。
`audit.md` の Table E は warm-sun も含めて warmth 期待をまとめた別集計。

### warm − cool（タグ参照）· warmth

| policy | warm | cool | contrast | 95% CI | 有意 | 両側 |
| --- | --- | --- | --- | --- | --- | --- |
| `legacy_exhibit` | +1.87 | −2.50 | +4.38 | [−3.94, +12.82] | − | 両側 |
| `nomask-skip1-alpha0.7` | −3.33 | −3.92 | +0.60 | [−15.72, +17.51] | − | − |
| `mask-skip1-alpha0.5` | +11.91 | +5.75 | +6.16 | [−1.89, +14.04] | − | − |
| `mask-skip1-alpha0.7` | +13.97 | +8.74 | +5.24 | [−5.17, +14.41] | − | − |
| `mask-skip1-alpha0.85` | +14.59 | +3.33 | +11.26 | [−3.52, +27.52] | − | − |
| `mask-skip1-alpha1.0` | +17.03 | +3.11 | +13.93 | [−8.25, +35.78] | − | − |
| `mask-skip1-alpha0.7-faneos` | +13.83 | +9.44 | +4.39 | [−5.86, +14.53] | − | − |
| `mask-skip1-alpha0.85-faneos` | +14.07 | +4.45 | +9.62 | [−5.09, +24.51] | − | − |
| `mask-skip8-alpha0.5` | +2.79 | −2.15 | +4.95 | [−2.61, +12.31] | − | 両側 |
| `mask-skip8-alpha0.7` | +0.62 | +0.41 | +0.21 | [−13.18, +12.59] | − | − |

**有意な policy はゼロ。両側なのは `legacy_exhibit` と `mask-skip8-alpha0.5` だけ。**
skip_pa [0] の mask ありは cool 側が +3〜+9 で、寒色を指定しても plain より暖色になる。

`audit.md` の Table E（warm-sun も含めた warmth 期待の集計、n 上 18 / 下 12）では
`mask-skip1-alpha0.85`（+10.84、CI [+1.04, +20.47]）、
`mask-skip1-alpha0.85-faneos`（+9.52、CI [+0.77, +19.37]）、
`mask-skip1-alpha1.0`（+15.29、CI [+1.80, +28.14]）が有意になる。
だが下げろ群の平均はそれぞれ +2.49 / +3.71 / +1.04 と正のままで、**いずれも片側**。

### warm − cool（自然文参照）· warmth

| policy | contrast | 両側 | タグの contrast | どちらが強いか |
| --- | --- | --- | --- | --- |
| `nomask-skip1-alpha0.7` | +8.27 | 両側 | +0.60 | 自然文 |
| `mask-skip1-alpha0.5` | +4.67 | − | +6.16 | タグ |
| `mask-skip1-alpha0.7` | +7.39 | − | +5.24 | 自然文 |
| `mask-skip1-alpha0.85` | +13.20 | − | +11.26 | 自然文 |
| `mask-skip1-alpha1.0` | +18.89 | 両側 | +13.93 | 自然文 |
| `mask-skip8-alpha0.5` | +4.37 | 両側 | +4.95 | タグ |
| `mask-skip8-alpha0.7` | +5.00 | 両側 | +0.21 | 自然文 |

**pad を塞いでも自然文の優位は消えない。** 7 policy 中 5 つで自然文が上。
タグが勝つのは alpha 0.5 の 2 点だけで、差もわずか。
短いタグが不利なのは pad 比率だけが理由ではない。

### cel − flat-graphic · darkline

全 policy で contrast は ±0.89 に収まり、CI はすべて 0 をまたぐ。
線は参照の向きに反応しない。e9 の結論と変わらない。

## 推奨

**両側の steering と被写体保持を同時に満たす mask policy はない。**

その条件に最も近いのは `mask-skip8-alpha0.5`。タグでも自然文でも両側に振れ、
|Δtarget| は 0.006。ただし contrast は +4.95 で有意ではなく、`legacy_exhibit` の +4.38 と大差ない。
そのうえ彩度を 11 落とし、エッジを 3.8 増やす。**legacy を置き換える理由は現時点でない。**

強い反応が欲しいなら `mask-skip1-alpha0.85` 以上で warmth の contrast は有意になるが、
寒色を指定しても暖色に寄るため「好みに合わせる」用途には使えない。

被写体保持だけを見るなら mask は明確に有効で、`mask-skip1-alpha0.5` は
Δtarget +0.003 と plain より僅かに良い。効きを上げる別の軸と組み合わせる価値はある。

## 既定 policy になった `mask_skip1_v1` について

この測定中に別担当が `mask_skip1_v1` を登録し、`fan-policies.json` の既定にした。
中身は本実験の `mask-skip1-alpha0.5` と同一（alpha 0.5・skip_pa [0]・mask あり・plain pooled・profiling all）。
本実験の測定値をそのまま読み替えられる。

| 観点 | 値 | 評価 |
| --- | --- | --- |
| Δtarget | **+0.003** | 全 policy 中で最良。plain より僅かに良い |
| warmth | +9.58 | 参照によらず暖色へ寄る |
| sat | −19.01 | legacy（−2.19）より 9 倍大きい |
| edge | −1.49 | legacy（+0.68）より線が減る |
| warm − cool（タグ） | +6.16、CI [−1.89, +14.04] | 有意でない |
| cool 参照の warmth | **+5.75** | 寒色を指定しても plain より暖色 |

**当初報告された症状は直っている。** `warm-sun`（暖色 + 強い日差し）で
legacy は warmth −2.00 / lum −8.10 と「暗く寒く」なっていたが、
`mask_skip1_v1` では warmth +10.66 / lum +9.47 と、指定どおり暖かく明るくなる。

**被写体保持は明確に良い。だが「寒色が選ばれたら寒くする」はできていない。**
展示の狙いが「選んだ好みの方向へ動かす」ことなら、この既定では暖色側しか表現できない。
両側に振れるのは `legacy_exhibit` と `mask-skip8-alpha0.5` の 2 つだけで、
どちらも contrast は有意でない。

## 診断ゲート — 今回は全通過

11 policy すべてが `passed=True`。topic `cat` / history `single` の値。

| 検査 | mask あり | mask なし・legacy |
| --- | --- | --- |
| `alpha_zero_vs_no_reference` cosine | 0.999984（rmse 1.87e-03） | 0.999985（rmse 1.67e-03） |
| `fan_no_reference_vs_pipeline` cosine | 1.000000 | 1.000000 |

### 前回の失敗がもう当てはまらない理由

9/22 の最初の e10-mask では mask あり 6 policy が `fan_no_reference_vs_pipeline` に落ちた
（cosine 0.718）。当時は `use_attn_mask=true` が target 自身の 2 行にも padding mask を配っており、
参照ゼロの encode でも pad 27 個が動いて素の pipeline と一致しなかった。
いまは参照の pad だけを key から外すので、参照なしの encode は素の pipeline と完全一致する。

そのため **plain 列（`legacy_exhibit` で生成）がどの policy の基準としても正しい**。
policy ごとの plain は要らない。

前回 `--resume` した 390 枚は、GPU worker が 22:20:03 に起動し
`fan_adapter` の修正が 22:23:19 に入ったため、修正前の意味で焼かれていた。
保存画像を現在のコードで再生成すると別のバイト列になることを確認したので、
その結果は破棄し、今回 468 枚を最初から焼き直した。

## 見るべき画像

`sheet-<topic>-<seed>.jpg` 6 枚。列は plain / legacy / 10 policy、行は history。
列見出しは短縮名（`mask .5` = `mask-skip1-alpha0.5`、`mask8 .7` = `mask-skip8-alpha0.7`、
`eos` 付きが fan_eos）。

## 残っているもの

- 宣言は `exhibit/configs/fan-strength.json` の `e10-mask`。
  policy が 10 本だと 8 history で 534 枚となり 512 枚の上限を超えるため、
  どの contrast にも使わない `t2-current` を外して 7 history にした。
  併走させていた `e10-faneos` の宣言は、policy が e10-mask に入ったので削除した。
- history は既存のものを再利用（`warm`/`cool` は `histories-screen.json`、
  残りは `histories-strength.json`）。新規追加なし。
- `audit.md` / `audit.json` は `audit_direction.py` の出力（`--no-samples`、この 1 実験のみ）。
  同スクリプトは `datetime.UTC` を使うため Python 3.11 以上が要るが、
  numpy と Pillow が入っているのは Python 3.10 の `fan-repro/.venv` だけなので、
  実行時に別名を定義する薄いラッパ経由で回した。スクリプト自体は変更していない。
- 人による評価はしていない。すべて画素指標と CLIP target_score と目視。
