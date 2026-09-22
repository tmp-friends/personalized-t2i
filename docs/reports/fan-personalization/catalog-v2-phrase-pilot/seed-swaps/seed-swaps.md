# catalog-v2 シード入れ替えの記録（2026-09-22、見積もり）

採用した 20 件。`exhibit/configs/catalog-v2.json` の `seed_overrides` に入り、その 20 枚だけを再生成した。

| card | seed | 入れ替え前に弱かった項目 | 直った内容 |
|---|---|---|---|
| girl-c0-l3-t1-m1 | 7601 | lighting=weak | seed 7601: 日の当たる壁に輪郭のはっきりした落ち影。[差し替え][heldout] |
| girl-c1-l2-t1-m2 | 5601 | subject=weak | seed 5601: 右上に太陽とグレア、肌の青染まりが消えた。[差し替え] |
| girl-c2-l0-t3-m0 | 3601 | texture=weak, mood=weak | seed 3601: 屋根と空に厚い筆致、閉じ目＋口を閉じた微笑。[差し替え] |
| girl-c2-l2-t2-m1 | 6601 | texture=weak | seed 6601: 背後に太陽のグレア、人物はほぼ線なしの平塗り。[差し替え][heldout] |
| girl-c3-l1-t0-m0 | 2601 | mood=uncertain | seed 2601: 閉じ目＋穏やかな微笑、鉛筆線あり。[差し替え] |
| girl-c3-l1-t2-m2 | 2601 | texture=weak | seed 2601: 人物は線なしの平塗り（背景に線が残るのは境界的）。[差し替え] |
| student-c0-l3-t1-m1 | 2602 | lighting=weak | seed 2602: 書架と顔を横切る硬い斜光と落ち影。[差し替え][heldout] |
| student-c2-l0-t3-m0 | 2602 | texture=weak | seed 2602: 書架と服に厚い筆致。[差し替え] |
| student-c2-l2-t2-m1 | 6602 | texture=weak | seed 6602: 左上から光条、人物は線なしの平塗り。[差し替え][heldout] |
| student-c3-l2-t1-m3 | 1602 | color=uncertain, lighting=uncertain | seed 1602: 太陽のフレアとリムライト、パステルが明瞭。[差し替え] |
| traveler-c0-l3-t1-m1 | 1603 | color=uncertain, lighting=weak | seed 1603: 草原を横切る硬い落ち影、琥珀色。閉じ目で笑うため M0 との差は境界的。[差し替え][heldout] |
| traveler-c1-l1-t3-m1 | 4603 | texture=weak | seed 4603: 空と草に明確な厚塗りの筆致。[差し替え][heldout] |
| traveler-c1-l2-t1-m2 | 2603 | subject=uncertain | seed 2603: 太陽のグレアとリム、肌の灰染まりが消えた。[差し替え] |
| traveler-c2-l0-t3-m0 | 4603 | texture=uncertain | seed 4603: 曇天と草に厚い筆致。[差し替え] |
| traveler-c2-l2-t2-m1 | 2603 | lighting=weak | seed 2603: 頭上に太陽とグレア、リムライト。[差し替え][heldout] |
| traveler-c3-l1-t2-m2 | 4603 | texture=weak | seed 4603: 完全な線なし平塗り、夜のランタン。[差し替え] |
| barista-c0-l3-t1-m1 | 1604 | lighting=weak | seed 1604: 床と什器に窓枠の硬い落ち影。[差し替え] |
| barista-c1-l2-t1-m2 | 4604 | lighting=uncertain | seed 4604: 右の窓が強く白飛びし肩と髪にリム。[差し替え] |
| barista-c2-l0-t3-m0 | 3604 | texture=uncertain, mood=weak | seed 3604: 閉じ目＋穏やかな微笑、厚い筆致。黄土色のエプロンで低彩度は境界的。[差し替え] |
| barista-c2-l2-t2-m1 | 3604 | texture=weak | seed 3604: 輪郭線がほぼ消えた平塗り、窓からの強い光。[差し替え] |

## 見送った（当てたシードがどれも5項目を満たさなかった）

- girl-c0-l0-t0-m2 — lighting=weak — sketch carries no lighting cue at all; L0 indistinguishable from L2/L3
- girl-c0-l3-t3-m3 — lighting=weak, mood=uncertain — no cast shadow; eyes read open rather than half-closed
- girl-c1-l0-t2-m3 — lighting=weak, texture=weak, mood=weak, subject=uncertain — lineart present; face in profile, eye hidden; skin lavender
- girl-c1-l1-t3-m1 — texture=uncertain — heldout; painterly but brushstrokes not clearly thick
- girl-c1-l3-t1-m0 — lighting=weak, subject=weak — skin fully blue-tinted; no cast shadow
- girl-c2-l3-t3-m2 — lighting=weak — pale sky, no hard cast shadow
- girl-c3-l0-t0-m1 — lighting=uncertain, texture=uncertain — sketch lines only on the coat; overcast not readable
- girl-c3-l2-t1-m3 — lighting=weak, texture=weak — pink haze, no rim/glare; soft rendering, not cel
- student-c0-l0-t0-m2 — color=weak, subject=weak — grey rainy exterior: not warm, library swapped away
- student-c0-l2-t2-m0 — texture=weak, mood=uncertain — visible lineart; grin rather than gentle smile
- student-c1-l0-t2-m3 — lighting=weak, texture=uncertain, mood=weak — flat blue field, no lighting cue; eyes open
- student-c2-l3-t3-m2 — lighting=weak, texture=uncertain — no cast shadow; brushwork only in the bg
- student-c3-l0-t0-m1 — color=uncertain, texture=weak — clean anime, no pencil lines
- student-c3-l1-t0-m0 — texture=uncertain — rough lines only in the bg
- student-c3-l1-t2-m2 — color=weak, texture=weak — dark purple, not pastel; lineart present
- traveler-c0-l0-t0-m2 — color=uncertain, lighting=weak — olive/grey rather than amber; sketch has no lighting
- traveler-c0-l3-t3-m3 — lighting=uncertain — high contrast but the light direction is not readable
- traveler-c1-l0-t2-m3 — lighting=weak, mood=weak, subject=uncertain — eyes open; the grassland reads as snow
- traveler-c1-l3-t1-m0 — lighting=weak — bright, no cast shadow (grassland has no surface)
- traveler-c2-l3-t3-m2 — lighting=weak, texture=weak — flat green field, no sun; no brushstrokes
- traveler-c3-l0-t0-m1 — lighting=uncertain, subject=uncertain — grassland almost absent behind the close-up
- traveler-c3-l2-t1-m3 — lighting=uncertain, texture=weak — soft rendering, not cel; glow only
- barista-c0-l0-t0-m2 — color=weak — grey-brown, not amber
- barista-c0-l3-t3-m3 — lighting=weak, mood=uncertain — no cast shadow; eyes read open
- barista-c1-l0-t2-m3 — lighting=weak, subject=weak — skin fully blue-grey; no lighting cue
- barista-c1-l3-t1-m0 — subject=weak — figure tiny at the right edge, face in far profile
- barista-c2-l3-t3-m2 — lighting=weak — bright corridor but no hard cast shadow
- barista-c3-l0-t0-m1 — texture=weak, subject=uncertain — clean anime, no pencil lines; reads as a girl, not the same shopkeeper
- barista-c3-l1-t0-m0 — lighting=weak, texture=uncertain, mood=weak, subject=uncertain — bright pink bg, not night; laugh reads cheerful; reads as a girl
- barista-c3-l1-t2-m2 — texture=weak, subject=uncertain — lineart present; reads as a girl
- barista-c3-l2-t1-m3 — lighting=weak, mood=weak, subject=uncertain — no rim/glare; eyes open; reads as a girl
