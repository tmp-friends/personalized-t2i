# catalog-v2 光の軸パイロット 第1〜3ラウンド（2026-09-21）

親の `../README.md`（改訂案2まで）の続き。生成条件・seed・凍結CLIPの補助指標は同じ。
`exhibit/configs/*`・`exhibit/assets/*` は変更していない。確認（review）エントリも書いていない。
数値は補助指標と目視の見積もりであり、確認の代わりではない。

## 結論

- 提案していた「曇天／夜／夕景／日中」は不採用を推す。`sunset, orange sky` は全パレットへ暖色を持ち込み
  （画像平均 R−B が夕景だけ +43、他は −20〜0）、`blue sky` は暖色カードへ寒色を持ち込み、
  `cloudy sky`／`night sky` は図書館を屋外に変える（`grid-candidate-3a-girl.jpg`、`candidate-3a.json`）。
- 光は「向きと硬さ」で4分するとパレットから独立する（`candidate-5.json`）。
  - L0 `overcast, diffused light, soft shadows`
  - L1 `night, dim lighting, lamplight, dark background`
  - L2 `backlighting, rim light, sun glare`
  - L3 `harsh sunlight, cast shadow, high contrast`
- 同時に直したもの: 寒色 `cool color palette, blue background`（青い肌が解消、`s3-cool.jpg`）、
  夢のような雰囲気 `dreamy, half-closed eyes, looking away`（明るい光でも残る、`s3-mood3.jpg`）、
  ネガティブへの被写体保護 `from behind, facing away, back turned, faceless, scenery focus`
  （後ろ向き 5枚→0枚、`ab-negative-prompt.jpg`、全文は `negative-protected.txt`、75/77トークン）。
- プロンプトは正・負とも77トークンで警告なく切り捨てられる。正は64枚すべて最大77で余地がない。

## candidate-5 を4被写体64枚で試した結果

| | color | lighting | texture | mood | 4軸一致 |
|---|---|---|---|---|---|
| 初期文言（凍結CLIP top-1 /64） | 39 | 20 | 27 | 31 | 5 |
| candidate-5（凍結CLIP top-1 /64） | 29 | 41 | 40 | 44 | 7 |
| candidate-5（目視の見積もり /64） | 45 | 51 | 46 | 59 | 26（被写体も含む） |

目視の水準別: 光 L0 16 / L1 16 / L2 12 / L3 7、色 c3（鮮やか）4/16、描画 T0（水彩）5/16・T3（油彩）9/16。
グリッド: `grid-candidate-5-{girl,student,traveler,barista}.jpg`（行=色、列=光）。

## 文言では直らない衝突（64/64 に届かない理由）

profile は `texture = color XOR lighting` で決まるため、相反する水準が必ず同居する。

1. L3「硬い直射光・落ち影」× T2「平塗り・最小陰影」／T0「水彩の淡いにじみ」— 同じ絵への相反する指示。
   被写体が潰れた2枚（girl/barista の c1-l3-t2）はこの組合せ。
2. L2「逆光のグレア」× T3「油彩の厚塗り」— グレアが筆致に埋もれる（c1-l2 の4枚中3枚）。
3. c3「鮮やか」× L0「曇天」／L3 — 彩度が出ない。
4. c0「暖色」× L1「夜」— 夜が暖色を消す。
5. traveler の基本文に `upper body` がなく、夜や曇天で人物が遠景の全身になる（被写体破綻6枚中4枚）。

打ち手はいずれも設計の変更で、未実施・未決定: profile の割り当てを XOR から手選びに変える／
T2 を陰影の量ではなく線と塗りの性質で定義する／c3 を明度に依存しない定義にする／
traveler に `upper body` を足す／64枚すべての確認という条件を見直す。
