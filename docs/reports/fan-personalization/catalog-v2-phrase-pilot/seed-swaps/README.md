# 64枚の見積もりと1枚ごとのシード差し替え（2026-09-22）

2026-09-22 の定義（パステル・ラフスケッチ・禁止ペア8組）で生成した64枚を、コーディングエージェントが
厳格基準（見えて、かつ同じ軸の他3水準と区別できる）で目視した見積もり。確認（review）ではなく、
`exhibit/configs/cards-v2-review.json` には何も書いていない。

- 5項目すべて ok: 差し替え前 13/64 → 後 33/64（同一基準どうしの比較。以前の 14/64・28/64 とは見積もり者と
  プロファイルが違うので直接は比べられない）。表は `estimate-before.md`・`estimate-after.md`。
- 51枚のうち33枚に4シード、8枚にさらに4シードを試し、20枚を `seed_overrides` に採用した（`seed-swaps.md`）。
  スクラッチ生成は被写体シードで既存アセットと sha256 が一致し、採用20枚も `prepare.py --only` の結果と一致した。
- 差し替えが効いた例: `pair-traveler-c0-l3-t1-m1.jpg`（草地に硬い落ち影）、`pair-student-c0-l3-t1-m1.jpg`、
  `pairB-girl-c2-l2-t2-m1.jpg`。
- シードでは直らないもの: 強い日差しの落ち影（面が写り込むかがシード任せ、`axis-lighting-B.jpg`）、
  パステル×スケッチと曇天×スケッチ（`axis-texture-B.jpg`）、カフェ店員のパステル4枚が女性に見える
  （`contact-barista.jpg`）。
- heldout の9枚のうち `girl-c1-l1-t3-m1` だけ描画が uncertain。所有者の確認結果が出てから、
  落ちた場合は `histories-heldout.json` の cool-oil を `barista-c1-l1-t3-m1` に替える。

次の改訂候補（未実施）: 光3に影の落ちる先を名指しする（`on the ground`。`on the wall` は場面を壊す）、
禁止ペアに c3×t0 と l0×t0 を足す、カフェ店員の基本文を補強する。
