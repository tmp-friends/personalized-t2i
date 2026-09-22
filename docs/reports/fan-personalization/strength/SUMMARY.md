# FAN 強度実験 · 人間向けまとめ

実施日: 2026-09-22　対象: `exhibit/`（Illustrious-XL v2.0 + FAN）　計画: `docs/superpowers/plans/2026-09-22-fan-strength-experiments.md`

このファイルは人が書いた結論です。数値表・目視シート・AI 判定の集計は自動生成の [README.md](README.md) にあり、
このまとめはそこから読み取れることだけを書いています。画像は `e1-settings/` などの各実験フォルダにあります。

## 1. 結論（3 行）

1. **今回試した「効きを強くする」設定のうち、既存の基準（target 床 −0.01、history 改善 +0.005）を満たしたのは `strong_v1`（skip_pa [0]・plain pooled・ratio 0.1・alpha 0.5）だけ**でした。これは前回の refine / heldout の勝者と同じ設定です。
2. alpha 0.7 以上、attention mask、embed_gain、参照 2〜4 本の profiling はいずれも history（好みへの近さ）を伸ばしますが、**主題（お題）を壊す代償が床を超えます**。特に alpha 0.7 以上と embed_gain 1.5 以上は目視でも一目で分かる崩れ（別人・獣耳・紅葉・ノイズ）が出ます。
3. **人による評価はまだ行っていません。** 材料（ブラインド評価ページ `review.html`、264 + 144 + 108 ペア）は用意済みです。既定 policy は `legacy_exhibit` のまま変えていません。

## 2. 何をしたか

| 段階 | 内容 | 結果の場所 |
|---|---|---|
| 事前修正 P0〜P4 | legacy を alpha 0.4 でも計測（P0）、`strong_v1` を `fan-policies.json` に登録（P1）、`fan_probe.py` に skip_pa / pooled / mask を貫通（P2）、heldout 勝者と legacy の並置シート（P3）、`fan_block` に alpha を追加（P4） | README「参考: 正式チェーンの run」、`heldout-3bb8cd97/` |
| e1-settings（B1/B2/B3/B4/A4） | 設定のみのスイープ 11 条件 × 24 件 | README `e1-settings`、`e1-settings/` |
| e2-adapter（C1/C2） | embed_gain 1.5/2.0/2.5、fan_eos pooled、その組合せ | README `e2-adapter`、`e2-adapter/` |
| e3-references（A3/A5） | タグ句 vs 自然文の参照、重み集中（mixed-focus） | README `e3-references`、`e3-references/` |
| A2（論文レジーム） | base SDXL 1.0 + 自然文 + 公式設定で alpha 0 / 0.4 / 0.7 | `a2-paper-regime/contact-sheet.jpg`、`report.json` |
| e5-official-sampler（A6） | negative なし・非 SDE DPM++・50 steps（生成設定が違うので参考値） | README `e5-official-sampler` |
| AI 判定 | plain との 2 枚比較を Claude（sonnet、cat の一部は opus も）が判定 | README 各実験の「AI 判定」 |
| 人による評価 | **未実施**（`review.html` で実施可能） | README「人による評価」 |

正式チェーン（screen → refine → heldout）は同日に別セッションが再実行しており、前回の記録と同じ数値（heldout: `strong_v1` Δhistory +0.0131、Δtarget −0.0076、合格）を再現しました。

## 3. 実験ごとの読み取り

### e1-settings（設定のみ）

- **alpha**: 0.5 → 0.7 で Δhistory は +0.017 → +0.024 に伸びるが、Δtarget は −0.010 → −0.073 に急落。0.85 / 1.0 では history の伸びも頭打ち（+0.012 / +0.014）で target は −0.11 まで落ちる。目視では 0.7 で背景が紅葉に変わり、1.0 では別人（狐耳・金眼）になる。
- **legacy の alpha 0.4 と 0.5 はほぼ同じ**（Δhistory −0.0002）。legacy の効きの弱さは alpha ではなく skip_pa [0..7] が原因という計画の前提が裏付けられた。
- **skip_pa [0..3]**（B1）は history +0.004 で改善基準未満。中庸の層数では足りない。
- **attention mask**（B4）は数値検査（α=0 で plain と一致するか）で不合格。計画で懸念した「causal mask が padding mask に上書きされる」挙動が実際に出ている。画像も alpha 0.5 で target −0.066 と大きく崩れる。
- **profiling count 2 / 4**（A4）は history +0.014 / +0.015 で ratio 0.1（実質 1 本）と大差なく、target は −0.015 で床割れ。
- AI 判定: plain に対する勝率は多くの条件で 0.6〜0.78 だが、**target 維持率は alpha 0.7 で 0.18、0.85 で 0.04、1.0 で 0.00**。`strong_v1` は勝率 0.72、維持率 0.64。legacy（alpha 0.4）は勝率 0.50・引き分け 16 件で、plain とほぼ区別がつかない。

### e2-adapter（アダプタ側の小改修）

- **embed_gain（C1）は不採用**。1.5 で既に画像がピンクのノイズになり（Δtarget −0.117）、2.0 / 2.5 では完全に崩壊。`hidden = plain + w·(pers − plain)` は encoder の出力分布から外れるので、外挿は効かない。
- **fan_eos（C2）** は alpha 0.5 で Δhistory +0.017（`strong_v1` と同じ）、Δtarget −0.012 で床をわずかに割る。pooled を個人化しても history は伸びず、target だけ少し悪化した。SDXL の pooled 経路を個人化する価値は、この参照語句の設計では小さい。
- AI 判定: fan_eos alpha 0.5 は勝率 0.59・維持率 0.42。embed_gain 系は勝率 0.43〜0.52・維持率 0〜0.13。

### e3-references（参照の設計）

- **自然文の参照（A3）は、タグ句より Δhistory が小さい**（例: warm +0.025 vs warm-sentence +0.015、`strong_v1`）。ただし history_score は参照文そのものとの CLIP 類似度なので、タグ句と自然文の数値は直接比べられません。目視・AI 判定でも自然文で効きが強くなる傾向は見えなかった。
- **重み集中（A5、mixed-focus）は効く**。mixed（warm:cool = 2:1）の +0.014 に対し mixed-focus（4:1）は +0.022。同じ alpha で「どこに効かせるか」を鋭くする手段として有効。ただし target は −0.026 と悪化する。
- 4 つのタグ句履歴の数値は e1 と完全に一致し（同じ画像 hash）、生成が決定的であることも確認できた。
- 9 履歴の合計では `strong_v1` の Δtarget が −0.0157 となり床を割る（自然文・重み集中の履歴が target を余分に削る）。
- AI 判定: `strong_v1` 勝率 0.80、alpha 0.7 は 0.81 だが維持率は 0.43 vs 0.07。履歴別に見ると **cool 系は勝率 0.5 以下**で、plain（Illustrious の既定は寒色・セル調）と区別がつかない。効きが見えるのは warm 系と mixed 系。

### A2（論文レジーム: base SDXL 1.0 + 自然文）

- alpha 0.4 で埋め込みの cos は hidden 0.947・pooled 0.94、画像は主題を保ったまま色調・描画が少し動く程度。alpha 0.7 で hidden 0.82・pooled 0.58〜0.65、描画様式が大きく変わり（写実 → アニメ調）、猫の扱いや体型が変わる。
- つまり **FAN 本来の土俵でも alpha 0.4 の見た目の効きは控えめ**で、exhibit の「弱い」体感は Illustrious 固有の問題だけではない。強くすると主題が動くのはこちらでも同じ。

### e5-official-sampler（A6、参考値）

- 公式に寄せた生成設定（negative なし・非 SDE の DPM++・50 steps）では、**差分はむしろ小さくなった**: `strong_v1` の Δhistory は +0.0055（本番設定では +0.0170）、alpha 0.7 でも +0.0096。legacy と plain の差も +0.0026。
- 「SDE の毎 step ノイズと長い negative が個人化の差分を洗い流している」という仮説は支持されない。本番の生成設定は効きの可視性の点で不利ではない。
- 生成設定が違うので rules は拘束しない（参考値）。
- AI 判定: `strong_v1` 勝率 0.68・維持率 0.38、alpha 0.7 は勝率 0.78・維持率 0.21（本番設定の e1 と同じ傾向）。

## 4. 推奨

1. **既定 policy は `legacy_exhibit` のまま**（計画の制約どおり）。切り替え候補は引き続き `strong_v1` の 1 つで、その根拠は heldout 合格＋今回の再現。
2. 「効きをもっと強く」の要望に対しては、**alpha を上げる方向は採らない**（0.6 以上で主題が壊れる）。次に試す価値があるのは、target を壊さずに history を伸ばせる可能性がある D1（層別 alpha）と D3（noise 空間で個人化差分だけを増幅）で、これらは今回未実施。
3. 参照設計では **重み集中（A5）を UI の strength / aspect_gain に接続する**（C4）のが安価で、来場者のダイヤルが実際に効くようになる。
4. 人による評価を行う場合は `review.html` を使い、先に `strong_v1` vs plain と `mixed-focus` の効果を確かめるのが順当。cool 系の好みは Illustrious の既定と重なるため、評価者に「差が見えない」ケースが多く出ることを見込んでおく。

## 5. 制約・注意

- AI 判定の判定者間一致率は低い（cat 44 ペアで sonnet と opus の選好一致 0.36、target 維持の一致 0.68）。policy 単位の傾向（target 維持率の崩れ）は CLIP 指標と整合するが、**個々のペアの判定は信頼しないでください**。
- 2 枚比較は「同じ topic × seed の plain 画像」が全条件で繰り返し現れるため、判定者が plain 側を推測できる余地があります（完全なブラインドではない）。人による評価では policy 同士の比較（`strong_v1` vs alpha 0.7 など）を混ぜることを勧めます。
- history_score は参照語句との CLIP 類似度で、参照の文言を変えた履歴（自然文）同士の比較にしか使えません。
- `docs/reports/fan-personalization/strength/` は画像で約 100 MB あります（うち `*/judge/pairs/` が約 65 MB）。Git に入れる範囲は判断してください。AI 判定の回答 JSON と判定者に渡した指示は `judge-answers/` にあり、`--judge-answers judge-answers/*.json` で集計を再現できます。

## 6. 再実行

```bash
# 実験（GPU、順に）
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py strength --config exhibit/configs/fan-strength.json --experiment e1-settings --resume
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py strength --config exhibit/configs/fan-strength.json --experiment e2-adapter --resume
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py strength --config exhibit/configs/fan-strength.json --experiment e3-references --resume
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/evaluate_fan.py strength --config exhibit/configs/fan-strength.json --experiment e5-official-sampler --resume
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=fan-repro/.work/upstream:exhibit/src fan-repro/.venv/bin/python exhibit/scripts/fan_paper_regime.py
# レポート（AI 判定・人の回答 JSON があれば --judge-answers に渡す）
PYTHONPATH=exhibit/src exhibit/.venv/bin/python exhibit/scripts/build_strength_report.py \
  --extra-run exhibit/outputs/fan-evaluation/3bb8cd97a9c11518ce5b31b26c1a9ac61beabc70fc605d1e8ae8210f3b7ff3df=heldout-3bb8cd97 \
  --judge-answers docs/reports/fan-personalization/strength/judge-answers/*.json
```
