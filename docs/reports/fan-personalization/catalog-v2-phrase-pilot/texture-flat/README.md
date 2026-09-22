# 描画2「線のない平塗り」の言い換えパイロット（2026-09-22）

現行の `flat color, lineless, no lineart` は CLIP-L で `flat color , line less , no line art` に割れる。
`line` が2回入り、CLIP は否定を読まないので、混ぜるほど線を足す向きに働くはずだった。
line を含まず否定もしない候補を7つ作り、(1) FAN 参照として（実験 `e9-texture-flat`）、
(2) カードの描画句そのものとして（8枚 × 8文言 = 64枚）測った。
判定は画素指標と目視で、harness の CLIP cosine は使っていない。

結論を先に書く。**FAN 参照としては、どの候補も現行と区別できない。**
**カード側では `flat graphic illustration, solid color fill`（`t2-flat-graphic`）だけが線を減らす。**

## 候補と token 数

すべて CLIP-L・bigG とも同数で、`line` トークンを含まない。表と内訳は `candidates.json`。
現行は 9 token、`student-c1-l0-t2-m3` と `barista-c1-l0-t2-m3` がちょうど 77 に貼り付いている。
`t2-flat-graphic` は 7 token なので、この2枚が 75 に下がり余裕が戻る。

| history | 句 | token |
| --- | --- | --- |
| `t2-current` | `flat color, lineless, no lineart` | 9 |
| `t2-cel-contrast` | `anime screencap, cel shading`（対極。候補ではない） | 6 |
| `t2-flat-smooth` | `flat color, flat shading, smooth edges` | 8 |
| `t2-flat-minimal` | `flat color illustration, minimal shading` | 6 |
| `t2-flat-vector` | `flat color, vector art, smooth shapes` | 8 |
| `t2-flat-borderless` | `flat colors, borderless, soft edges` | 8 |
| `t2-flat-matte` | `flat color, matte, simple shapes` | 7 |
| `t2-flat-fill` | `flat color, solid color fill, smooth edges` | 9 |
| `t2-flat-graphic` | `flat graphic illustration, solid color fill` | 7 |

`borderless` は `border` + `less` に割れる。弱い否定なので候補としては筋が悪いが、比較のため残した。
自前の追加は `t2-flat-fill` と `t2-flat-graphic` の2つ。

## 指標

`edge` は色の境目も数えるので、べた塗りでも色面の縁で高く出る。線の量の判定には使えない。
そこで原寸で測る `darkline`（自分自身の GaussianBlur(2.0) より 18 以上暗い画素の割合）を足した。
細い暗い線だけを数える。定義と全数値は `metrics.md` と `metrics.json` / `card-metrics.json`。

## Part 1 — FAN 参照。文言は効かない

実験 `e9-texture-flat`（topic `cat`/`forest`/`cafe` × seed 2 × history 10 × policy 3、186枚）。
`exhibit/outputs/fan-evaluation/strength/98ee313e2f0c496e3d3403200fcdf5533abae3d8cf2ce7286419e7255bd7bbdb`。

既定 policy `legacy_exhibit` での plain との Δdarkline は、どの候補も −0.7〜−1.6 に収まる。
セル6個のばらつき（sd 1.0〜1.4）より小さく、候補間の差は読み取れない。
決め手は対極の `t2-cel-contrast`（線を足すはずの句）が −0.69 で、候補と同じ側に同じ量だけ動くこと。
**参照の意味ではなく、参照を混ぜたこと自体が線をわずかに減らしている。**

`strong_v1` と `skip1-all-alpha0.5` では Δdarkline が −3.9〜−5.6、Δedge が −3.1〜−5.9 と大きく動くが、
`t2-cel-contrast` も −4.80 と同じだけ下がる。Δsat は −41〜−45。
これは既知の霞（[strength の e6–e8](../../strength/README.md)）で、平塗りではない。
`sheet-cat-230923.jpg` の3・4列目を見れば、線が減ったのではなく絵が溶けているとわかる。

参照ごとの絵の違い自体はある。同じ topic × seed で history を変えたときの画素差は、
plain との差に対して `legacy_exhibit` で 0.74、`strong_v1` で 0.49 の比。
つまり参照は絵を変えてはいるが、変える向きが句の意味と結びついていない。

## Part 2 — カード側。`t2-flat-graphic` だけ線が減る

texture 2 の4プロファイル × `girl` / `student` の8枚を、描画句だけ差し替えて同じ seed・同じ設定で生成した
（`exhibit/outputs/pilot-texture-flat/cards/`、assets には書いていない）。

| 候補 | Δdarkline | Δedge | 8枚中で改善 |
| --- | --- | --- | --- |
| `t2-flat-graphic` | **−0.275** | +0.03 | 5 |
| `t2-flat-vector` | +0.070 | −0.52 | 3 |
| `t2-flat-matte` | +0.189 | +0.43 | 3 |
| `t2-flat-borderless` | +0.437 | +1.33 | — |
| `t2-flat-smooth` | +0.819 | +1.35 | — |
| `t2-flat-fill` | +0.952 | +2.06 | — |
| `t2-flat-minimal` | +1.031 | +1.62 | — |

現行より線が減るのは `t2-flat-graphic` だけ。ほかはすべて増える。
`t2-flat-vector` は Δedge が最小だが darkline は増えており、色面の境が減っただけで線は減っていない。

目視でも同じ。`cards-zoom-lineart.jpg`（原寸切り出し）で、現行は顎と目に黒い輪郭が残る。
`t2-flat-graphic` は4つのうち輪郭がいちばん少なく、線が色の境に置き換わっている。
`t2-flat-vector` は下段（`girl-c0-l2-t2-m0`）で顔と髪にはっきり線が出ており、数字と一致する。
1枚あたりの振れは ±3 と大きく、8枚では有意とは言えない。効果の向きが揃っているだけ。

`t2-flat-vector` は `girl-c2-l2-t2-m1` で顔が白く潰れた（`cards-candidates-compare.jpg` の4行目中央）。
この候補は採らない。

## 推奨

**`flat graphic illustration, solid color fill`**（`t2-flat-graphic`、7 token）。

- カード側で現行より線が減る唯一の候補（8枚中5枚、平均 Δdarkline −0.275）。
- `line` トークンを含まず否定もしないので、FAN 参照として混ぜても線を足す向きには働かない。
- 7 token なので 77 に貼り付いた2枚に余裕が戻る。

ただし **これは描画2の見た目を大きく変える施策ではない**。
カード側の差は小さく、FAN 参照側では差が出ない。
個人化が効かない・暗くなるという本題は文言では直らない。

## warm-sun — 暗く寒くなる問題は policy 側

`warm color palette, amber tones` + `harsh sunlight, hard cast shadow, high contrast` の2参照。

| policy | Δwarmth | Δlum | Δsat | Δcontrast |
| --- | --- | --- | --- | --- |
| `legacy_exhibit` | −2.00 | −8.10 | −4.19 | +2.46 |
| `strong_v1` | +0.22 | +13.81 | −44.72 | −9.61 |
| `skip1-all-alpha0.5` | +1.32 | +16.68 | −41.49 | −10.22 |

既定 policy では、暖色と強い日差しを参照しているのに **暖かくも明るくもならない**（warmth −2.0、lum −8.1、sat −4.2）。
報告された症状がそのまま数字に出ている。
skip1 系では lum は上がるが sat が −41 以上落ちる。明るくなったのではなく白く飛んでいる。
どの policy でも「暖かく・明るく・コントラストが高い」方向には動かない。

なお `strong_v1` と `skip1-all-alpha0.5` は、参照が1件の history では完全に同じ値になる。
profiling が `ratio 0.1` でも `all` でも、1件なら選びようがないため。
差が出るのは2参照の `warm-sun` だけ。今後の実験で単一参照を使うときは policy を1つに減らしてよい。

## 見るべき画像

- `cards-zoom-lineart.jpg` — 原寸切り出し。線が残るかどうかはここで判断する。
- `cards-candidates-compare.jpg` — 8文言 × 3枚。
- `cards-t2-*.jpg` — 文言ごとに8枚。
- `sheet-cat-230923.jpg` ほか5枚 — Part 1。列は plain / legacy_exhibit / strong_v1 / skip1-all-alpha0.5。

## 残していないこと

- 人による評価はしていない。すべて画素指標と目視。
- `t2-flat-graphic` を採るなら描画2の全16枚を作り直して確認が要る。今回は8枚だけ。
- `exhibit/configs/catalog-v2.json` は触っていない。実験宣言と history だけを
  `fan-strength.json` / `histories-strength.json` に残した。
