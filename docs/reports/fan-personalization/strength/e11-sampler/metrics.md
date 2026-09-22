# e11-sampler 画素指標（metrics.md）

Δ はすべて **同じ variant の plain** との差。variant ごとに plain を焼き直しているので、生成設定そのものの違いは打ち消されている。

`laplacian_ratio` は原寸グレースケールの 4 近傍ラプラシアンの分散比（個人化 / plain）。1 未満は「個人化すると高周波が減る」、つまり細部が潰れていることを示す。`blown` は輝度 245 超の画素割合の差（ポイント）。

## control

設定: guidance 5.0 / steps 30 / `sde-dpmsolver++`。1枚あたり 5.02 秒。plain の絶対値: lum 126.55, laplacian分散 516.0, blown 7.412%

| policy | Δlum | Δsat | laplacian_ratio | Δblown | Δdarkline | Δedge | Δtarget | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy_exhibit` | -7.62 | -1.40 | 0.741 | +0.731 | -1.10 | +0.70 | +0.0001 | 24 |
| `mask-skip1-alpha0.5` | +13.05 | -18.65 | 0.391 | -1.753 | -2.90 | -2.63 | +0.0040 | 24 |
| `mask-skip1-alpha0.7` | +18.40 | -24.26 | 0.291 | -2.907 | -4.20 | -7.04 | -0.0072 | 24 |

## guidance7

設定: guidance 7.0 / steps 30 / `sde-dpmsolver++`。1枚あたり 5.02 秒。plain の絶対値: lum 125.8, laplacian分散 855.4, blown 12.507%

| policy | Δlum | Δsat | laplacian_ratio | Δblown | Δdarkline | Δedge | Δtarget | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy_exhibit` | -10.33 | +0.93 | 0.778 | -3.192 | -1.23 | -0.60 | +0.0030 | 24 |
| `mask-skip1-alpha0.5` | +17.33 | -26.44 | 0.475 | -3.930 | -2.99 | -2.60 | +0.0020 | 24 |
| `mask-skip1-alpha0.7` | +20.95 | -30.74 | 0.281 | -4.663 | -5.15 | -7.62 | -0.0080 | 24 |

## nonsde

設定: guidance 5.0 / steps 30 / `dpmsolver++`。1枚あたり 5.04 秒。plain の絶対値: lum 130.7, laplacian分散 386.0, blown 4.088%

| policy | Δlum | Δsat | laplacian_ratio | Δblown | Δdarkline | Δedge | Δtarget | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy_exhibit` | -1.74 | +4.42 | 0.865 | -0.106 | -0.64 | -0.98 | -0.0017 | 24 |
| `mask-skip1-alpha0.5` | +13.00 | -6.55 | 0.674 | -0.554 | -0.92 | -0.91 | -0.0071 | 24 |
| `mask-skip1-alpha0.7` | +14.85 | -13.01 | 0.605 | -1.070 | -2.11 | -4.57 | -0.0181 | 24 |

## steps50

設定: guidance 5.0 / steps 50 / `sde-dpmsolver++`。1枚あたり 8.05 秒。plain の絶対値: lum 129.82, laplacian分散 492.9, blown 8.131%

| policy | Δlum | Δsat | laplacian_ratio | Δblown | Δdarkline | Δedge | Δtarget | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy_exhibit` | -6.16 | +1.96 | 0.771 | +0.755 | -0.70 | +0.98 | +0.0004 | 24 |
| `mask-skip1-alpha0.5` | +12.42 | -22.85 | 0.451 | -2.094 | -2.41 | -1.72 | +0.0016 | 24 |
| `mask-skip1-alpha0.7` | +10.83 | -25.00 | 0.304 | -3.493 | -3.58 | -5.64 | -0.0166 | 24 |

## 対照との比較（mask-skip1-alpha0.5）

| variant | Δsat | laplacian_ratio | Δdarkline | 秒/枚 |
| --- | --- | --- | --- | --- |
| control | -18.65 | 0.391 | -2.90 | 5.02 |
| guidance7 | -26.44 | 0.475 | -2.99 | 5.02 |
| nonsde | -6.55 | 0.674 | -0.92 | 5.04 |
| steps50 | -22.85 | 0.451 | -2.41 | 8.05 |
