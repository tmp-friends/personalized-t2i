# e12-hidden-norm 画素指標（metrics.md）

Δ は同じ topic × seed の plain との差（history 4 本 × topic 3 × seed 2、n=24）。
`laplacian_ratio` は原寸ラプラシアン分散の個人化/plain 比。1 に近いほど細部が残る。
`blown` は輝度 245 超の画素割合の差。

| policy | Δlum | Δsat | laplacian_ratio | Δblown | Δdarkline | Δedge | Δtarget | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy_exhibit` | -7.62 | -1.40 | 0.741 | +0.73 | -1.10 | +0.70 | +0.0001 | 24 |
| `mask-skip1-alpha0.5` | +13.05 | -18.65 | 0.391 | -1.75 | -2.90 | -2.63 | +0.0040 | 24 |
| `mask-skip1-alpha0.5-plainpad` | +4.27 | -11.20 | 0.846 | +0.93 | -0.82 | -0.39 | +0.0027 | 24 |
| `mask-skip1-alpha0.5-plainpadtoken` | +4.21 | -16.90 | 0.866 | +1.22 | -1.05 | +0.49 | +0.0059 | 24 |
| `mask-skip1-alpha0.7-plainpad` | +13.77 | -15.17 | 0.681 | +0.75 | -1.85 | -2.00 | -0.0130 | 24 |
| `mask-skip1-alpha0.7-plainpadtoken` | -23.30 | -15.50 | 0.491 | -0.27 | -4.16 | -9.38 | -0.0476 | 24 |
| `mask-skip1-alpha0.7-plaintoken` | +60.83 | -57.44 | 0.030 | -7.41 | -6.87 | -23.15 | -0.1516 | 24 |

## warm − cool（warmth）

| policy | warm | cool | contrast | 95% CI | 有意 | 両側 |
| --- | --- | --- | --- | --- | --- | --- |
| `legacy_exhibit` | +1.87 | -2.50 | +4.38 | [-4.32, +12.94] | - | **両側** |
| `mask-skip1-alpha0.5` | +11.91 | +5.75 | +6.16 | [-2.06, +13.78] | - | - |
| `mask-skip1-alpha0.5-plainpad` | +4.23 | -3.84 | +8.07 | [+1.04, +14.46] | **有意** | **両側** |
| `mask-skip1-alpha0.5-plainpadtoken` | +9.93 | -2.61 | +12.54 | [+7.01, +17.67] | **有意** | **両側** |
| `mask-skip1-alpha0.7-plainpad` | +10.26 | +3.35 | +6.91 | [-6.64, +19.84] | - | - |
| `mask-skip1-alpha0.7-plainpadtoken` | -33.05 | -31.65 | -1.40 | [-18.11, +15.39] | - | - |
| `mask-skip1-alpha0.7-plaintoken` | +1.47 | -2.33 | +3.80 | [-15.68, +23.08] | - | **両側** |
