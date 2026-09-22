# e13-nonsde-skip8 画素指標（metrics.md）

非 SDE sampler（DPM++ 2M・Karras・30 steps）。Δ は同じ topic × seed の plain との差（n=24）。

| policy | Δlum | Δsat | laplacian_ratio | Δblown | Δdarkline | Δedge | Δtarget |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy_exhibit` | -1.74 | +4.42 | 0.865 | -0.11 | -0.64 | -0.98 | -0.0017 |
| `mask-skip8-alpha0.5` | +0.10 | +1.80 | 1.096 | +0.30 | +0.99 | +3.43 | -0.0038 |
| `mask-skip8-alpha0.7` | +1.94 | -5.43 | 0.973 | -0.49 | +0.44 | +3.10 | -0.0054 |

## history ごとの Δwarmth

| policy | warm | cool | t2-flat-graphic | warm-sun |
| --- | --- | --- | --- | --- |
| `legacy_exhibit` | +2.88 | -1.30 | +0.98 | +3.27 |
| `mask-skip8-alpha0.5` | +2.19 | -1.85 | -0.47 | +1.15 |
| `mask-skip8-alpha0.7` | +1.84 | -3.11 | -1.72 | +1.51 |

## warm − cool（warmth）

| policy | warm | cool | contrast | 95% CI | 有意 | 両側 |
| --- | --- | --- | --- | --- | --- | --- |
| `legacy_exhibit` | +2.88 | -1.30 | +4.18 | [-0.34, +8.39] | - | **両側** |
| `mask-skip8-alpha0.5` | +2.19 | -1.85 | +4.04 | [+1.38, +6.38] | **有意** | **両側** |
| `mask-skip8-alpha0.7` | +1.84 | -3.11 | +4.95 | [-1.43, +11.26] | - | **両側** |
