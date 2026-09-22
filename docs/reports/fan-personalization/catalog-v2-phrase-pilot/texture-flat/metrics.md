# 画素指標（metrics.md）

指標は 256x320 に縮小してから測る。`warmth` = mean(R−B)、`lum` = 平均輝度、`sat` = HSV の平均 S、`contrast` = 輝度の標準偏差、`edge` = グレースケールの FIND_EDGES が 40 を超える画素の割合（%）。

`darkline` だけは原寸で測る。自分自身の GaussianBlur(2.0) より 18 以上暗い画素の割合（%）で、細い暗い線だけを数える。`edge` は色の境目も数えるので、べた塗りでも色面の縁で高く出る。線の量を見るのはこちらが適切。

## 候補と token 数（CLIP-L / bigG は同数）

| history | 句 | tokens |
| --- | --- | --- |
| `t2-current` | `flat color, lineless, no lineart` | 9 |
| `t2-cel-contrast` | `anime screencap, cel shading` | 6 |
| `t2-flat-smooth` | `flat color, flat shading, smooth edges` | 8 |
| `t2-flat-minimal` | `flat color illustration, minimal shading` | 6 |
| `t2-flat-vector` | `flat color, vector art, smooth shapes` | 8 |
| `t2-flat-borderless` | `flat colors, borderless, soft edges` | 8 |
| `t2-flat-matte` | `flat color, matte, simple shapes` | 7 |
| `t2-flat-fill` | `flat color, solid color fill, smooth edges` | 9 |
| `t2-flat-graphic` | `flat graphic illustration, solid color fill` | 7 |
| `warm-sun` | `warm color palette, amber tones` + `harsh sunlight, hard cast shadow, high contrast` | 6 + 9 |

## Part 1 — FAN 参照。plain との差（topic 3 × seed 2 の平均）

plain の絶対値: edge 24.715 / darkline 6.87 / warmth 21.845 / lum 126.554 / sat 93.852 / contrast 75.698

### Δedge

| history | legacy_exhibit | strong_v1 | skip1-all-alpha0.5 |
| --- | --- | --- | --- |
| `t2-current` | -0.696 | -4.481 | -4.481 |
| `t2-cel-contrast` | +0.206 | -5.606 | -5.606 |
| `t2-flat-smooth` | -0.222 | -3.069 | -3.069 |
| `t2-flat-minimal` | +0.633 | -4.955 | -4.955 |
| `t2-flat-vector` | +0.340 | -3.558 | -3.558 |
| `t2-flat-borderless` | +0.245 | -5.615 | -5.615 |
| `t2-flat-matte` | +0.706 | -5.677 | -5.677 |
| `t2-flat-fill` | -0.280 | -5.892 | -5.892 |
| `t2-flat-graphic` | +1.756 | -5.738 | -5.738 |
| `warm-sun` | +0.253 | -4.809 | -5.448 |

### Δdarkline

| history | legacy_exhibit | strong_v1 | skip1-all-alpha0.5 |
| --- | --- | --- | --- |
| `t2-current` | -1.397 | -4.915 | -4.915 |
| `t2-cel-contrast` | -0.693 | -4.802 | -4.802 |
| `t2-flat-smooth` | -1.047 | -4.229 | -4.229 |
| `t2-flat-minimal` | -0.685 | -4.482 | -4.482 |
| `t2-flat-vector` | -0.854 | -3.918 | -3.918 |
| `t2-flat-borderless` | -1.622 | -5.195 | -5.195 |
| `t2-flat-matte` | -1.259 | -5.132 | -5.132 |
| `t2-flat-fill` | -1.082 | -5.553 | -5.553 |
| `t2-flat-graphic` | -0.859 | -5.226 | -5.226 |
| `warm-sun` | -1.334 | -4.825 | -4.939 |

### Δwarmth

| history | legacy_exhibit | strong_v1 | skip1-all-alpha0.5 |
| --- | --- | --- | --- |
| `t2-current` | -0.277 | +0.349 | +0.349 |
| `t2-cel-contrast` | -0.052 | -1.397 | -1.397 |
| `t2-flat-smooth` | -1.502 | +0.380 | +0.380 |
| `t2-flat-minimal` | +2.492 | +1.107 | +1.107 |
| `t2-flat-vector` | +0.790 | +0.300 | +0.300 |
| `t2-flat-borderless` | +0.252 | -0.250 | -0.250 |
| `t2-flat-matte` | +0.190 | -1.174 | -1.174 |
| `t2-flat-fill` | +0.007 | +0.958 | +0.958 |
| `t2-flat-graphic` | -1.407 | +0.963 | +0.963 |
| `warm-sun` | -2.003 | +0.216 | +1.317 |

### Δlum

| history | legacy_exhibit | strong_v1 | skip1-all-alpha0.5 |
| --- | --- | --- | --- |
| `t2-current` | -11.408 | +12.800 | +12.800 |
| `t2-cel-contrast` | +0.699 | +12.809 | +12.809 |
| `t2-flat-smooth` | -7.095 | +13.852 | +13.852 |
| `t2-flat-minimal` | -3.411 | +12.732 | +12.732 |
| `t2-flat-vector` | -11.445 | +11.394 | +11.394 |
| `t2-flat-borderless` | -10.864 | +13.166 | +13.166 |
| `t2-flat-matte` | -11.103 | +13.527 | +13.527 |
| `t2-flat-fill` | -5.986 | +16.816 | +16.816 |
| `t2-flat-graphic` | -13.064 | +11.556 | +11.556 |
| `warm-sun` | -8.102 | +13.808 | +16.679 |

### Δsat

| history | legacy_exhibit | strong_v1 | skip1-all-alpha0.5 |
| --- | --- | --- | --- |
| `t2-current` | +3.611 | -49.020 | -49.020 |
| `t2-cel-contrast` | -1.908 | -39.347 | -39.347 |
| `t2-flat-smooth` | +1.693 | -44.721 | -44.721 |
| `t2-flat-minimal` | +0.672 | -38.125 | -38.125 |
| `t2-flat-vector` | +8.263 | -46.443 | -46.443 |
| `t2-flat-borderless` | +3.933 | -43.726 | -43.726 |
| `t2-flat-matte` | +2.792 | -48.550 | -48.550 |
| `t2-flat-fill` | +1.509 | -47.891 | -47.891 |
| `t2-flat-graphic` | +5.442 | -42.111 | -42.111 |
| `warm-sun` | -4.190 | -44.721 | -41.485 |

### Δcontrast

| history | legacy_exhibit | strong_v1 | skip1-all-alpha0.5 |
| --- | --- | --- | --- |
| `t2-current` | +0.540 | -14.510 | -14.510 |
| `t2-cel-contrast` | +2.894 | -8.094 | -8.094 |
| `t2-flat-smooth` | +2.271 | -11.435 | -11.435 |
| `t2-flat-minimal` | +2.599 | -8.106 | -8.106 |
| `t2-flat-vector` | +3.356 | -9.120 | -9.120 |
| `t2-flat-borderless` | +0.605 | -12.233 | -12.233 |
| `t2-flat-matte` | +2.230 | -11.501 | -11.501 |
| `t2-flat-fill` | +1.321 | -13.556 | -13.556 |
| `t2-flat-graphic` | +3.324 | -10.324 | -10.324 |
| `warm-sun` | +2.455 | -9.607 | -10.221 |

## Part 2 — カード側。8枚（girl / student × texture 2 の4プロファイル）

| 候補 | edge | Δedge | darkline | Δdarkline | lum | sat |
| --- | --- | --- | --- | --- | --- | --- |
| `t2-current` | 15.77 | +0.00 | 3.893 | +0.000 | 127.9 | 71.5 |
| `t2-flat-smooth` | 17.11 | +1.35 | 4.712 | +0.819 | 136.2 | 72.6 |
| `t2-flat-minimal` | 17.38 | +1.62 | 4.924 | +1.031 | 132.2 | 77.6 |
| `t2-flat-vector` | 15.25 | -0.52 | 3.963 | +0.070 | 126.1 | 78.5 |
| `t2-flat-borderless` | 17.10 | +1.33 | 4.330 | +0.437 | 133.2 | 72.2 |
| `t2-flat-matte` | 16.20 | +0.43 | 4.082 | +0.189 | 133.6 | 74.3 |
| `t2-flat-fill` | 17.82 | +2.06 | 4.845 | +0.952 | 138.7 | 72.5 |
| `t2-flat-graphic` | 15.80 | +0.03 | 3.618 | -0.275 | 131.3 | 74.0 |

### カードごとの darkline

| 候補 | g:c0-l2-t2-m0 | g:c1-l0-t2-m3 | g:c2-l2-t2-m1 | g:c3-l1-t2-m2 | s:c0-l2-t2-m0 | s:c1-l0-t2-m3 | s:c2-l2-t2-m1 | s:c3-l1-t2-m2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `t2-current` | 3.09 | 2.71 | 1.39 | 4.63 | 3.65 | 6.62 | 5.92 | 3.12 |
| `t2-flat-smooth` | 3.89 | 4.91 | 2.27 | 5.68 | 4.55 | 5.54 | 6.29 | 4.55 |
| `t2-flat-minimal` | 4.54 | 6.48 | 3.83 | 4.52 | 4.22 | 5.15 | 5.60 | 5.04 |
| `t2-flat-vector` | 3.84 | 3.41 | 1.40 | 4.57 | 4.17 | 5.42 | 4.77 | 4.13 |
| `t2-flat-borderless` | 3.13 | 4.00 | 2.15 | 4.42 | 4.83 | 6.00 | 4.42 | 5.69 |
| `t2-flat-matte` | 2.67 | 4.70 | 2.26 | 4.77 | 3.77 | 4.48 | 5.58 | 4.43 |
| `t2-flat-fill` | 3.43 | 5.65 | 2.06 | 5.71 | 5.30 | 5.54 | 6.04 | 5.04 |
| `t2-flat-graphic` | 1.34 | 5.64 | 2.39 | 3.74 | 2.85 | 5.22 | 3.83 | 3.94 |
