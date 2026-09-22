# FAN 個人化 hidden state の長さと平坦さ

対象プロンプト: topic `cat`（`configs/demo.json`）。
CPU・fp32・実チェックポイントの text encoder 2 本のみ。
CLIP-L は実トークン 50 個、bigG は 50 個。
位置分けは CLIP-L のトークナイザ基準で、content は BOS と EOS を除いた実トークン。

再現: `exhibit/scripts/fan_hidden_stats.py`（使い方は docstring）。

## 仮説

attention は value ベクトルの加重平均なので、個人化した token は plain より
短く、系列平均に寄る（平坦になる）はず。どちらも UNet の CFG の効きを削るので、
mask あり policy の眠い・未収束な絵の説明になる。

## 参照語句

- `warm-pair`: `warm color palette, amber tones`（重み 3.0）、`harsh sunlight, hard cast shadow, high contrast`（重み 3.0）
- `flat-single`: `flat graphic illustration, solid color fill`（重み 3.0）

## 指標

- `norm ratio`: 個人化 token の L2 ノルム ÷ 同じ位置の plain の L2 ノルム。
- `cos`: 同じ位置の plain token との cosine。
- `collapse`: 各 token と系列平均の cosine の平均（pad 除く）。高いほど平坦。
  `個人化 / plain` の順に並べている。
- `shift/CFG`: ‖個人化 − plain‖ ÷ ‖plain − uncond‖（Frobenius, 全 77 位置）。
  uncond は `demo.json` の negative prompt。個人化のずれが CFG の向きと
  どれくらいの大きさで競合するか。

## 結果

| policy | 参照 | view | norm ratio content | EOS | pad | cos content | collapse 個人化/plain | shift/CFG |
|---|---|---|---|---|---|---|---|---|
| `legacy_exhibit` | warm-pair | clip_l | 0.9926 [0.8839, 1.0158] | 0.9843 | 0.9987 | 0.9966 | 0.7422 / 0.7393 | 0.075 |
| `legacy_exhibit` | warm-pair | big_g | 0.9988 [0.714, 1.3911] | 0.9763 | 0.9882 | 0.9354 | 0.5894 / 0.5864 | 0.3402 |
| `legacy_exhibit` | warm-pair | concat | 0.9964 [0.8092, 1.3186] | 0.977 | 0.9855 | 0.9491 | 0.6348 / 0.6318 | 0.2908 |
| `legacy_exhibit` | flat-single | clip_l | 0.9926 [0.8837, 1.0288] | 0.9864 | 0.9981 | 0.9944 | 0.7412 / 0.7393 | 0.0892 |
| `legacy_exhibit` | flat-single | big_g | 1.0031 [0.7141, 1.3912] | 0.9794 | 0.99 | 0.9305 | 0.5884 / 0.5864 | 0.3491 |
| `legacy_exhibit` | flat-single | concat | 0.9979 [0.8082, 1.3184] | 0.98 | 0.9873 | 0.9446 | 0.6343 / 0.6318 | 0.2994 |
| `nomask-skip1-alpha0.5` | warm-pair | clip_l | 0.9991 [0.5223, 1.2875] | 1.0484 | 2.3674 | 0.9681 | 0.7637 / 0.7393 | 0.2512 |
| `nomask-skip1-alpha0.5` | warm-pair | big_g | 1.0905 [0.7581, 1.9227] | 1.0657 | 3.5698 | 0.8551 | 0.6211 / 0.5864 | 0.6454 |
| `nomask-skip1-alpha0.5` | warm-pair | concat | 1.0494 [0.7263, 1.6739] | 1.0643 | 3.1285 | 0.8744 | 0.6567 / 0.6318 | 0.5626 |
| `nomask-skip1-alpha0.5` | flat-single | clip_l | 0.9974 [0.5293, 1.2782] | 1.0537 | 2.3225 | 0.9636 | 0.7607 / 0.7393 | 0.2566 |
| `nomask-skip1-alpha0.5` | flat-single | big_g | 1.0922 [0.7501, 1.9421] | 1.0542 | 3.5113 | 0.8526 | 0.6167 / 0.5864 | 0.6472 |
| `nomask-skip1-alpha0.5` | flat-single | concat | 1.0505 [0.6603, 1.6873] | 1.0541 | 3.0729 | 0.871 | 0.6548 / 0.6318 | 0.5648 |
| `nomask-skip1-alpha0.7` | warm-pair | clip_l | 1.0121 [0.483, 1.5556] | 1.1841 | 3.2968 | 0.9432 | 0.7734 / 0.7393 | 0.3338 |
| `nomask-skip1-alpha0.7` | warm-pair | big_g | 1.1771 [0.4323, 3.3336] | 1.0879 | 5.7759 | 0.7952 | 0.647 / 0.5864 | 0.7648 |
| `nomask-skip1-alpha0.7` | warm-pair | concat | 1.0953 [0.5306, 2.6332] | 1.096 | 5.0232 | 0.8217 | 0.6763 / 0.6318 | 0.6715 |
| `nomask-skip1-alpha0.7` | flat-single | clip_l | 1.0126 [0.4759, 1.5388] | 1.1888 | 3.2476 | 0.936 | 0.7681 / 0.7393 | 0.3383 |
| `nomask-skip1-alpha0.7` | flat-single | big_g | 1.1416 [0.294, 3.217] | 1.0409 | 5.6503 | 0.7952 | 0.6304 / 0.5864 | 0.7688 |
| `nomask-skip1-alpha0.7` | flat-single | concat | 1.0835 [0.4355, 2.5519] | 1.0536 | 4.9318 | 0.8198 | 0.666 / 0.6318 | 0.6754 |
| `mask-skip1-alpha0.5` | warm-pair | clip_l | 0.9457 [0.5331, 1.0523] | 1.0367 | 0.7491 | 0.9771 | 0.7378 / 0.7393 | 0.2834 |
| `mask-skip1-alpha0.5` | warm-pair | big_g | 0.9397 [0.4411, 1.2665] | 1.0124 | 0.5577 | 0.8831 | 0.5752 / 0.5864 | 0.6486 |
| `mask-skip1-alpha0.5` | warm-pair | concat | 0.9344 [0.5102, 1.1953] | 1.0144 | 0.5878 | 0.8993 | 0.6245 / 0.6318 | 0.5695 |
| `mask-skip1-alpha0.5` | flat-single | clip_l | 0.9393 [0.5316, 1.0821] | 1.0438 | 0.7508 | 0.9697 | 0.7349 / 0.7393 | 0.3003 |
| `mask-skip1-alpha0.5` | flat-single | big_g | 0.9103 [0.3372, 1.259] | 0.9996 | 0.5435 | 0.8766 | 0.5649 / 0.5864 | 0.6572 |
| `mask-skip1-alpha0.5` | flat-single | concat | 0.9128 [0.4497, 1.1955] | 1.0032 | 0.5764 | 0.891 | 0.6221 / 0.6318 | 0.5789 |
| `mask-skip1-alpha0.7` | warm-pair | clip_l | 0.9301 [0.483, 1.0765] | 1.0431 | 0.9423 | 0.9558 | 0.7383 / 0.7393 | 0.3712 |
| `mask-skip1-alpha0.7` | warm-pair | big_g | 0.8914 [0.2572, 3.3328] | 1.0105 | 0.4622 | 0.8109 | 0.5732 / 0.5864 | 0.7477 |
| `mask-skip1-alpha0.7` | warm-pair | concat | 0.8778 [0.3998, 1.5715] | 1.0132 | 0.5752 | 0.8399 | 0.6338 / 0.6318 | 0.6632 |
| `mask-skip1-alpha0.7` | flat-single | clip_l | 0.9259 [0.4759, 1.1202] | 1.0593 | 0.9765 | 0.9413 | 0.731 / 0.7393 | 0.3874 |
| `mask-skip1-alpha0.7` | flat-single | big_g | 0.7882 [0.095, 1.2717] | 0.9677 | 0.4634 | 0.7777 | 0.5112 / 0.5864 | 0.7693 |
| `mask-skip1-alpha0.7` | flat-single | concat | 0.8216 [0.2491, 1.1728] | 0.9754 | 0.5859 | 0.8146 | 0.6226 / 0.6318 | 0.6832 |
| `mask-skip8-alpha0.5` | warm-pair | clip_l | 0.9936 [0.8857, 1.0169] | 0.9821 | 0.9986 | 0.9965 | 0.7427 / 0.7393 | 0.0752 |
| `mask-skip8-alpha0.5` | warm-pair | big_g | 0.966 [0.6962, 1.3763] | 0.9619 | 0.9646 | 0.9385 | 0.5723 / 0.5864 | 0.4059 |
| `mask-skip8-alpha0.5` | warm-pair | concat | 0.974 [0.7924, 1.3063] | 0.9635 | 0.9704 | 0.9515 | 0.6274 / 0.6318 | 0.346 |
| `mask-skip8-alpha0.5` | flat-single | clip_l | 0.9933 [0.8852, 1.0292] | 0.9844 | 0.998 | 0.9943 | 0.7417 / 0.7393 | 0.0898 |
| `mask-skip8-alpha0.5` | flat-single | big_g | 0.9647 [0.6778, 1.3889] | 0.9728 | 0.9668 | 0.9365 | 0.5698 / 0.5864 | 0.4078 |
| `mask-skip8-alpha0.5` | flat-single | concat | 0.9732 [0.7901, 1.3165] | 0.9738 | 0.9723 | 0.9491 | 0.626 / 0.6318 | 0.3486 |
| `mask-skip1-alpha0.7+plain_token` | warm-pair | clip_l | 1.0 [0.9998, 1.0003] | 1.0001 | 1.0 | 0.9558 | 0.7407 / 0.7393 | 0.3299 |
| `mask-skip1-alpha0.7+plain_token` | warm-pair | big_g | 1.0 [0.9999, 1.0001] | 1.0001 | 1.0 | 0.8109 | 0.5801 / 0.5864 | 0.9331 |
| `mask-skip1-alpha0.7+plain_token` | warm-pair | concat | 1.0 [0.9999, 1.0001] | 1.0001 | 1.0 | 0.8475 | 0.6245 / 0.6318 | 0.8093 |
| `mask-skip1-alpha0.7+plain_token` | flat-single | clip_l | 1.0 [0.9998, 1.0002] | 1.0001 | 1.0 | 0.9413 | 0.7334 / 0.7393 | 0.3529 |
| `mask-skip1-alpha0.7+plain_token` | flat-single | big_g | 1.0 [0.9998, 1.0002] | 1.0 | 1.0 | 0.7777 | 0.5342 / 0.5864 | 0.9807 |
| `mask-skip1-alpha0.7+plain_token` | flat-single | concat | 1.0 [0.9998, 1.0001] | 1.0 | 1.0 | 0.8069 | 0.5972 / 0.6318 | 0.8513 |
| `mask-skip1-alpha0.5+plain_pad` | warm-pair | clip_l | 0.9457 [0.5331, 1.0523] | 1.0367 | 1.0 | 0.9771 | 0.7378 / 0.7393 | 0.1826 |
| `mask-skip1-alpha0.5+plain_pad` | warm-pair | big_g | 0.9397 [0.4411, 1.2665] | 1.0124 | 1.0 | 0.8831 | 0.5752 / 0.5864 | 0.423 |
| `mask-skip1-alpha0.5+plain_pad` | warm-pair | concat | 0.9344 [0.5102, 1.1953] | 1.0144 | 1.0 | 0.8993 | 0.6245 / 0.6318 | 0.3711 |
| `mask-skip1-alpha0.5+plain_pad` | flat-single | clip_l | 0.9393 [0.5316, 1.0821] | 1.0438 | 1.0 | 0.9697 | 0.7349 / 0.7393 | 0.2004 |
| `mask-skip1-alpha0.5+plain_pad` | flat-single | big_g | 0.9103 [0.3372, 1.259] | 0.9996 | 1.0 | 0.8766 | 0.5649 / 0.5864 | 0.4308 |
| `mask-skip1-alpha0.5+plain_pad` | flat-single | concat | 0.9128 [0.4497, 1.1955] | 1.0032 | 1.0 | 0.891 | 0.6221 / 0.6318 | 0.38 |
| `mask-skip1-alpha0.5+plain_pad_token` | warm-pair | clip_l | 1.0 [0.9997, 1.0002] | 0.9999 | 1.0 | 0.9771 | 0.7397 / 0.7393 | 0.1844 |
| `mask-skip1-alpha0.5+plain_pad_token` | warm-pair | big_g | 1.0 [0.9998, 1.0001] | 0.9999 | 1.0 | 0.8831 | 0.5771 / 0.5864 | 0.4319 |
| `mask-skip1-alpha0.5+plain_pad_token` | warm-pair | concat | 1.0 [0.9999, 1.0001] | 0.9999 | 1.0 | 0.902 | 0.626 / 0.6318 | 0.3786 |
| `mask-skip1-alpha0.5+plain_pad_token` | flat-single | clip_l | 1.0 [0.9998, 1.0003] | 1.0001 | 1.0 | 0.9697 | 0.7368 / 0.7393 | 0.2074 |
| `mask-skip1-alpha0.5+plain_pad_token` | flat-single | big_g | 1.0 [0.9999, 1.0002] | 1.0001 | 1.0 | 0.8766 | 0.5684 / 0.5864 | 0.4445 |
| `mask-skip1-alpha0.5+plain_pad_token` | flat-single | concat | 1.0 [0.9999, 1.0001] | 1.0001 | 1.0 | 0.8918 | 0.6206 / 0.6318 | 0.3922 |
| `mask-skip1-alpha0.7+plain_pad` | warm-pair | clip_l | 0.9301 [0.483, 1.0765] | 1.0431 | 1.0 | 0.9558 | 0.7383 / 0.7393 | 0.2412 |
| `mask-skip1-alpha0.7+plain_pad` | warm-pair | big_g | 0.8914 [0.2572, 3.3328] | 1.0105 | 1.0 | 0.8109 | 0.5732 / 0.5864 | 0.4627 |
| `mask-skip1-alpha0.7+plain_pad` | warm-pair | concat | 0.8778 [0.3998, 1.5715] | 1.0132 | 1.0 | 0.8399 | 0.6338 / 0.6318 | 0.4123 |
| `mask-skip1-alpha0.7+plain_pad` | flat-single | clip_l | 0.9259 [0.4759, 1.1202] | 1.0593 | 1.0 | 0.9413 | 0.731 / 0.7393 | 0.2631 |
| `mask-skip1-alpha0.7+plain_pad` | flat-single | big_g | 0.7882 [0.095, 1.2717] | 0.9677 | 1.0 | 0.7777 | 0.5112 / 0.5864 | 0.4953 |
| `mask-skip1-alpha0.7+plain_pad` | flat-single | concat | 0.8216 [0.2491, 1.1728] | 0.9754 | 1.0 | 0.8146 | 0.6226 / 0.6318 | 0.4421 |
| `mask-skip1-alpha0.7+plain_pad_token` | warm-pair | clip_l | 1.0 [0.9998, 1.0003] | 1.0001 | 1.0 | 0.9558 | 0.7407 / 0.7393 | 0.253 |
| `mask-skip1-alpha0.7+plain_pad_token` | warm-pair | big_g | 1.0 [0.9999, 1.0001] | 1.0001 | 1.0 | 0.8109 | 0.5801 / 0.5864 | 0.5121 |
| `mask-skip1-alpha0.7+plain_pad_token` | warm-pair | concat | 1.0 [0.9999, 1.0001] | 1.0001 | 1.0 | 0.8475 | 0.6245 / 0.6318 | 0.454 |
| `mask-skip1-alpha0.7+plain_pad_token` | flat-single | clip_l | 1.0 [0.9998, 1.0002] | 1.0001 | 1.0 | 0.9413 | 0.7334 / 0.7393 | 0.2779 |
| `mask-skip1-alpha0.7+plain_pad_token` | flat-single | big_g | 1.0 [0.9998, 1.0002] | 1.0 | 1.0 | 0.7777 | 0.5342 / 0.5864 | 0.5945 |
| `mask-skip1-alpha0.7+plain_pad_token` | flat-single | concat | 1.0 [0.9998, 1.0001] | 1.0 | 1.0 | 0.8069 | 0.5972 / 0.6318 | 0.5246 |

詳細（BOS 単独、min/max、token ごとの shift/CFG）は `hidden-stats.json` にある。

## alpha=0 の確認（mask あり skip_pa [0]、参照あり）

どの変種でも alpha=0 は plain に戻らなければならない。

| hidden_norm | 全体 最大絶対差 | 実トークン | pad |
|---|---|---|---|
| `none` | 0.00048828 | 0.00048828 | 0.00048828 |
| `plain_token` | 0.00048828 | 0.00048828 | 0.00048828 |
| `plain_pad` | 0.00048828 | 0.00048828 | 0.0 |
| `plain_pad_token` | 0.00048828 | 0.00048828 | 0.0 |

実トークンに残る値は fp16 の丸め（plain は 1 本、個人化は 4 本まとめて encode するため）。`plain_pad` 系では pad が完全一致になる。

## 読み

**収縮は mask が作っている。** concat の content の norm ratio は、mask なしだと `nomask-skip1-alpha0.5` 1.0494 / `nomask-skip1-alpha0.7` 1.0953 で plain より**伸びる**。参照の pad を key から外した途端に 0.9344 / 0.8778 へ縮む。skip_pa は倍率を決めるだけで、向きを決めているのは mask のほう（`legacy_exhibit` 0.9964 → `mask-skip8-alpha0.5` 0.974）。
**効いているのはほぼ pad。** mask なしでは pad の norm ratio が 3.1285〜5.0232 倍に膨らむ（これが pad haze）。mask ありでは逆に 0.5878 まで潰れる。pad は 27 個あって UNet は全部読むので、絵の差はここが大きい。
**bigG の振れ幅が大きい。** big_g 行は clip_l 行より常に外側にあり、UNet が受け取る 2048 次元のうち 1280 が bigG 由来。
**平坦化は起きていない。** collapse は個人化と plain でほぼ同じ。mask なしでわずかに高く（0.6567 / 0.6318）、mask ありでわずかに低い（0.6245 / 0.6318）。系列平均に寄る、という仮説の後半は支持されない。
**ずれの大きさ自体は mask で変わらない。** shift/CFG は同じ alpha なら mask なし 0.6715 と mask あり 0.6632 でほぼ同じ。`legacy_exhibit` の 0.2908 に対して倍以上あり、個人化のずれは cond−uncond の 2/3 ほどの大きさになる。変わるのは大きさではなく向きと分布。
**仮説の判定。** 前半（ノルムが縮む）は mask あり policy について支持される。後半（系列平均への collapse）は支持されない。mask あり policy の眠い絵は「plain より短い条件ベクトル」と整合し、mask なし policy が眠く見えないこととも整合する（そちらは plain より長い）。
**`plain_token` は長さだけを戻す。** norm ratio は 1.0000、各 encoder 内の向きは不変（clip_l / big_g の cos は mask 版と同値）。concat の cos だけ動くのは 2 つの encoder を別倍率で伸ばすため。shift/CFG は 0.6632 → 0.8093 に増える。
**注意**: `plain_token` は pad の長さも plain に戻す。mask なしの 3〜5 倍の haze には戻らないが、mask で潰していたぶんは戻る。その潰れが「平板さ」に効いていたなら一緒に消える。画で確かめる話。
**`plain_pad` は pad を plain に差し替えるだけ。** 実トークンは個人化したまま伸縮もしないので、norm ratio の content 列は mask 版と同値（0.8778 → 0.8778）、pad は 1.0。shift/CFG は 0.6632 → 0.4123 に下がる。個人化のずれから pad 由来のぶんが抜けるため。
**`plain_pad_token` は両方。** pad は plain、実トークンは plain と同じ長さで向きだけ個人化。shift/CFG は 0.454。`plain_token` 単体より小さいのは pad 由来のずれが消えるからで、条件ベクトルの大きさの分布は plain とほぼ同じになる。

