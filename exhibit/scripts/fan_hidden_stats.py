#!/usr/bin/env python3
"""Do FAN's personalized hidden states come out shorter and flatter than plain?

Attention mixing averages value vectors, so the personalized tokens may end up
with a smaller L2 norm and closer to the sequence mean than the plain ones. Both
would weaken what SDXL's classifier-free guidance has to work with, which is the
washed-out, under-denoised look the masked policies produce. This measures the
norms, the per-token cosines, the collapse toward the sequence mean and the size
of the personalization shift against the conditional/unconditional gap.

It also covers the ``hidden_norm: plain_token`` policy field, which restores the
plain per-token lengths after encoding.

    cd <repo root>
    CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=fan-repro/.work/upstream:exhibit/src \
    fan-repro/.venv/bin/python exhibit/scripts/fan_hidden_stats.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from exhibit.config import CONFIG, REPO
from exhibit.domain import target_prompt
from exhibit.fan_adapter import SDXL_CLIP_L_DIM, encode_conditioning
from exhibit.workers import build_encoder

# Loading the two text encoders on CPU is shared with the mask verification.
from fan_mask_check import load_text_encoders

REFERENCE_SETS = {
    "warm-pair": [
        {"text": "warm color palette, amber tones", "weight": 3.0},
        {"text": "harsh sunlight, hard cast shadow, high contrast", "weight": 3.0},
    ],
    "flat-single": [
        {"text": "flat graphic illustration, solid color fill", "weight": 3.0}
    ],
}
LEGACY_SKIP_PA = [0, 1, 2, 3, 4, 5, 6, 7]


def _policy(*, alpha, skip_pa, use_attn_mask, hidden_norm=None):
    value = {
        "alpha": alpha,
        "skip": -2,
        "skip_pa": list(skip_pa),
        "use_attn_mask": use_attn_mask,
        "pooled_mode": "plain",
        "profiling": {"mode": "all"},
        "reference_unit": "aspect_phrase",
    }
    if hidden_norm is not None:
        value["hidden_norm"] = hidden_norm
    return value


POLICIES = {
    "legacy_exhibit": _policy(alpha=0.5, skip_pa=LEGACY_SKIP_PA, use_attn_mask=False),
    # Controls: same skip_pa as the masked policies, without the mask, so the
    # mask's own contribution can be told apart from personalizing 11 layers.
    "nomask-skip1-alpha0.5": _policy(alpha=0.5, skip_pa=[0], use_attn_mask=False),
    "nomask-skip1-alpha0.7": _policy(alpha=0.7, skip_pa=[0], use_attn_mask=False),
    "mask-skip1-alpha0.5": _policy(alpha=0.5, skip_pa=[0], use_attn_mask=True),
    "mask-skip1-alpha0.7": _policy(alpha=0.7, skip_pa=[0], use_attn_mask=True),
    "mask-skip8-alpha0.5": _policy(
        alpha=0.5, skip_pa=LEGACY_SKIP_PA, use_attn_mask=True
    ),
    "mask-skip1-alpha0.7+plain_token": _policy(
        alpha=0.7, skip_pa=[0], use_attn_mask=True, hidden_norm="plain_token"
    ),
    "mask-skip1-alpha0.5+plain_pad": _policy(
        alpha=0.5, skip_pa=[0], use_attn_mask=True, hidden_norm="plain_pad"
    ),
    "mask-skip1-alpha0.5+plain_pad_token": _policy(
        alpha=0.5, skip_pa=[0], use_attn_mask=True, hidden_norm="plain_pad_token"
    ),
    "mask-skip1-alpha0.7+plain_pad": _policy(
        alpha=0.7, skip_pa=[0], use_attn_mask=True, hidden_norm="plain_pad"
    ),
    "mask-skip1-alpha0.7+plain_pad_token": _policy(
        alpha=0.7, skip_pa=[0], use_attn_mask=True, hidden_norm="plain_pad_token"
    ),
}
VIEWS = {
    "clip_l": slice(None, SDXL_CLIP_L_DIM),
    "big_g": slice(SDXL_CLIP_L_DIM, None),
    "concat": slice(None),
}
DEFAULT_OUT = REPO / "docs/reports/fan-personalization/strength/hidden-stats"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic", default="cat", help="topic id from configs/demo.json"
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="report directory")
    parser.add_argument(
        "--upstream",
        default=str(REPO / CONFIG["fan"]["upstream"]),
        help="materialized FAN upstream tree (contains fan/ and weight/)",
    )
    return parser.parse_args()


def token_groups(encoder, prompt):
    """BOS, content, EOS and pad positions, read off CLIP-L's own tokenizer."""
    counts = {}
    for name, component in encoder._fan_components.items():
        counts[name] = int(component.preprocess([prompt])["attention_mask"].sum())
    length = counts["clip_l"]
    positions = torch.arange(77)
    return {
        "bos": positions == 0,
        "content": (positions >= 1) & (positions < length - 1),
        "eos": positions == length - 1,
        "pad": positions >= length,
    }, counts


def _summary(values):
    return {
        "mean": round(values.mean().item(), 4),
        "min": round(values.min().item(), 4),
        "max": round(values.max().item(), 4),
    }


def _collapse(hidden, keep):
    """How close each token sits to the sequence mean; higher means flatter."""
    tokens = hidden[0][keep]
    centre = tokens.mean(dim=0, keepdim=True)
    return round(
        torch.nn.functional.cosine_similarity(tokens, centre, dim=-1).mean().item(), 4
    )


def view_stats(personalized, plain, uncond, groups, section):
    left = personalized[0][..., section].float()
    right = plain[0][..., section].float()
    gap = uncond[0][..., section].float()
    ratio = left.norm(dim=-1) / right.norm(dim=-1).clamp_min(1e-6)
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=-1)
    shift = (left - right).norm()
    guidance = (right - gap).norm()
    keep = ~groups["pad"]
    stats = {
        "norm_ratio": {
            name: _summary(ratio[mask]) for name, mask in groups.items() if mask.any()
        },
        "cosine_to_plain": {
            name: _summary(cosine[mask]) for name, mask in groups.items() if mask.any()
        },
        "collapse_personalized": _collapse(personalized[..., section], keep),
        "collapse_plain": _collapse(plain[..., section], keep),
        "shift_over_guidance": round((shift / guidance.clamp_min(1e-6)).item(), 4),
        "plain_token_norm_mean": round(right.norm(dim=-1)[keep].mean().item(), 3),
    }
    per_token = (left - right).norm(dim=-1) / (right - gap).norm(dim=-1).clamp_min(1e-6)
    stats["shift_over_guidance_per_token"] = _summary(per_token[keep])
    return stats


def measure(encoder, prompt, negative):
    cache = {}

    def plain(policy_id, text):
        # A reference-free encode depends on the prompt and the skip layer only.
        key = (text, POLICIES[policy_id]["skip"])
        if key not in cache:
            cache[key] = encode_conditioning(encoder, text, None, POLICIES[policy_id])[
                "hidden"
            ]
        return cache[key]

    groups, counts = token_groups(encoder, prompt)
    rows = []
    for policy_id in POLICIES:
        base = plain(policy_id, prompt)
        uncond = plain(policy_id, negative)
        for set_id, refs in REFERENCE_SETS.items():
            hidden = encode_conditioning(encoder, prompt, refs, POLICIES[policy_id])[
                "hidden"
            ]
            rows.append(
                {
                    "policy_id": policy_id,
                    "reference_set": set_id,
                    "views": {
                        name: view_stats(hidden, base, uncond, groups, section)
                        for name, section in VIEWS.items()
                    },
                }
            )
    return rows, counts


def alpha_zero(encoder, prompt, groups):
    """alpha=0 must still land on the plain encoding through every variant."""
    base = encode_conditioning(encoder, prompt, None, POLICIES["mask-skip1-alpha0.5"])[
        "hidden"
    ]
    rows = []
    for mode in ("none", "plain_token", "plain_pad", "plain_pad_token"):
        settings = _policy(
            alpha=0.0,
            skip_pa=[0],
            use_attn_mask=True,
            hidden_norm=None if mode == "none" else mode,
        )
        hidden = encode_conditioning(
            encoder, prompt, REFERENCE_SETS["warm-pair"], settings
        )["hidden"]
        delta = (hidden.float() - base.float()).abs()
        rows.append(
            {
                "hidden_norm": mode,
                "max_abs_diff": round(delta.max().item(), 8),
                "pad_max_abs_diff": round(delta[0][groups["pad"]].max().item(), 8),
                "real_max_abs_diff": round(delta[0][~groups["pad"]].max().item(), 8),
            }
        )
    return rows


def markdown(report):
    counts = report["token_counts"]
    lines = [
        "# FAN 個人化 hidden state の長さと平坦さ",
        "",
        f"対象プロンプト: topic `{report['topic']}`（`configs/demo.json`）。",
        "CPU・fp32・実チェックポイントの text encoder 2 本のみ。",
        f"CLIP-L は実トークン {counts['clip_l']} 個、bigG は {counts['clip_g']} 個。",
        "位置分けは CLIP-L のトークナイザ基準で、content は BOS と EOS を除いた実トークン。",
        "",
        "再現: `exhibit/scripts/fan_hidden_stats.py`（使い方は docstring）。",
        "",
        "## 仮説",
        "",
        "attention は value ベクトルの加重平均なので、個人化した token は plain より",
        "短く、系列平均に寄る（平坦になる）はず。どちらも UNet の CFG の効きを削るので、",
        "mask あり policy の眠い・未収束な絵の説明になる。",
        "",
        "## 参照語句",
        "",
    ]
    for set_id, refs in REFERENCE_SETS.items():
        lines.append(
            f"- `{set_id}`: "
            + "、".join(f"`{ref['text']}`（重み {ref['weight']}）" for ref in refs)
        )
    lines += [
        "",
        "## 指標",
        "",
        "- `norm ratio`: 個人化 token の L2 ノルム ÷ 同じ位置の plain の L2 ノルム。",
        "- `cos`: 同じ位置の plain token との cosine。",
        "- `collapse`: 各 token と系列平均の cosine の平均（pad 除く）。高いほど平坦。",
        "  `個人化 / plain` の順に並べている。",
        "- `shift/CFG`: ‖個人化 − plain‖ ÷ ‖plain − uncond‖（Frobenius, 全 77 位置）。",
        "  uncond は `demo.json` の negative prompt。個人化のずれが CFG の向きと",
        "  どれくらいの大きさで競合するか。",
        "",
        "## 結果",
        "",
        "| policy | 参照 | view | norm ratio content | EOS | pad | cos content | collapse 個人化/plain | shift/CFG |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        for view, stats in row["views"].items():
            ratio = stats["norm_ratio"]
            lines.append(
                f"| `{row['policy_id']}` | {row['reference_set']} | {view} | "
                f"{ratio['content']['mean']} "
                f"[{ratio['content']['min']}, {ratio['content']['max']}] | "
                f"{ratio['eos']['mean']} | {ratio['pad']['mean']} | "
                f"{stats['cosine_to_plain']['content']['mean']} | "
                f"{stats['collapse_personalized']} / {stats['collapse_plain']} | "
                f"{stats['shift_over_guidance']} |"
            )
    warm = {
        row["policy_id"]: row["views"]
        for row in report["rows"]
        if row["reference_set"] == "warm-pair"
    }

    def concat(policy_id, key):
        return warm[policy_id]["concat"][key]

    def ratio(policy_id):
        return concat(policy_id, "norm_ratio")["content"]["mean"]

    lines += [
        "",
        "詳細（BOS 単独、min/max、token ごとの shift/CFG）は `hidden-stats.json` にある。",
        "",
        "## alpha=0 の確認（mask あり skip_pa [0]、参照あり）",
        "",
        "どの変種でも alpha=0 は plain に戻らなければならない。",
        "",
        "| hidden_norm | 全体 最大絶対差 | 実トークン | pad |",
        "|---|---|---|---|",
    ]
    for row in report["alpha_zero"]:
        lines.append(
            f"| `{row['hidden_norm']}` | {row['max_abs_diff']} | "
            f"{row['real_max_abs_diff']} | {row['pad_max_abs_diff']} |"
        )
    lines += [
        "",
        (
            "実トークンに残る値は fp16 の丸め（plain は 1 本、個人化は 4 本まとめて "
            "encode するため）。`plain_pad` 系では pad が完全一致になる。"
        ),
        "",
        "## 読み",
        "",
        (
            "**収縮は mask が作っている。** concat の content の norm ratio は、"
            f"mask なしだと `nomask-skip1-alpha0.5` {ratio('nomask-skip1-alpha0.5')} / "
            f"`nomask-skip1-alpha0.7` {ratio('nomask-skip1-alpha0.7')} で plain より"
            "**伸びる**。参照の pad を key から外した途端に "
            f"{ratio('mask-skip1-alpha0.5')} / {ratio('mask-skip1-alpha0.7')} へ縮む。"
            "skip_pa は倍率を決めるだけで、向きを決めているのは mask のほう"
            f"（`legacy_exhibit` {ratio('legacy_exhibit')} → "
            f"`mask-skip8-alpha0.5` {ratio('mask-skip8-alpha0.5')}）。"
        ),
        (
            "**効いているのはほぼ pad。** mask なしでは pad の norm ratio が "
            f"{concat('nomask-skip1-alpha0.5', 'norm_ratio')['pad']['mean']}〜"
            f"{concat('nomask-skip1-alpha0.7', 'norm_ratio')['pad']['mean']} 倍に膨らむ"
            "（これが pad haze）。mask ありでは逆に "
            f"{concat('mask-skip1-alpha0.5', 'norm_ratio')['pad']['mean']} まで潰れる。"
            "pad は 27 個あって UNet は全部読むので、絵の差はここが大きい。"
        ),
        (
            "**bigG の振れ幅が大きい。** big_g 行は clip_l 行より常に外側にあり、"
            "UNet が受け取る 2048 次元のうち 1280 が bigG 由来。"
        ),
        (
            "**平坦化は起きていない。** collapse は個人化と plain でほぼ同じ。"
            "mask なしでわずかに高く"
            f"（{concat('nomask-skip1-alpha0.5', 'collapse_personalized')} / "
            f"{concat('nomask-skip1-alpha0.5', 'collapse_plain')}）、"
            "mask ありでわずかに低い"
            f"（{concat('mask-skip1-alpha0.5', 'collapse_personalized')} / "
            f"{concat('mask-skip1-alpha0.5', 'collapse_plain')}）。"
            "系列平均に寄る、という仮説の後半は支持されない。"
        ),
        (
            "**ずれの大きさ自体は mask で変わらない。** shift/CFG は同じ alpha なら "
            f"mask なし {concat('nomask-skip1-alpha0.7', 'shift_over_guidance')} と "
            f"mask あり {concat('mask-skip1-alpha0.7', 'shift_over_guidance')} でほぼ同じ。"
            f"`legacy_exhibit` の {concat('legacy_exhibit', 'shift_over_guidance')} に対して"
            "倍以上あり、個人化のずれは cond−uncond の 2/3 ほどの大きさになる。"
            "変わるのは大きさではなく向きと分布。"
        ),
        (
            "**仮説の判定。** 前半（ノルムが縮む）は mask あり policy について支持される。"
            "後半（系列平均への collapse）は支持されない。"
            "mask あり policy の眠い絵は「plain より短い条件ベクトル」と整合し、"
            "mask なし policy が眠く見えないこととも整合する（そちらは plain より長い）。"
        ),
        (
            "**`plain_token` は長さだけを戻す。** norm ratio は 1.0000、"
            "各 encoder 内の向きは不変（clip_l / big_g の cos は mask 版と同値）。"
            "concat の cos だけ動くのは 2 つの encoder を別倍率で伸ばすため。"
            f"shift/CFG は {concat('mask-skip1-alpha0.7', 'shift_over_guidance')} → "
            f"{concat('mask-skip1-alpha0.7+plain_token', 'shift_over_guidance')} に増える。"
        ),
        (
            "**注意**: `plain_token` は pad の長さも plain に戻す。"
            "mask なしの 3〜5 倍の haze には戻らないが、mask で潰していたぶんは戻る。"
            "その潰れが「平板さ」に効いていたなら一緒に消える。画で確かめる話。"
        ),
        (
            "**`plain_pad` は pad を plain に差し替えるだけ。** 実トークンは個人化"
            "したまま伸縮もしないので、norm ratio の content 列は mask 版と同値"
            f"（{ratio('mask-skip1-alpha0.7')} → "
            f"{ratio('mask-skip1-alpha0.7+plain_pad')}）、pad は 1.0。"
            f"shift/CFG は {concat('mask-skip1-alpha0.7', 'shift_over_guidance')} → "
            f"{concat('mask-skip1-alpha0.7+plain_pad', 'shift_over_guidance')} に下がる。"
            "個人化のずれから pad 由来のぶんが抜けるため。"
        ),
        (
            "**`plain_pad_token` は両方。** pad は plain、実トークンは plain と同じ長さ"
            f"で向きだけ個人化。shift/CFG は "
            f"{concat('mask-skip1-alpha0.7+plain_pad_token', 'shift_over_guidance')}。"
            "`plain_token` 単体より小さいのは pad 由来のずれが消えるからで、"
            "条件ベクトルの大きさの分布は plain とほぼ同じになる。"
        ),
        "",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    topic = next(t for t in CONFIG["topics"] if t["id"] == args.topic)
    prompt = target_prompt(topic)
    negative = CONFIG["generation"]["negative_prompt"]
    pipe = load_text_encoders(CONFIG["generation"])
    encoder = build_encoder(pipe, args.upstream)

    rows, counts = measure(encoder, prompt, negative)
    groups, _ = token_groups(encoder, prompt)
    report = {
        "topic": args.topic,
        "prompt": prompt,
        "negative_prompt": negative,
        "device": "cpu",
        "dtype": "float32",
        "fan_commit": CONFIG["fan"]["commit"],
        "reference_sets": REFERENCE_SETS,
        "policies": POLICIES,
        "token_counts": counts,
        "rows": rows,
        "alpha_zero": alpha_zero(encoder, prompt, groups),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hidden-stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (out / "README.md").write_text(markdown(report) + "\n")
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
