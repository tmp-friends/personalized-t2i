#!/usr/bin/env python3
"""Verify the FAN reference-only padding mask on the exhibit's real text encoders.

Upstream keeps only the last of the two additive masks transformers' CLIP
encoder layer passes down (losing the causal mask) and applies the padding mask
to the target prompt's own rows as well as the references'; see
``exhibit.fan_mask`` and
``docs/reports/fan-personalization/strength/e10-mask/README.md``. This script
measures the fix against the encoders the exhibit actually ships, on CPU in
fp32.

    cd <repo root>
    CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=fan-repro/.work/upstream:exhibit/src \
    fan-repro/.venv/bin/python exhibit/scripts/fan_mask_check.py
"""

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from exhibit.config import CONFIG, REPO
from exhibit.domain import target_prompt
from exhibit.fan_adapter import encode_conditioning
from exhibit.fan_mask import install_mask_fix, uninstall_mask_fix
from exhibit.workers import build_encoder

REFS = [
    {"text": "warm color palette, amber tones", "weight": 3.0},
    {"text": "harsh sunlight, hard cast shadow, high contrast", "weight": 3.0},
]
LEGACY_SKIP_PA = [0, 1, 2, 3, 4, 5, 6, 7]
PROBE_POSITIONS = (5, 20, 45)
DEFAULT_OUT = (
    REPO / "docs/reports/fan-personalization/strength/mask-fix"
    if (REPO / "docs").is_dir()
    else Path("docs/reports/fan-personalization/strength/mask-fix")
)


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


def policy(*, alpha, use_attn_mask, skip_pa=(0,)):
    """A strength-experiment policy; only alpha, the mask and skip_pa vary."""
    return {
        "alpha": alpha,
        "skip": -2,
        "skip_pa": list(skip_pa),
        "use_attn_mask": use_attn_mask,
        "pooled_mode": "plain",
        "profiling": {"mode": "all"},
        "reference_unit": "aspect_phrase",
    }


def load_text_encoders(settings):
    """The pinned single-file checkpoint's two text encoders, fp32 on CPU.

    ``exhibit.workers.load_pipeline`` builds the whole SDXL pipeline on the GPU;
    the mask only touches the text encoders, so this loads the same two modules
    from the same checkpoint and the same architecture config, and nothing else.
    """
    from diffusers.loaders.single_file import load_single_file_checkpoint
    from diffusers.loaders.single_file_utils import (
        create_diffusers_clip_model_from_ldm,
    )
    from huggingface_hub import hf_hub_download
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    path = hf_hub_download(
        settings["model"],
        filename=settings["checkpoint"],
        revision=settings["revision"],
        local_files_only=True,
    )
    architecture = settings["pipeline_config"]
    config = str(
        Path(
            hf_hub_download(
                architecture["model"],
                filename="model_index.json",
                revision=architecture["revision"],
                local_files_only=True,
            )
        ).parent
    )
    checkpoint = load_single_file_checkpoint(path, local_files_only=True)
    encoders = {}
    for name, cls in (
        ("text_encoder", CLIPTextModel),
        ("text_encoder_2", CLIPTextModelWithProjection),
    ):
        encoders[name] = create_diffusers_clip_model_from_ldm(
            cls,
            checkpoint,
            subfolder=name,
            config=config,
            torch_dtype=torch.float32,
            local_files_only=True,
        ).eval()
    del checkpoint
    return SimpleNamespace(
        text_encoder=encoders["text_encoder"],
        text_encoder_2=encoders["text_encoder_2"],
        tokenizer=CLIPTokenizer.from_pretrained(
            config, subfolder="tokenizer", local_files_only=True
        ),
        tokenizer_2=CLIPTokenizer.from_pretrained(
            config, subfolder="tokenizer_2", local_files_only=True
        ),
    )


def diff(left, right):
    a, b = left.detach().float(), right.detach().float()
    return round((a - b).abs().max().item(), 8)


def metrics(left, right):
    """The two numbers ``evaluation_worker`` gates a policy on, plus the raw diff."""
    a, b = left.detach().float(), right.detach().float()
    base = torch.sqrt(torch.mean(b * b)).clamp_min(1e-6)
    return {
        "max_abs_diff": diff(a, b),
        "normalized_rmse": round(
            (torch.sqrt(torch.mean((a - b) ** 2)) / base).item(), 8
        ),
        "mean_cosine": round(
            torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item(), 8
        ),
    }


def cosine(left, right):
    a, b = left.detach().float(), right.detach().float()
    return round(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item(), 6)


def encode(encoder, prompt, refs, settings):
    return encode_conditioning(encoder, prompt, refs, settings)


@contextmanager
def upstream_behaviour(encoder):
    """The adapter re-installs the fix by itself, so measuring upstream needs both off."""
    from exhibit import fan_mask

    real = fan_mask.install_mask_fix
    fan_mask.install_mask_fix = lambda *args, **kwargs: None
    uninstall_mask_fix(encoder)
    try:
        yield
    finally:
        fan_mask.install_mask_fix = real
        install_mask_fix(encoder)


def equivalence(encoder, prompt, *, use_attn_mask):
    """alpha=0 must reproduce the plain encoding at all 77 positions."""
    settings = policy(alpha=0.0, use_attn_mask=use_attn_mask)
    plain = encode(encoder, prompt, None, settings)
    with upstream_behaviour(encoder):
        before = encode(encoder, prompt, REFS, settings)
    after = encode(encoder, prompt, REFS, settings)
    return {
        "use_attn_mask": use_attn_mask,
        "limits": {"normalized_rmse": 5e-3, "mean_cosine": 0.999},
        "upstream": {
            "hidden": metrics(before["hidden"], plain["hidden"]),
            "pooled": metrics(before["pooled"], plain["pooled"]),
        },
        "fixed": {
            "hidden": metrics(after["hidden"], plain["hidden"]),
            "pooled": metrics(after["pooled"], plain["pooled"]),
        },
    }


def unpatched_identity(encoder, prompt):
    """Without a padding mask the fix must change nothing at all."""
    settings = policy(alpha=0.5, use_attn_mask=False)
    with upstream_behaviour(encoder):
        before = encode(encoder, prompt, REFS, settings)
    after = encode(encoder, prompt, REFS, settings)
    return {
        "hidden_max_abs_diff": diff(after["hidden"], before["hidden"]),
        "pooled_max_abs_diff": diff(after["pooled"], before["pooled"]),
    }


def _split_positions(encoder, prompt):
    """Which of the 77 CLIP-L positions hold real tokens and which hold pads."""
    mask = encoder._fan_components["clip_l"].preprocess([prompt])["attention_mask"]
    return mask[0].bool(), ~mask[0].bool()


def _position_l2(left, right):
    return (left.detach().float() - right.detach().float()).norm(dim=-1)[0]


def reference_free(encoder, prompt):
    """A reference-free encode must equal the plain pipeline's, pads included.

    The adapter keeps ``use_attn_mask`` off whenever there are no references, so
    the first row is 0 by construction. The second row is what upstream's
    semantics did to the very same encode, and is the drift the
    ``fan_no_reference_vs_pipeline`` diagnostic rejected.
    """
    import torch

    real, pad = _split_positions(encoder, prompt)
    plain = encode(encoder, prompt, None, policy(alpha=0.5, use_attn_mask=False))
    masked = encode(encoder, prompt, None, policy(alpha=0.5, use_attn_mask=True))
    with torch.no_grad():
        hidden, pooled = encoder(
            prompt,
            None,
            skip=-2,
            sample_size=0,
            skip_pa=[0],
            use_attn_mask=True,
        )
    upstream = {"hidden": hidden.to(torch.float16), "pooled": pooled.to(torch.float16)}
    rows = []
    for label, candidate in (("adapter", masked), ("upstream", upstream)):
        distance = _position_l2(candidate["hidden"], plain["hidden"])
        rows.append(
            {
                "encode": label,
                "hidden_max_abs_diff": diff(candidate["hidden"], plain["hidden"]),
                "hidden_mean_cosine": metrics(candidate["hidden"], plain["hidden"])[
                    "mean_cosine"
                ],
                "real_token_max_l2": round(distance[real].max().item(), 6),
                "pad_mean_l2": round(distance[pad].mean().item(), 6),
                "pad_max_l2": round(distance[pad].max().item(), 6),
                "pooled_max_abs_diff": diff(candidate["pooled"], plain["pooled"]),
            }
        )
    return {
        "real_tokens": int(real.sum().item()),
        "pad_positions": int(pad.sum().item()),
        "rows": rows,
    }


def mask_shapes(encoder, prompt):
    """What transformers 4.57 actually hands the patched attention forward."""
    from exhibit import fan_mask

    seen = {}
    real = fan_mask._merge_masks

    def recording(call_kwargs, weight, n_token):
        if not seen:
            causal = call_kwargs.get("causal_attention_mask")
            padding = call_kwargs.get("attention_mask")
            seen.update(
                {
                    "causal_attention_mask": None
                    if causal is None
                    else list(causal.shape),
                    "attention_mask": None if padding is None else list(padding.shape),
                    "weight": list(weight.shape),
                    "n_token": int(n_token),
                    "rows_per_item": fan_mask._rows_per_item(weight, n_token),
                }
            )
        return real(call_kwargs, weight, n_token)

    fan_mask._merge_masks = recording
    try:
        encode(encoder, prompt, REFS, policy(alpha=0.5, use_attn_mask=True))
    finally:
        fan_mask._merge_masks = real
    return seen


def _recompute_weights(call, alpha):
    """Upstream ``personalized_attention``'s own softmax, kept verifiable.

    The function returns ``w @ value`` only, so the weights are rebuilt here and
    checked against the recorded output before any of them is reported.
    """
    query, key, value = call["query"], call["key"], call["value"]
    mask, weight, n_token, scale = (
        call["mask"],
        call["weight"],
        call["n_token"],
        call["scale"],
    )
    b2, h, s = key.shape[0], key.shape[1], query.shape[2]
    s2 = key.shape[2]
    epsilon = torch.finfo(query.dtype).eps
    score = torch.matmul(query, key.transpose(-1, -2)) * scale
    if mask is not None:
        score = score + mask
    weight = weight.unsqueeze(1)
    if weight.dim() != score.dim():
        weight = weight.unsqueeze(2)
    target, ref = torch.split(score, [n_token, max(s2 - n_token, 1)], dim=-1)
    target = torch.nn.functional.softmax(target, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    ws = weight.shape[-1]
    target_weight, ref_weight = torch.split(
        weight, [n_token, max(ws - n_token, 1)], dim=-1
    )
    ref = ref.view(b2, h, s, ws - n_token, -1)
    ref = torch.nn.functional.softmax(ref, dim=-1, dtype=torch.float32).to(query.dtype)
    ref_weight = ref_weight.unsqueeze(-1)
    target = target * target_weight
    ref = (ref * ref_weight).view(b2, h, s, -1)
    target = target * (1 - alpha)
    ref = (ref / (ref.sum(dim=-1, keepdim=True) + epsilon)) * alpha
    weights = torch.cat([target, ref], dim=-1)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + epsilon)
    return weights, diff(torch.matmul(weights, value), call["output"])


def attention_mass(encoder, prompt, *, use_attn_mask, alpha=0.5):
    """Where the personalized query's reference attention actually lands."""
    large = encoder._fan_components["clip_l"]
    fan_model = encoder._fan_model_module
    original = fan_model.personalized_attention
    calls = []

    def recording(query, key, value, mask=None, weight=1, **kwargs):
        out = original(query, key, value, mask=mask, weight=weight, **kwargs)
        if not calls:
            calls.append(
                {
                    "query": query,
                    "key": key,
                    "value": value,
                    "mask": mask,
                    "weight": weight,
                    "n_token": kwargs["n_token"],
                    "scale": kwargs["scale"],
                    "output": out,
                }
            )
        return out

    fan_model.personalized_attention = recording
    try:
        with torch.no_grad():
            large(
                prompt,
                [ref["text"] for ref in REFS],
                weight=[ref["weight"] for ref in REFS],
                alpha=alpha,
                pooling=False,
                skip=-2,
                sample_size=0,
                skip_pa=[0],
                use_attn_mask=use_attn_mask,
                normalize=False,
            )
    finally:
        fan_model.personalized_attention = original

    weights, residual = _recompute_weights(calls[0], alpha)
    # Key layout: the plain target copy, then one 77-token block per reference.
    n_token = calls[0]["n_token"]
    ref_mask = large.preprocess([ref["text"] for ref in REFS])["attention_mask"]
    content = torch.cat(
        [torch.ones(n_token, dtype=torch.bool), ref_mask.bool().view(-1)]
    )
    pad = torch.cat([torch.zeros(n_token, dtype=torch.bool), ~ref_mask.bool().view(-1)])
    per_position = weights[0].mean(dim=0)  # average over heads
    rows = {}
    for position in PROBE_POSITIONS:
        row = per_position[position]
        rows[str(position)] = {
            "target_block": round(row[:n_token].sum().item(), 6),
            "ref_content": round(row[content][n_token:].sum().item(), 6),
            "ref_pad": round(row[pad].sum().item(), 6),
        }
    return {
        "use_attn_mask": use_attn_mask,
        "alpha": alpha,
        "layer": "clip_l layer 1 (first personalized layer with skip_pa=[0])",
        "recomputed_weights_max_abs_diff": residual,
        "reference_tokens": [int(count) for count in ref_mask.sum(dim=-1).tolist()],
        "positions": rows,
    }


def strength(encoder, prompt):
    """How far the personalized hidden states sit from the plain ones."""
    rows = []
    for skip_pa in ([0], LEGACY_SKIP_PA):
        for use_attn_mask in (False, True):
            settings = policy(alpha=0.5, use_attn_mask=use_attn_mask, skip_pa=skip_pa)
            plain = encode(encoder, prompt, None, settings)
            personalized = encode(encoder, prompt, REFS, settings)
            rows.append(
                {
                    "skip_pa": list(skip_pa),
                    "use_attn_mask": use_attn_mask,
                    "cosine_vs_plain": cosine(personalized["hidden"], plain["hidden"]),
                    "hidden_max_abs_diff": diff(
                        personalized["hidden"], plain["hidden"]
                    ),
                }
            )
    return rows


def markdown(report):
    free = report["reference_free"]
    shapes = report["mask_shapes"]
    by_skip = [
        {
            row["use_attn_mask"]: row["cosine_vs_plain"]
            for row in report["strength"]
            if row["skip_pa"] == skip_pa
        }
        for skip_pa in ([0], LEGACY_SKIP_PA)
    ]
    lines = [
        "# FAN attention mask 修正 · 検証",
        "",
        f"対象プロンプト: topic `{report['topic']}`（`configs/demo.json`）",
        "",
        "参照は "
        + "、".join(f"`{ref['text']}`（重み {ref['weight']}）" for ref in REFS)
        + "。CPU・fp32・実チェックポイントの text encoder 2 本のみで計測。",
        "",
        "再現: `exhibit/scripts/fan_mask_check.py`（使い方は docstring）。",
        "",
        "## `use_attn_mask=true` の意味",
        "",
        "**参照側の pad トークンを attention の key から外す、それだけ。**",
        "target プロンプトの encode は素の pipeline と 1 bit も変えません（pad 位置も含む）。",
        "",
        "upstream には 2 つの問題があります。1 つ目は `fan.model.wrapper_forward` が",
        "causal mask と padding mask のうち最後の 1 本しか個人化 attention に渡さないこと。",
        "transformers 4.57 の CLIP は 2 本を別々に attention へ渡し `CLIPAttention` の中で",
        "加算するので（FAN が差し替える層より 1 段下）、mask ありでは causal 構造が消えます。",
        "2 つ目は `FAN.__call__` が target 自身の 2 行にも padding mask を配ること。",
        "target の実トークンは causal のおかげで無傷ですが pad 27 個が動き、",
        "UNet は 77 位置すべてを読むため参照ゼロでも絵が変わります（e10-mask 参照）。",
        "",
        "修正は upstream の wrapper が見る前に padding mask の target 2 行を 0 にし、",
        "causal mask と加算して 1 本の `attention_mask` に畳み込みます",
        "（`exhibit/src/exhibit/fan_mask.py`）。参照なしの encode は",
        "`fan_adapter._encode_once` と `evaluation_worker._raw_official` が",
        "そもそも mask を渡しません。",
        "",
        (
            "transformers 4.57 が実際に渡す形（実測）: "
            f"`causal_attention_mask` {shapes['causal_attention_mask']}、"
            f"`attention_mask` {shapes['attention_mask']}、"
            f"`weight` {shapes['weight']}、`n_token` {shapes['n_token']} で、"
            f"1 件あたり {shapes['rows_per_item']} 行（target 2 + 参照 {len(REFS)}）。"
        ),
        "",
        "## a. 参照なしの encode（素の pipeline と同じであること）",
        "",
        (
            f"CLIP-L は実トークン {free['real_tokens']} 個・pad "
            f"{free['pad_positions']} 個。基準は mask なしの参照なし encode"
            "（e10-mask で pipeline と完全一致を確認済み）。"
        ),
        "",
        "| encode | 最大絶対差 | cosine | 実トークン最大 L2 | pad 平均 L2 | pad 最大 L2 | pooled 最大絶対差 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in free["rows"]:
        label = (
            "修正後（adapter 経由）"
            if row["encode"] == "adapter"
            else "upstream の mask 意味"
        )
        lines.append(
            f"| {label} | {row['hidden_max_abs_diff']} | {row['hidden_mean_cosine']} | "
            f"{row['real_token_max_l2']} | {row['pad_mean_l2']} | {row['pad_max_l2']} | "
            f"{row['pooled_max_abs_diff']} |"
        )
    lines += [
        "",
        "## b. alpha=0 と plain の一致（全 77 位置）",
        "",
        (
            "合否は `evaluation_worker` の PERSONAL_LIMITS（normalized_rmse ≤ 5e-3、"
            "mean_cosine ≥ 0.999）と同じ。出力は fp16。"
        ),
        "",
        "| use_attn_mask | 版 | hidden 最大絶対差 | hidden rmse | hidden cosine | pooled 最大絶対差 |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["equivalence"]:
        for label, key in (("upstream", "upstream"), ("修正後", "fixed")):
            hidden = row[key]["hidden"]
            lines.append(
                f"| {str(row['use_attn_mask']).lower()} | {label} | "
                f"{hidden['max_abs_diff']} | {hidden['normalized_rmse']} | "
                f"{hidden['mean_cosine']} | {row[key]['pooled']['max_abs_diff']} |"
            )
    identity = report["no_mask_identity"]
    lines += [
        "",
        (
            "upstream の行は、素の pipeline と同じ plain を基準にしているので "
            "a の参照なしのずれをそのまま含みます。e10 の診断 "
            "`alpha_zero_vs_no_reference` は mask ありの plain を基準にしていたため "
            "0.99999 で通っていました。"
        ),
        "",
        (
            "`use_attn_mask=false`・alpha=0.5 での修正前後の差: "
            f"hidden {identity['hidden_max_abs_diff']} / pooled "
            f"{identity['pooled_max_abs_diff']}（完全一致）。"
        ),
        "",
        "## c. 参照 pad へ流れる attention（alpha=0.5, skip_pa=[0]）",
        "",
        f"{report['attention'][0]['layer']}。head 平均、target 位置ごとの重み合計。",
        "",
        "| target 位置 | use_attn_mask | target 側 | 参照の実トークン | 参照の pad |",
        "|---|---|---|---|---|",
    ]
    for row in report["attention"]:
        for position in PROBE_POSITIONS:
            cell = row["positions"][str(position)]
            lines.append(
                f"| {position} | {str(row['use_attn_mask']).lower()} | "
                f"{cell['target_block']} | {cell['ref_content']} | {cell['ref_pad']} |"
            )
    lines += [
        "",
        "参照の実トークン数: "
        + " / ".join(str(count) for count in report["attention"][0]["reference_tokens"])
        + "。causal mask があるため、target 位置が参照長より手前なら pad は見えません。",
        "",
        "## d. plain との cosine（alpha=0.5, hidden 2048 次元, token 平均）",
        "",
        "plain は mask の有無によらず同一（a のとおり）なので、この列は個人化の強さそのものです。",
        "",
        "| skip_pa | use_attn_mask | cosine | 最大絶対差 |",
        "|---|---|---|---|",
    ]
    for row in report["strength"]:
        label = "0-7" if row["skip_pa"] == LEGACY_SKIP_PA else str(row["skip_pa"])
        lines.append(
            f"| {label} | {str(row['use_attn_mask']).lower()} | "
            f"{row['cosine_vs_plain']} | {row['hidden_max_abs_diff']} |"
        )
    lines += [
        "",
        "## 留意",
        "",
        (
            "- alpha=0 に残る差は fp16 の丸め。plain は 1 本、個人化は 4 本まとめて "
            "encode するため fp32 の結果がわずかに異なります。mask の有無で同じ値です。"
        ),
        (
            "- 参照側の encode には padding mask が入るので、参照の hidden state は "
            "mask なしの版と変わります。ただし参照の pad は key から外れるため、"
            "個人化 attention が読むのは実トークンだけです。"
        ),
        (
            "- 強度の向きは skip_pa によって変わります（d の表）。"
            f"skip_pa [0] は cosine {by_skip[0][False]} → {by_skip[0][True]} で弱まり、"
            f"skip_pa 0-7 は {by_skip[1][False]} → {by_skip[1][True]} で強まります。"
            "採用するなら alpha は取り直しになります。"
        ),
        "",
        "## 以前の版",
        "",
        "最初の修正は causal mask を戻すだけで、target 自身の pad も mask していました。",
        "そのときの値（この README の前版と e10-mask）:",
        "",
        (
            "- 参照なしの encode が pipeline と一致せず、"
            "`fan_no_reference_vs_pipeline` が cosine 0.718。"
            "実トークンと pooled は一致、pad 27 個だけが平均 L2 24 で動いていた。"
        ),
        "- 参照ゼロの生成画像が平均絶対画素差 40.9/255（topic `cat`）で別物になった。",
        "- plain との cosine は skip_pa [0] で 0.931、skip_pa 0-7 で 0.968。",
        "  いまの値と違うのは、当時の plain 基準が mask ありだったため。",
        "",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    topic = next(t for t in CONFIG["topics"] if t["id"] == args.topic)
    prompt = target_prompt(topic)
    pipe = load_text_encoders(CONFIG["generation"])
    encoder = build_encoder(pipe, args.upstream)
    install_mask_fix(encoder)

    report = {
        "topic": args.topic,
        "prompt": prompt,
        "refs": REFS,
        "device": "cpu",
        "dtype": "float32",
        "fan_commit": CONFIG["fan"]["commit"],
        "semantics": "use_attn_mask excludes reference pad keys only",
        "mask_shapes": mask_shapes(encoder, prompt),
        "reference_free": reference_free(encoder, prompt),
        "equivalence": [
            equivalence(encoder, prompt, use_attn_mask=False),
            equivalence(encoder, prompt, use_attn_mask=True),
        ],
        "no_mask_identity": unpatched_identity(encoder, prompt),
    }
    install_mask_fix(encoder)
    report["attention"] = [
        attention_mass(encoder, prompt, use_attn_mask=False),
        attention_mass(encoder, prompt, use_attn_mask=True),
    ]
    report["strength"] = strength(encoder, prompt)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mask-fix.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (out / "README.md").write_text(markdown(report) + "\n")
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
