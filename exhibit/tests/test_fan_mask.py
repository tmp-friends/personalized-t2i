"""The FAN reference-only padding mask, on a tiny random CLIP text encoder.

Skipped wherever torch, transformers or the pinned FAN tree is missing, which is
every environment but ``fan-repro/.venv`` with ``fan-repro/.work/upstream`` on
the path. No real weights are loaded: three random layers are enough to show
that the causal structure survives, that the target keeps its plain encoding
including its pads, and that reference pads stop being attended.
"""

import pytest
from exhibit.fan_mask import install_mask_fix, uninstall_mask_fix

N_TOKEN = 16
TARGET_LEN = 9
REF_LENS = (5, 7)
PAD_ID = 0
OTHER_PAD_ID = 41


def _row(torch, length, start, pad_id):
    ids = torch.full((N_TOKEN,), pad_id, dtype=torch.long)
    ids[:length] = torch.arange(start, start + length) % 60 + 1
    mask = torch.zeros(N_TOKEN, dtype=torch.long)
    mask[:length] = 1
    return ids, mask


@pytest.fixture
def fan_encoder():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("fan.model")
    from fan import FAN

    config = transformers.CLIPTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        max_position_embeddings=N_TOKEN,
    )
    torch.manual_seed(0)
    encoder = FAN(transformers.CLIPTextModel(config).eval(), processor=None)
    yield encoder
    uninstall_mask_fix(encoder)


def _prompts(torch, pad_id=PAD_ID, target_pad_id=PAD_ID):
    ids, mask = _row(torch, TARGET_LEN, 3, target_pad_id)
    target = {"input_ids": ids[None], "attention_mask": mask[None]}
    rows = [
        _row(torch, length, 11 * (n + 1), pad_id) for n, length in enumerate(REF_LENS)
    ]
    refs = {
        "input_ids": torch.stack([ids for ids, _ in rows])[None],
        "attention_mask": torch.stack([mask for _, mask in rows])[None],
    }
    return target, refs


def _personalize(encoder, target, refs, *, alpha, use_attn_mask):
    import torch

    with torch.no_grad():
        return encoder(
            target,
            refs,
            weight=[3.0, 3.0],
            alpha=alpha,
            pooling=False,
            skip=-1,
            skip_pa=[],
            use_attn_mask=use_attn_mask,
        )


def test_alpha_zero_matches_the_unmasked_plain_encoding(fan_encoder):
    """The target keeps the plain encoding at every position, pads included."""
    import torch

    target, refs = _prompts(torch)
    with torch.no_grad():
        # No mask: exactly what the plain SDXL pipeline computes.
        plain = fan_encoder.encode_prompt(target, pooling=False, skip=-1)

    uninstall_mask_fix(fan_encoder)
    upstream = _personalize(fan_encoder, target, refs, alpha=0.0, use_attn_mask=True)
    install_mask_fix(fan_encoder)
    fixed = _personalize(fan_encoder, target, refs, alpha=0.0, use_attn_mask=True)

    assert (upstream - plain).abs().max().item() > 1e-3
    assert (fixed - plain).abs().max().item() < 1e-6
    pads = slice(TARGET_LEN, N_TOKEN)
    assert (fixed[:, pads] - plain[:, pads]).abs().max().item() < 1e-6


def test_no_padding_mask_is_untouched(fan_encoder):
    """With a single mask the fix must not change a single bit."""
    import torch

    target, refs = _prompts(torch)
    uninstall_mask_fix(fan_encoder)
    upstream = _personalize(fan_encoder, target, refs, alpha=0.5, use_attn_mask=False)
    install_mask_fix(fan_encoder)
    fixed = _personalize(fan_encoder, target, refs, alpha=0.5, use_attn_mask=False)
    assert torch.equal(fixed, upstream)


def test_reference_pads_get_no_attention(fan_encoder):
    """Changing only the references' pad tokens may not move the output."""
    import torch

    install_mask_fix(fan_encoder)
    target, refs = _prompts(torch)
    _, other_refs = _prompts(torch, pad_id=OTHER_PAD_ID)
    masked = _personalize(fan_encoder, target, refs, alpha=0.5, use_attn_mask=True)
    repadded = _personalize(
        fan_encoder, target, other_refs, alpha=0.5, use_attn_mask=True
    )
    assert torch.equal(masked, repadded)

    unmasked = _personalize(fan_encoder, target, refs, alpha=0.5, use_attn_mask=False)
    unmasked_repadded = _personalize(
        fan_encoder, target, other_refs, alpha=0.5, use_attn_mask=False
    )
    # Without the padding mask the same pads do reach the personalized query.
    assert not torch.equal(unmasked, unmasked_repadded)


def test_target_pads_still_carry_their_own_tokens(fan_encoder):
    """The mask is reference-only, so the target's pads keep affecting the output."""
    import torch

    install_mask_fix(fan_encoder)
    target, refs = _prompts(torch)
    other_target, _ = _prompts(torch, target_pad_id=OTHER_PAD_ID)
    masked = _personalize(fan_encoder, target, refs, alpha=0.5, use_attn_mask=True)
    repadded = _personalize(
        fan_encoder, other_target, refs, alpha=0.5, use_attn_mask=True
    )
    assert not torch.equal(masked, repadded)
