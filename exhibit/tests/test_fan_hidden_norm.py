"""The ``hidden_norm`` rescaling and pad splicing, on real tensors.

Skipped wherever torch is missing, which is the exhibit's own environment; the
policy plumbing around them is covered without torch in ``test_fan_adapter.py``.
"""

from types import SimpleNamespace

import pytest
from exhibit.fan_adapter import (
    SDXL_CLIP_L_DIM,
    _match_plain_token_norms,
    _splice_plain_pads,
)

SEQ = 12
SECTIONS = (slice(None, SDXL_CLIP_L_DIM), slice(SDXL_CLIP_L_DIM, None))


def _pair(torch):
    torch.manual_seed(0)
    plain = torch.randn(1, SEQ, 2048, dtype=torch.float16) * 4
    # Shorter and on a different scale per encoder, like attention mixing leaves it.
    personalized = torch.randn(1, SEQ, 2048, dtype=torch.float16)
    personalized[..., SDXL_CLIP_L_DIM:] *= 0.25
    return personalized, plain


def _encoder(torch, clip_l=7, clip_g=7):
    """Only the two tokenizers' attention masks matter to the splice."""

    def component(length):
        def preprocess(texts):
            mask = torch.zeros(len(texts), SEQ, dtype=torch.long)
            mask[:, :length] = 1
            return {"attention_mask": mask}

        return SimpleNamespace(preprocess=preprocess)

    return SimpleNamespace(
        _fan_components={"clip_l": component(clip_l), "clip_g": component(clip_g)}
    )


def test_every_token_ends_up_with_the_plain_length_in_both_encoders():
    torch = pytest.importorskip("torch")
    personalized, plain = _pair(torch)

    result = _match_plain_token_norms(personalized, plain)

    for section in (slice(None, SDXL_CLIP_L_DIM), slice(SDXL_CLIP_L_DIM, None)):
        assert torch.allclose(
            result[..., section].float().norm(dim=-1),
            plain[..., section].float().norm(dim=-1),
            rtol=3e-3,
        )


def test_only_the_length_changes_and_the_two_encoders_scale_apart():
    """Direction is kept inside each encoder; the 2048-d direction does move."""
    torch = pytest.importorskip("torch")
    personalized, plain = _pair(torch)

    result = _match_plain_token_norms(personalized, plain)

    for section in (slice(None, SDXL_CLIP_L_DIM), slice(SDXL_CLIP_L_DIM, None)):
        cosine = torch.nn.functional.cosine_similarity(
            result[..., section].float(), personalized[..., section].float(), dim=-1
        )
        assert cosine.min().item() > 0.999
    factors = [
        (
            result[..., section].float().norm(dim=-1)
            / personalized[..., section].float().norm(dim=-1)
        ).mean()
        for section in (slice(None, SDXL_CLIP_L_DIM), slice(SDXL_CLIP_L_DIM, None))
    ]
    # One shared factor could not have matched both halves.
    assert factors[1] / factors[0] > 2


def test_a_hidden_state_without_both_encoders_is_refused():
    torch = pytest.importorskip("torch")
    small = torch.randn(1, 3, SDXL_CLIP_L_DIM, dtype=torch.float16)

    with pytest.raises(ValueError, match="hidden_norm"):
        _match_plain_token_norms(small, small)


def test_pad_positions_come_back_bit_for_bit_and_real_tokens_do_not_move():
    torch = pytest.importorskip("torch")
    personalized, plain = _pair(torch)

    result = _splice_plain_pads(personalized, plain, _encoder(torch), "target")

    assert torch.equal(result[:, 7:], plain[:, 7:])
    assert torch.equal(result[:, :7], personalized[:, :7])


def test_each_encoder_splices_at_its_own_pad_boundary():
    torch = pytest.importorskip("torch")
    personalized, plain = _pair(torch)

    result = _splice_plain_pads(
        personalized, plain, _encoder(torch, clip_l=7, clip_g=5), "target"
    )

    for section, length in zip(SECTIONS, (7, 5)):
        assert torch.equal(result[:, length:, section], plain[:, length:, section])
        assert torch.equal(
            result[:, :length, section], personalized[:, :length, section]
        )


def test_an_alpha_zero_encoding_comes_through_the_splice_unchanged():
    torch = pytest.importorskip("torch")
    _, plain = _pair(torch)

    # alpha=0 makes the personalized encoding the plain one; nothing may move.
    result = _splice_plain_pads(plain.clone(), plain, _encoder(torch), "target")

    assert torch.equal(result, plain)


def test_plain_pad_token_keeps_plain_pads_and_plain_lengths_on_the_real_tokens():
    torch = pytest.importorskip("torch")
    personalized, plain = _pair(torch)

    # The order ``encode_conditioning`` applies: rescale everything, then splice.
    result = _splice_plain_pads(
        _match_plain_token_norms(personalized, plain), plain, _encoder(torch), "target"
    )

    assert torch.equal(result[:, 7:], plain[:, 7:])
    for section in SECTIONS:
        assert torch.allclose(
            result[:, :7, section].float().norm(dim=-1),
            plain[:, :7, section].float().norm(dim=-1),
            rtol=3e-3,
        )
        cosine = torch.nn.functional.cosine_similarity(
            result[:, :7, section].float(),
            personalized[:, :7, section].float(),
            dim=-1,
        )
        assert cosine.min().item() > 0.999
