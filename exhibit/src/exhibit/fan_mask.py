"""Reference-only padding masks for FAN's personalized attention.

Two upstream problems, one patch.

**The dropped causal mask.** ``fan.model.wrapper_forward`` rebuilds, for the
personalized query row, one additive mask spanning the concatenated key rows::

    for m_key in mask_key:
        mask = kwargs[m_key].view(batch_size, -1, s, s)[:, 1:]
        kwargs[m_key] = mask.reshape(-1, 1, s, s)
    if mask is not None:
        mask = mask.transpose(1, 2).reshape(batch_size, 1, s, -1)

``mask`` leaves the loop holding only the *last* key. transformers' CLIP encoder
layer hands the attention module ``causal_attention_mask`` and ``attention_mask``
separately and merges them one level below the point FAN patches (4.57's
``CLIPAttention.forward`` does ``attention_mask + causal_attention_mask``), so
with ``use_attn_mask=True`` both arrive and the personalized path keeps the
padding mask alone: the causal structure is silently dropped.

**The masked target.** ``FAN.__call__`` feeds the padding mask to every row of
its batch, including the target prompt's own two rows. The target's real tokens
are unaffected (causal attention never looks forward at its own pads) but its 27
pad positions are, and SDXL's UNet cross-attends to all 77 of them: a
reference-free image changes by ~40/255 mean absolute pixel difference, and the
``fan_no_reference_vs_pipeline`` diagnostic drops to cosine 0.72. See
``docs/reports/fan-personalization/strength/e10-mask/README.md``.

So here ``use_attn_mask=True`` means **exclude the references' pad tokens as
attention keys, and nothing else**. Before upstream's wrapper inspects them this
patch zeroes the padding mask on the two target rows, then folds the causal mask
into it and hands over a single ``attention_mask``. Upstream slices one mask
that already carries both constraints. The duplicate target row therefore gets
causal + 0, bit-identical to what the plain pipeline computes; the personalized
query sees the duplicate's 77 positions exactly as ordinary self-attention would
and the reference blocks with their pad columns removed. With no padding mask
present (``use_attn_mask=False``) nothing is touched at all.

Callers must keep reference-free encodings off the masked path entirely, which
``fan_adapter._encode_once`` and ``evaluation_worker._raw_official`` do by
passing ``use_attn_mask`` only when the encode actually has references.
"""

import importlib

# ``FAN.__call__`` stacks [target, duplicate of target, ref_1..ref_R] per item.
TARGET_ROWS = 2


def _fan_model(encoder=None, module=None):
    if module is not None:
        return module
    found = getattr(encoder, "_fan_model_module", None)
    return found if found is not None else importlib.import_module("fan.model")


def _rows_per_item(weight, n_token):
    """How many stacked rows one prompt occupies, read off FAN's weight tensor.

    Upstream builds it as ``cat([ones(b1, n_token), reference_weights], dim=1)``,
    so its width names the reference count that survived profiling.
    """
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) != 2 or shape[-1] <= n_token:
        return None
    return TARGET_ROWS + int(shape[-1]) - int(n_token)


def _merge_masks(call_kwargs, weight, n_token):
    """Drop the target rows' padding constraint, then fold in the causal mask."""
    import torch

    padding = call_kwargs.get("attention_mask")
    causal = call_kwargs.get("causal_attention_mask")
    if padding is None:
        return
    rows = _rows_per_item(weight, n_token)
    if rows is None:
        raise RuntimeError(
            "FAN passed no per-reference weight tensor, so the target rows of "
            "the padding mask cannot be told apart from the reference rows"
        )
    padding = padding.reshape(-1, rows, *padding.shape[1:]).clone()
    padding[:, :TARGET_ROWS] = 0
    padding = padding.flatten(0, 1)
    if causal is not None:
        # Two ``finfo.min`` entries overflow to ``-inf``; a fully masked row
        # would then soft-max to NaN instead of torch's uniform fallback.
        padding = (causal + padding).clamp_min(torch.finfo(padding.dtype).min)
        call_kwargs["causal_attention_mask"] = None
    call_kwargs["attention_mask"] = padding


def _mask_fixing_wrapper(original):
    # The signature mirrors the pinned upstream one so ``weight`` and
    # ``n_token`` are in hand whichever way FAN passes them.
    def wrapper_forward(
        old_forward, batch_size=1, weight=1, alpha=0.4, n_token=77, memory_batch_size=64
    ):
        patched = original(
            old_forward,
            batch_size=batch_size,
            weight=weight,
            alpha=alpha,
            n_token=n_token,
            memory_batch_size=memory_batch_size,
        )

        def new_forward(self, hidden_states, **call_kwargs):
            _merge_masks(call_kwargs, weight, n_token)
            return patched(self, hidden_states, **call_kwargs)

        return new_forward

    wrapper_forward.exhibit_mask_fix = True
    wrapper_forward.original = original
    return wrapper_forward


def install_mask_fix(encoder=None, *, module=None):
    """Make ``fan.model``'s CLIP attention wrapper mask reference pads only.

    Idempotent, and a no-op for encodings that pass a single mask.
    """
    fan_model = _fan_model(encoder, module)
    if not getattr(fan_model.wrapper_forward, "exhibit_mask_fix", False):
        fan_model.wrapper_forward = _mask_fixing_wrapper(fan_model.wrapper_forward)
    return fan_model


def uninstall_mask_fix(encoder=None, *, module=None):
    """Restore upstream's wrapper; only the verification script and tests need it."""
    fan_model = _fan_model(encoder, module)
    original = getattr(fan_model.wrapper_forward, "original", None)
    if original is not None:
        fan_model.wrapper_forward = original
    return fan_model
