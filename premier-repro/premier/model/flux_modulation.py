"""Token-specific preference modulation for FLUX (MM-DiT).

In FLUX every DiT block builds its AdaLN parameters from one modulation vector
    y = M_p(CLIP(p)) + M_t(t) (+ M_g(guidance))            (diffusers: `temb`, [B, 3072])
that is shared by all tokens.  Premier gives every *text token i* its own vector

    y_i^j = y + Δ_shared(e_u, e_{p_i}) + Δ^j_distinct(e_u, e_{p_i})

in block j, while image tokens keep y.  This file re-implements the forward pass of
`FluxTransformerBlock` / `FluxSingleTransformerBlock` / `FluxTransformer2DModel`
(diffusers 0.40) so that the text-token AdaLN can take a per-token embedding
[B, T, 3072].  Weights and the attention processor are reused untouched, so with
`delta_* = None` the result is bit-identical to the stock model.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.utils.checkpoint as ckpt
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel


def _ada_ln_zero_tokens(norm, x: torch.Tensor, emb_tokens: torch.Tensor):
    """AdaLayerNormZero with a per-token conditioning vector. emb_tokens: [B, T, D]"""
    emb = norm.linear(norm.silu(emb_tokens))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
    x = norm.norm(x) * (1 + scale_msa) + shift_msa
    return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


def double_block_forward(block, hidden_states, encoder_hidden_states, temb, temb_txt, image_rotary_emb,
                         joint_attention_kwargs: dict[str, Any] | None = None):
    if temb_txt is None:
        return block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb,
                     image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs)
    joint_attention_kwargs = joint_attention_kwargs or {}
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(hidden_states, emb=temb)
    norm_enc, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = _ada_ln_zero_tokens(
        block.norm1_context, encoder_hidden_states, temb_txt.to(encoder_hidden_states.dtype))

    attention_outputs = block.attn(hidden_states=norm_hidden_states, encoder_hidden_states=norm_enc,
                                   image_rotary_emb=image_rotary_emb, **joint_attention_kwargs)
    attn_output, context_attn_output = attention_outputs[0], attention_outputs[1]

    # image stream (unchanged)
    hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
    norm_hidden_states = block.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * block.ff(norm_hidden_states)
    if len(attention_outputs) == 3:
        hidden_states = hidden_states + attention_outputs[2]

    # text stream with per-token modulation
    encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output
    norm_enc = block.norm2_context(encoder_hidden_states)
    norm_enc = norm_enc * (1 + c_scale_mlp) + c_shift_mlp
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp * block.ff_context(norm_enc)
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    return encoder_hidden_states, hidden_states


def single_block_forward(block, hidden_states, encoder_hidden_states, temb, temb_txt, image_rotary_emb,
                         joint_attention_kwargs: dict[str, Any] | None = None):
    if temb_txt is None:
        return block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb,
                     image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs)
    joint_attention_kwargs = joint_attention_kwargs or {}
    t = encoder_hidden_states.shape[1]
    n_img = hidden_states.shape[1]
    hs = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    residual = hs
    norm = block.norm
    emb_img = norm.linear(norm.silu(temb))                                   # [B, 3D]
    emb_txt = norm.linear(norm.silu(temb_txt.to(hs.dtype)))                  # [B, T, 3D]
    emb = torch.cat([emb_txt, emb_img[:, None].expand(-1, n_img, -1)], dim=1)  # [B, T+N, 3D]
    shift_msa, scale_msa, gate = emb.chunk(3, dim=-1)
    norm_hs = norm.norm(hs) * (1 + scale_msa) + shift_msa
    mlp_hidden_states = block.act_mlp(block.proj_mlp(norm_hs))
    attn_output = block.attn(hidden_states=norm_hs, image_rotary_emb=image_rotary_emb, **joint_attention_kwargs)
    hs = gate * block.proj_out(torch.cat([attn_output, mlp_hidden_states], dim=2))
    hs = residual + hs
    if hs.dtype == torch.float16:
        hs = hs.clip(-65504, 65504)
    return hs[:, :t], hs[:, t:]


def premier_transformer_forward(
    transformer: FluxTransformer2DModel,
    hidden_states: torch.Tensor,            # packed latents [B, N, 64]
    encoder_hidden_states: torch.Tensor,    # T5 embeds [B, T, 4096]
    pooled_projections: torch.Tensor,       # CLIP pooled [B, 768]
    timestep: torch.Tensor,                 # [B] in (0, 1)
    img_ids: torch.Tensor,                  # [N, 3]
    txt_ids: torch.Tensor,                  # [T, 3]
    guidance: torch.Tensor | None = None,   # [B]
    delta_shared: torch.Tensor | None = None,    # [B, T, 3072]
    delta_distinct: torch.Tensor | None = None,  # [B, T, J, 3072]
    block_group: list[int | None] | None = None,  # DiT block j -> group index into delta_distinct (or None)
    joint_attention_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Mirror of FluxTransformer2DModel.forward with token-specific text modulation."""
    tr = transformer
    hidden_states = tr.x_embedder(hidden_states)
    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
        temb = tr.time_text_embed(timestep, guidance, pooled_projections)
    else:
        temb = tr.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = tr.context_embedder(encoder_hidden_states)

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = tr.pos_embed(ids)

    n_double = len(tr.transformer_blocks)
    n_total = n_double + len(tr.single_transformer_blocks)
    if block_group is None:
        block_group = [None] * n_total
    use_ckpt = torch.is_grad_enabled() and getattr(tr, "gradient_checkpointing", False)

    def text_mod(j: int):
        if delta_shared is None and delta_distinct is None:
            return None
        m = temb[:, None, :]
        if delta_shared is not None:
            m = m + delta_shared.to(m.dtype)
        g = block_group[j]
        if delta_distinct is not None and g is not None:
            m = m + delta_distinct[:, :, g].to(m.dtype)
        return m

    for j, block in enumerate(tr.transformer_blocks):
        args = (block, hidden_states, encoder_hidden_states, temb, text_mod(j), image_rotary_emb, joint_attention_kwargs)
        if use_ckpt:
            encoder_hidden_states, hidden_states = ckpt.checkpoint(double_block_forward, *args, use_reentrant=False)
        else:
            encoder_hidden_states, hidden_states = double_block_forward(*args)

    for i, block in enumerate(tr.single_transformer_blocks):
        j = n_double + i
        args = (block, hidden_states, encoder_hidden_states, temb, text_mod(j), image_rotary_emb, joint_attention_kwargs)
        if use_ckpt:
            encoder_hidden_states, hidden_states = ckpt.checkpoint(single_block_forward, *args, use_reentrant=False)
        else:
            encoder_hidden_states, hidden_states = single_block_forward(*args)

    hidden_states = tr.norm_out(hidden_states, temb)
    return tr.proj_out(hidden_states)
