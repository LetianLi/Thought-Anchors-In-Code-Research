"""Attention suppression hook manager for Qwen3.5 hybrid (GatedDeltaNet + softmax) models."""

from __future__ import annotations

from types import MethodType
from typing import Optional

import torch
import torch.nn as nn

from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class QwenAttentionHookManager:
    """Context manager that suppresses attention from a source token range in all softmax-attention layers.

    For Qwen3.5 hybrid models, only `self_attn` (full-attention) layers are patched;
    `linear_attn` (GatedDeltaNet) layers are left untouched.

    Usage::

        with QwenAttentionHookManager(model, [src_start, src_end]):
            outputs = model(**inputs)
    """

    def __init__(self, model: nn.Module, token_range: list):
        self.model = model
        # Normalise to list-of-ranges format
        if token_range and isinstance(token_range[0], int):
            self._ranges: list[list[int]] = [list(token_range)]
        else:
            self._ranges = [list(r) for r in token_range]

        self._originals: dict[str, object] = {}
        self._target_modules: list[tuple[str, nn.Module]] = []

    def __enter__(self):
        self._apply()
        return self

    def __exit__(self, *_):
        self._restore()

    def _apply(self):
        for name, module in self.model.named_modules():
            if not (name.startswith("model.layers") and name.endswith("self_attn")):
                continue
            required = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
            if not all(hasattr(module, attr) for attr in required):
                continue
            new_fwd = MethodType(self._make_masked_forward(), module)
            self._originals[name] = module.forward
            module.forward = new_fwd
            self._target_modules.append((name, module))

    def _restore(self):
        for name, module in self._target_modules:
            if name in self._originals:
                module.forward = self._originals[name]
        self._originals.clear()
        self._target_modules.clear()

    def _make_masked_forward(self):
        """Return a masked forward compatible with Qwen3_5Attention's exact calling convention."""
        ranges = self._ranges

        def masked_forward(
            self_attn,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.shape
            head_dim = self_attn.head_dim          # 256
            num_heads = self_attn.config.num_attention_heads   # 8 (Q heads)
            num_kv_heads = self_attn.config.num_key_value_heads  # 2
            num_kv_groups = self_attn.num_key_value_groups     # 4
            scaling = self_attn.scaling            # 1/sqrt(head_dim)

            # Q projection → split query and gate (Qwen3.5-specific gating)
            q_full = self_attn.q_proj(hidden_states)  # [bsz, q_len, num_heads * head_dim * 2]
            q_full = q_full.view(bsz, q_len, num_heads, head_dim * 2)
            query_states, gate = q_full.chunk(2, dim=-1)     # each [bsz, q_len, num_heads, head_dim]
            gate = gate.reshape(bsz, q_len, -1)              # [bsz, q_len, num_heads * head_dim]

            # Normalise Q and K (Qwen3-style per-head RMSNorm)
            query_states = self_attn.q_norm(query_states).transpose(1, 2)   # [bsz, num_heads, q_len, head_dim]
            key_states = self_attn.k_norm(
                self_attn.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim)
            ).transpose(1, 2)                                                 # [bsz, num_kv_heads, q_len, head_dim]
            value_states = self_attn.v_proj(hidden_states).view(
                bsz, q_len, num_kv_heads, head_dim
            ).transpose(1, 2)                                                 # [bsz, num_kv_heads, q_len, head_dim]

            # RoPE — position_embeddings=(cos, sin) passed in from the decoder layer.
            # Uses Qwen3_5's partial-RoPE apply_rotary_pos_emb (first 64 of 256 dims).
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            kv_seq_len = key_states.shape[-2]

            # GQA expansion
            key_states = _repeat_kv(key_states, num_kv_groups)    # [bsz, num_heads, kv_len, head_dim]
            value_states = _repeat_kv(value_states, num_kv_groups)

            # Attention weights with source suppression
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

            mask_val = torch.finfo(attn_weights.dtype).min
            for rng in ranges:
                eff_end = min(rng[1], kv_seq_len)
                eff_start = min(rng[0], eff_end)
                if eff_start < eff_end:
                    attn_weights[..., eff_start:eff_end] = mask_val

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            attn_output = torch.matmul(attn_weights, value_states)  # [bsz, num_heads, q_len, head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous()  # [bsz, q_len, num_heads, head_dim]
            attn_output = attn_output.reshape(bsz, q_len, -1)       # [bsz, q_len, num_heads * head_dim]

            # Gating (Qwen3.5-specific)
            attn_output = attn_output * torch.sigmoid(gate)

            attn_output = self_attn.o_proj(attn_output)
            return attn_output, None

        return masked_forward


