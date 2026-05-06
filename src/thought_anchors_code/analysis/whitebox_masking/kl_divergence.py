"""KL divergence utilities for attention-suppression causal matrix computation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_log_kl(
    base_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    temperature: float = 0.6,
) -> float:
    """KL(P‖Q) with temperature scaling, then log(KL + 1e-9).

    P comes from base_logits, Q from masked_logits.  Negative KL values
    (numerical noise) are clipped to 0 before taking the log.
    """
    log_p = F.log_softmax(base_logits.float() / temperature, dim=-1)
    log_q = F.log_softmax(masked_logits.float() / temperature, dim=-1)
    p = torch.exp(log_p)

    kl_terms = p * (log_p - log_q)
    kl_terms = torch.where(p == 0, torch.zeros_like(kl_terms), kl_terms)
    kl = kl_terms.sum().item()

    kl = max(kl, 0.0)
    return float(np.log(kl + 1e-9))


def sentence_mean_log_kl(
    base_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    token_start: int,
    token_end: int,
    temperature: float = 0.6,
) -> float:
    """Mean log-KL over the token slice [token_start, token_end)."""
    if token_start >= token_end:
        return float("nan")
    values = [
        compute_log_kl(base_logits[k], masked_logits[k], temperature)
        for k in range(token_start, token_end)
    ]
    return float(np.mean(values))
