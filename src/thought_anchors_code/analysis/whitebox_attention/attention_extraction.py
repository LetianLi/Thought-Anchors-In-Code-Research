"""Attention extraction helpers for standard Transformers models."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import hashlib
import json

import numpy as np
import torch

from thought_anchors_code.analysis.whitebox_attention.tokenization import (
    average_attention_by_sentence,
    get_sentence_token_boundaries,
)
from thought_anchors_code.engine import get_local_model, get_model_input_device


def compute_attention_tensors(
    text: str,
    model_name_or_path: str,
    float32: bool = False,
    device_map: str = "auto",
) -> tuple[list[np.ndarray], list[int]]:
    model, tokenizer = get_local_model(
        model_name_or_path=model_name_or_path,
        float32=float32,
        device_map=device_map,
    )
    inputs = tokenizer(text, return_tensors="pt")
    input_device = get_model_input_device(model)
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs, output_attentions=True, use_cache=False, return_dict=True
        )

    if outputs.attentions is None:
        raise ValueError(
            "Model did not return attention tensors. Use eager attention implementation."
        )

    token_ids = inputs["input_ids"][0].detach().cpu().tolist()
    attn_layers = [
        layer[0].detach().cpu().numpy().astype(np.float32)
        for layer in outputs.attentions
    ]
    return attn_layers, token_ids


def build_sentence_attention_cache(
    text: str,
    sentences: Sequence[str],
    model_name_or_path: str,
    cache_dir: Path | None = None,
) -> np.ndarray:
    model, tokenizer = get_local_model(
        model_name_or_path=model_name_or_path, float32=False, device_map="auto"
    )
    cache_path = None
    if cache_dir is not None:
        cache_path = (
            cache_dir
            / f"{_attention_cache_key(model_name_or_path, text, sentences)}.npy"
        )
        if cache_path.exists():
            return np.load(cache_path)

    inputs = tokenizer(text, return_tensors="pt")
    input_device = get_model_input_device(model)
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}
    with torch.no_grad():
        outputs = model(
            **inputs, output_attentions=True, use_cache=False, return_dict=True
        )

    boundaries = get_sentence_token_boundaries(text, sentences, tokenizer)
    matrices = []
    for layer_attention in outputs.attentions:
        layer_heads = layer_attention[0].detach().cpu().numpy().astype(np.float32)
        matrices.append(
            [
                average_attention_by_sentence(head_matrix, boundaries)
                for head_matrix in layer_heads
            ]
        )

    stacked = np.asarray(matrices, dtype=np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, stacked)
    return stacked


def _attention_cache_key(
    model_name_or_path: str,
    text: str,
    sentences: Sequence[str],
) -> str:
    payload = json.dumps(
        {
            "model": model_name_or_path,
            "text": text,
            "sentences": list(sentences),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
