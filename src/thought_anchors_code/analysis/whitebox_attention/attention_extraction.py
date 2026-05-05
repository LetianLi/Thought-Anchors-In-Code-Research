"""Attention extraction helpers for standard Transformers models."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import hashlib
import json

import numpy as np
import torch

from thought_anchors_code.analysis.whitebox_attention.tokenization import (
    average_attention_heads_by_sentence,
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
        float32=True if _needs_float32_attention(model_name_or_path) else float32,
        device_map=device_map,
    )
    inputs = tokenizer(text, return_tensors="pt")
    input_device = get_model_input_device(model)
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}

    with torch.no_grad():
        outputs = _run_attention_backbone(model, inputs)

    if outputs.attentions is None:
        raise ValueError(
            "Model did not return attention tensors. Use eager attention implementation."
        )

    token_ids = inputs["input_ids"][0].detach().cpu().tolist()
    attn_layers = [
        layer[0].detach().to(torch.float32).cpu().numpy()
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
        model_name_or_path=model_name_or_path,
        float32=True if _needs_float32_attention(model_name_or_path) else False,
        device_map="auto",
    )
    cache_path = get_sentence_attention_cache_path(
        text=text,
        sentences=sentences,
        model_name_or_path=model_name_or_path,
        cache_dir=cache_dir,
    )
    if cache_path is not None:
        if cache_path.exists():
            return np.load(cache_path)

    inputs = tokenizer(text, return_tensors="pt")
    input_device = get_model_input_device(model)
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}
    with torch.no_grad():
        outputs = _run_attention_backbone(model, inputs)

    boundaries = get_sentence_token_boundaries(text, sentences, tokenizer)
    matrices = []
    for layer_attention in outputs.attentions:
        layer_heads = layer_attention[0].detach().to(torch.float32).cpu().numpy()
        matrices.append(average_attention_heads_by_sentence(layer_heads, boundaries))

    stacked = _expand_sparse_attention_layers(
        np.asarray(matrices, dtype=np.float32), model
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, stacked)
    return stacked


def get_sentence_attention_cache_path(
    text: str,
    sentences: Sequence[str],
    model_name_or_path: str,
    cache_dir: Path | None = None,
) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / f"{_attention_cache_key(model_name_or_path, text, sentences)}.npy"


def _needs_float32_attention(model_name_or_path: str) -> bool:
    return "Qwen3.5" in model_name_or_path or "Qwen3_5" in model_name_or_path


def _run_attention_backbone(model, inputs: dict[str, torch.Tensor]):
    backbone = getattr(model, "model", None)
    if backbone is not None:
        return backbone(
            **inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    return model(
        **inputs,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
    )


def _expand_sparse_attention_layers(stacked: np.ndarray, model) -> np.ndarray:
    config = getattr(model, "config", None)
    layer_types = getattr(config, "layer_types", None)
    text_config = getattr(config, "text_config", None)
    if isinstance(text_config, dict):
        layer_types = layer_types or text_config.get("layer_types")
    elif layer_types is None:
        layer_types = getattr(text_config, "layer_types", None)
    if not layer_types:
        return stacked

    full_attention_layers = [
        index for index, layer_type in enumerate(layer_types) if layer_type == "full_attention"
    ]
    if len(full_attention_layers) != stacked.shape[0]:
        return stacked

    expanded_shape = (len(layer_types),) + stacked.shape[1:]
    expanded = np.full(expanded_shape, np.nan, dtype=np.float32)
    for compressed_layer, actual_layer in enumerate(full_attention_layers):
        expanded[actual_layer] = stacked[compressed_layer]
    return expanded


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
            "version": 4,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
