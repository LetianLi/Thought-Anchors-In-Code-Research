"""Local model loading utilities."""

from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from thought_anchors_code.config import (
    DEFAULT_MODEL_ID,
    HF_CACHE_DIR,
    resolve_local_model_path,
)


class ModelLoader:
    """Instance-based local model loader with caching support."""

    def __init__(self):
        self._model_cache = {}
        self._tokenizer_cache = {}

    def get_model(
        self,
        model_name_or_path: str = DEFAULT_MODEL_ID,
        float32: bool = True,
        device_map: str = "auto",
        do_flash_attn: bool = False,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        cache_key = (model_name_or_path, float32, device_map, do_flash_attn)

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model, tokenizer = self._load_model(
            model_name_or_path=model_name_or_path,
            float32=float32,
            device_map=device_map,
            do_flash_attn=do_flash_attn,
        )
        self._model_cache[cache_key] = (model, tokenizer)
        return model, tokenizer

    def _load_model(
        self,
        model_name_or_path: str,
        float32: bool,
        device_map: str,
        do_flash_attn: bool,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        try:
            model_source = resolve_local_model_path(model_name_or_path)
        except FileNotFoundError:
            model_source = model_name_or_path

        try:
            return self._load_model_from_source(
                model_source=model_source,
                float32=float32,
                device_map=device_map,
                do_flash_attn=do_flash_attn,
            )
        except FileNotFoundError:
            if isinstance(model_source, str):
                raise
            return self._load_model_from_source(
                model_source=model_name_or_path,
                float32=float32,
                device_map=device_map,
                do_flash_attn=do_flash_attn,
            )

    def _load_model_from_source(
        self,
        model_source: str | object,
        float32: bool,
        device_map: str,
        do_flash_attn: bool,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(model_source, cache_dir=HF_CACHE_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "device_map": device_map,
            "force_download": False,
        }

        if do_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            model_kwargs["attn_implementation"] = "eager"

        model_path_str = str(model_source)

        if float32:
            model_kwargs["dtype"] = torch.float32
        elif "gpt-oss" in model_path_str:
            model_kwargs["dtype"] = torch.bfloat16
        else:
            model_kwargs["dtype"] = torch.float16

        model_kwargs["cache_dir"] = HF_CACHE_DIR
        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
        return model, tokenizer

    def get_tokenizer(
        self,
        model_name_or_path: str = DEFAULT_MODEL_ID,
    ) -> AutoTokenizer:
        cache_key = model_name_or_path

        if cache_key in self._tokenizer_cache:
            return self._tokenizer_cache[cache_key]

        try:
            model_source = resolve_local_model_path(model_name_or_path)
        except FileNotFoundError:
            model_source = model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_source, cache_dir=HF_CACHE_DIR)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._tokenizer_cache[cache_key] = tokenizer
        return tokenizer

    def clear_cache(self):
        self._model_cache.clear()
        self._tokenizer_cache.clear()


_default_loader = ModelLoader()


def get_local_model(
    model_name_or_path: str = DEFAULT_MODEL_ID,
    float32: bool = True,
    device_map: str = "auto",
    do_flash_attn: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    return _default_loader.get_model(
        model_name_or_path=model_name_or_path,
        float32=float32,
        device_map=device_map,
        do_flash_attn=do_flash_attn,
    )


def get_tokenizer(
    model_name_or_path: str = DEFAULT_MODEL_ID,
) -> AutoTokenizer:
    return _default_loader.get_tokenizer(
        model_name_or_path=model_name_or_path,
    )


def get_model_input_device(model: AutoModelForCausalLM) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for placement in hf_device_map.values():
            if placement in {"disk", "cpu"}:
                continue
            return torch.device(placement)

    try:
        return next(model.parameters()).device
    except StopIteration:
        pass

    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)

    raise ValueError("Could not determine model input device.")
