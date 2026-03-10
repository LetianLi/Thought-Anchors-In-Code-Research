"""Local model loading utilities."""

import warnings
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def resolve_local_model_path(model_name_or_path: str, data_dir: str | Path = "data") -> Path:
    """Resolve a downloaded local model directory."""
    direct_path = Path(model_name_or_path)
    if direct_path.exists():
        return direct_path

    candidate = Path(data_dir) / "models" / model_name_or_path.split("/")[-1]
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not find local model for '{model_name_or_path}'. "
        f"Checked {direct_path} and {candidate}."
    )


class ModelLoader:
    """Instance-based local model loader with caching support."""

    def __init__(self):
        self._model_cache = {}
        self._tokenizer_cache = {}

    def get_model(
        self,
        model_name_or_path: str = DEFAULT_MODEL_ID,
        data_dir: str | Path = "data",
        float32: bool = True,
        device_map: str = "auto",
        do_flash_attn: bool = False,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        cache_key = (model_name_or_path, str(data_dir), float32, device_map, do_flash_attn)

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model, tokenizer = self._load_model(
            model_name_or_path=model_name_or_path,
            data_dir=data_dir,
            float32=float32,
            device_map=device_map,
            do_flash_attn=do_flash_attn,
        )
        self._model_cache[cache_key] = (model, tokenizer)
        return model, tokenizer

    def _load_model(
        self,
        model_name_or_path: str,
        data_dir: str | Path,
        float32: bool,
        device_map: str,
        do_flash_attn: bool,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        model_path = resolve_local_model_path(model_name_or_path, data_dir)

        warnings.filterwarnings(
            "ignore", message="Sliding Window Attention is enabled but not implemented"
        )
        warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
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

        model_path_str = str(model_path)
        if not any(name in model_path_str for name in ["Llama", "DeepSeek-R1", "gpt-oss"]):
            model_kwargs["sliding_window"] = None

        if float32:
            model_kwargs["torch_dtype"] = torch.float32
        elif "gpt-oss" in model_path_str:
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        return model, tokenizer

    def get_tokenizer(
        self,
        model_name_or_path: str = DEFAULT_MODEL_ID,
        data_dir: str | Path = "data",
    ) -> AutoTokenizer:
        cache_key = (model_name_or_path, str(data_dir))

        if cache_key in self._tokenizer_cache:
            return self._tokenizer_cache[cache_key]

        model_path = resolve_local_model_path(model_name_or_path, data_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

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
    data_dir: str | Path = "data",
    float32: bool = True,
    device_map: str = "auto",
    do_flash_attn: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    return _default_loader.get_model(
        model_name_or_path=model_name_or_path,
        data_dir=data_dir,
        float32=float32,
        device_map=device_map,
        do_flash_attn=do_flash_attn,
    )


def get_tokenizer(
    model_name_or_path: str = DEFAULT_MODEL_ID,
    data_dir: str | Path = "data",
) -> AutoTokenizer:
    return _default_loader.get_tokenizer(
        model_name_or_path=model_name_or_path,
        data_dir=data_dir,
    )
