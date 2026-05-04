"""Project-wide configuration for data, models, and analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = ASSETS_DIR / "data"
MODEL_DIR = ASSETS_DIR / "model"
HF_CACHE_DIR = ASSETS_DIR / "hf-cache"
CACHE_DIR = ASSETS_DIR / "cache"
ROLLOUT_DIR = ASSETS_DIR / "rollouts"

LEGACY_MODEL_DIR = PROJECT_ROOT / "model"

DATASET_ALIASES = {
    "humaneval": "openai_humaneval",
    "human_eval": "openai_humaneval",
    "openai_humaneval": "openai_humaneval",
    "mbpp": "mbpp",
}

LEGACY_DATASET_DIRS = {
    "openai_humaneval": ("openai_humaneval", "human_eval"),
    "mbpp": ("mbpp",),
}


@dataclass(frozen=True)
class ModelArchitecture:
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int | None = None


def canonical_dataset_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized not in DATASET_ALIASES:
        allowed = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"Unsupported dataset '{name}'. Choose from: {allowed}")
    return DATASET_ALIASES[normalized]


def resolve_dataset_dir(name: str, data_dir: Path = DATA_DIR) -> Path:
    canonical_name = canonical_dataset_name(name)
    for candidate_name in LEGACY_DATASET_DIRS[canonical_name]:
        candidate = data_dir / candidate_name
        if candidate.exists():
            return candidate
    return data_dir / canonical_name


def resolve_local_model_path(model_name_or_path: str = DEFAULT_MODEL_ID) -> Path:
    direct_path = Path(model_name_or_path)
    if _is_model_package(direct_path):
        return direct_path

    candidate = local_model_dir(model_name_or_path)
    if _is_model_package(candidate):
        return candidate

    cached_snapshot = _resolve_hf_snapshot_dir(HF_CACHE_DIR, model_name_or_path)
    if cached_snapshot is not None:
        return cached_snapshot

    if _is_model_package(LEGACY_MODEL_DIR):
        return LEGACY_MODEL_DIR

    raise FileNotFoundError(
        f"Could not find local model for '{model_name_or_path}'. Checked {direct_path}, {MODEL_DIR}, "
        f"{LEGACY_MODEL_DIR}, and {candidate}."
    )


def load_local_model_architecture(model_path: Path | None = None) -> ModelArchitecture:
    resolved = model_path or resolve_local_model_path(DEFAULT_MODEL_ID)
    config_path = resolved / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    return ModelArchitecture(
        num_hidden_layers=int(config["num_hidden_layers"]),
        num_attention_heads=int(config["num_attention_heads"]),
        num_key_value_heads=(
            int(config["num_key_value_heads"])
            if config.get("num_key_value_heads") is not None
            else None
        ),
    )


def ensure_analysis_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ROLLOUT_DIR.mkdir(parents=True, exist_ok=True)


def local_model_dir(model_name_or_path: str = DEFAULT_MODEL_ID) -> Path:
    return MODEL_DIR / model_name_or_path.split("/")[-1]


def _is_model_root(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


def _is_model_package(path: Path) -> bool:
    if not _is_model_root(path):
        return False
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    ]
    return any((path / name).exists() for name in tokenizer_files)


def _resolve_hf_snapshot_dir(base_dir: Path, model_name_or_path: str) -> Path | None:
    repo_dir = base_dir / f"models--{model_name_or_path.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    snapshot_candidates = sorted(
        [path for path in snapshots_dir.iterdir() if _is_model_package(path)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return snapshot_candidates[0] if snapshot_candidates else None
