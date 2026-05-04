"""Helpers for loading local datasets from inconsistent on-disk layouts."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

from thought_anchors_code.config import canonical_dataset_name, resolve_dataset_dir


def load_local_dataset(dataset_name: str):
    """Load a dataset from either save_to_disk output or HF cache layout."""
    canonical_name = canonical_dataset_name(dataset_name)
    dataset_dir = resolve_dataset_dir(canonical_name)
    payload_path = _resolve_dataset_payload_path(dataset_dir)

    if _is_saved_dataset_dir(payload_path):
        return load_from_disk(str(payload_path))

    if _looks_like_hf_cache_dir(payload_path):
        return load_dataset(
            canonical_name,
            split=_default_split_name(canonical_name),
            cache_dir=str(dataset_dir.parent),
        )

    if payload_path.suffix == ".arrow":
        return Dataset.from_file(str(payload_path))

    raise FileNotFoundError(
        f"Unsupported dataset layout for '{dataset_name}' at {payload_path}."
    )


def _resolve_dataset_payload_path(dataset_dir: Path) -> Path:
    current = dataset_dir
    while current.is_dir():
        entries = [
            entry for entry in current.iterdir() if not entry.name.startswith(".")
        ]
        if _is_saved_dataset_dir(current) or _looks_like_hf_cache_dir(current):
            return current
        if len(entries) != 1 or not entries[0].is_dir():
            break
        current = entries[0]

    arrow_files = sorted(dataset_dir.rglob("*.arrow"))
    if len(arrow_files) == 1:
        return arrow_files[0]
    return dataset_dir


def _is_saved_dataset_dir(path: Path) -> bool:
    return path.is_dir() and (path / "state.json").exists()


def _looks_like_hf_cache_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and any(path.rglob("*.arrow"))
        and any(path.rglob("dataset_info.json"))
    )


def _default_split_name(dataset_name: str) -> str:
    if dataset_name in {"mbpp", "openai_humaneval"}:
        return "test"
    raise ValueError(f"No default split configured for {dataset_name}")
