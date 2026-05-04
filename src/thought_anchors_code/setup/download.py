"""Download models and datasets into the canonical assets layout."""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from huggingface_hub import snapshot_download

from thought_anchors_code.config import (
    DATA_DIR,
    DEFAULT_MODEL_ID,
    HF_CACHE_DIR,
    MODEL_DIR,
    local_model_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download local model and datasets.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID, help="Model id to download."
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip downloading the model and tokenizer.",
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Skip downloading MBPP and HumanEval.",
    )
    return parser.parse_args()


def download_assets(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    skip_model: bool = False,
    skip_datasets: bool = False,
) -> None:
    target_model_dir = local_model_dir(model_id)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if os.getenv("HF_TOKEN"):
        print("HF_TOKEN found in .env")
    else:
        print("HF_TOKEN not found in .env")

    if not skip_model:
        print(f"Downloading {model_id} to {target_model_dir}...")
        target_model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            cache_dir=HF_CACHE_DIR,
            local_dir=target_model_dir,
        )

    if not skip_datasets:
        print("Downloading datasets...")
        mbpp = load_dataset("mbpp", split="test", cache_dir=DATA_DIR)
        mbpp.save_to_disk(DATA_DIR / "mbpp")

        human_eval = load_dataset("openai_humaneval", split="test", cache_dir=DATA_DIR)
        human_eval.save_to_disk(DATA_DIR / "openai_humaneval")

    print("Download complete")


def main() -> None:
    args = parse_args()
    download_assets(
        model_id=args.model,
        skip_model=args.skip_model,
        skip_datasets=args.skip_datasets,
    )


if __name__ == "__main__":
    main()
