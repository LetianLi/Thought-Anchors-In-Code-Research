"""CLI for computing causal matrices via attention suppression."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from thought_anchors_code.analysis.whitebox_attention.trace_utils import (
    load_rollouts_jsonl,
)
from thought_anchors_code.analysis.whitebox_masking.core import compute_causal_matrix
from thought_anchors_code.config import (
    DEFAULT_MODEL_ID,
    PROJECT_ROOT,
    ROLLOUT_DIR,
    canonical_dataset_name,
)
from thought_anchors_code.engine.model_loader import get_local_model


_DATASET_FILE_PREFIX = {
    "openai_humaneval": "humaneval",
    "mbpp": "mbpp",
}


def _rollout_filename(canonical_dataset: str) -> str:
    prefix = _DATASET_FILE_PREFIX.get(canonical_dataset, canonical_dataset)
    return f"{prefix}_qwen3_5_0_8b_full.jsonl"


def _output_dir(canonical_dataset: str) -> Path:
    prefix = _DATASET_FILE_PREFIX.get(canonical_dataset, canonical_dataset)
    model_slug = DEFAULT_MODEL_ID.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    return PROJECT_ROOT / "results" / f"causal_matrices_{prefix}_{model_slug}"


def _npz_path(out_dir: Path, task_id: str, sample_id: int) -> Path:
    task_slug = task_id.replace("/", "_").replace(" ", "_")
    return out_dir / f"{task_slug}_s{sample_id}.npz"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute sentence-level causal matrices.")
    parser.add_argument("--dataset", required=True, help="humaneval or mbpp")
    parser.add_argument("--max-rollouts", type=int, default=None)
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--resume", action="store_true", help="skip already-computed rollouts")
    parser.add_argument(
        "--correct-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="only process is_correct==True rollouts (default: True)",
    )
    parser.add_argument("--rollout-file", default=None, help="override rollout JSONL path")
    args = parser.parse_args(argv)

    dataset = canonical_dataset_name(args.dataset)
    rollout_path = Path(args.rollout_file) if args.rollout_file else ROLLOUT_DIR / _rollout_filename(dataset)

    if not rollout_path.exists():
        print(f"Rollout file not found: {rollout_path}", file=sys.stderr)
        sys.exit(1)

    rollouts = load_rollouts_jsonl(rollout_path)
    if args.correct_only:
        rollouts = [r for r in rollouts if r.is_correct is True]
        print(f"Filtered to {len(rollouts)} correct rollouts.")

    if args.max_rollouts is not None:
        rollouts = rollouts[: args.max_rollouts]

    out_dir = _output_dir(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        done = {p.stem for p in out_dir.glob("*.npz")}
        before = len(rollouts)
        rollouts = [
            r for r in rollouts
            if _npz_path(out_dir, r.task_id, r.sample_id).stem not in done
        ]
        print(f"Resume: skipping {before - len(rollouts)} already-computed rollouts.")

    print(f"Loading model {DEFAULT_MODEL_ID} …")
    model, tokenizer = get_local_model(
        DEFAULT_MODEL_ID,
        float32=True,
        device_map=args.device,
        do_flash_attn=False,
    )

    errors = 0
    for idx, rollout in enumerate(rollouts, 1):
        out_path = _npz_path(out_dir, rollout.task_id, rollout.sample_id)
        print(f"[{idx}/{len(rollouts)}] {rollout.task_id} s{rollout.sample_id} …", end=" ", flush=True)
        try:
            matrix, sentences, _ = compute_causal_matrix(rollout, model, tokenizer)
            M = len(sentences)
            np.savez_compressed(
                out_path,
                causal_matrix=matrix,
                sentence_indices=np.arange(M, dtype=np.int32),
                task_id=rollout.task_id,
                sample_id=rollout.sample_id,
                dataset_name=rollout.dataset_name,
                num_sentences=M,
            )
            print(f"done (M={M})")
        except Exception as exc:
            print(f"ERROR: {exc}")
            errors += 1

    print(f"\nFinished. {len(rollouts) - errors}/{len(rollouts)} succeeded. Output: {out_dir}")


if __name__ == "__main__":
    main()
