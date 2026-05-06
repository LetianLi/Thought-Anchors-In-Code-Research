"""Print the strongest causal (source → target) sentence pairs for inspection.

For each rollout, finds the top-N (i, j) pairs by causal influence score and
prints the actual sentence text so you can read what kinds of sentences are
strongly influencing what.

Usage:
    uv run python show_causal_examples.py --dataset humaneval
    uv run python show_causal_examples.py --dataset humaneval --top 3 --rollouts 5
    uv run python show_causal_examples.py --dataset humaneval --task HumanEval/7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from thought_anchors_code.analysis.whitebox_attention.trace_utils import (
    load_rollouts_jsonl,
    split_reasoning_steps,
)
from thought_anchors_code.config import ROLLOUT_DIR, canonical_dataset_name

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

_DATASET_FILE_PREFIX = {
    "openai_humaneval": "humaneval",
    "mbpp": "mbpp",
}


def _causal_dir(canonical: str) -> Path:
    prefix = _DATASET_FILE_PREFIX.get(canonical, canonical)
    return RESULTS_DIR / f"causal_matrices_{prefix}_qwen3_5_0_8b"


def _rollout_file(canonical: str) -> Path:
    prefix = _DATASET_FILE_PREFIX.get(canonical, canonical)
    return ROLLOUT_DIR / f"{prefix}_qwen3_5_0_8b_full.jsonl"


def _npz_stem(task_id: str, sample_id: int) -> str:
    return task_id.replace("/", "_").replace(" ", "_") + f"_s{sample_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="humaneval")
    parser.add_argument("--top", type=int, default=3, help="top N pairs per rollout")
    parser.add_argument("--rollouts", type=int, default=5, help="how many rollouts to show")
    parser.add_argument("--task", default=None, help="show only this task_id e.g. HumanEval/7")
    parser.add_argument("--min-sentences", type=int, default=6,
                        help="skip rollouts with fewer than this many sentences")
    args = parser.parse_args()

    canonical = canonical_dataset_name(args.dataset)
    causal_dir = _causal_dir(canonical)
    rollout_path = _rollout_file(canonical)

    rollouts = {
        (r.task_id, r.sample_id): r
        for r in load_rollouts_jsonl(rollout_path)
        if r.is_correct is True
    }

    npz_files = sorted(causal_dir.glob("*.npz"))
    shown = 0

    for npz_path in npz_files:
        if shown >= args.rollouts and args.task is None:
            break

        data = np.load(npz_path, allow_pickle=True)
        task_id = str(data["task_id"])
        sample_id = int(data["sample_id"])
        matrix = data["causal_matrix"].astype(np.float32)
        M = int(data["num_sentences"])

        if args.task and task_id != args.task:
            continue
        if M < args.min_sentences:
            continue

        rollout = rollouts.get((task_id, sample_id))
        if rollout is None:
            continue

        sentences = split_reasoning_steps(rollout.reasoning)
        if len(sentences) != M:
            continue

        # Collect all finite (i, j) pairs and sort by score descending
        pairs = []
        for i in range(M):
            for j in range(i + 1, M):
                val = matrix[i, j]
                if np.isfinite(val):
                    pairs.append((val, i, j))

        if not pairs:
            continue

        pairs.sort(reverse=True)
        top_pairs = pairs[: args.top]

        print(f"\n{'═' * 70}")
        print(f"  {task_id}  sample={sample_id}  M={M}")
        print(f"{'═' * 70}")

        for rank, (score, i, j) in enumerate(top_pairs, 1):
            src = sentences[i].strip()
            tgt = sentences[j].strip()
            print(f"\n  #{rank}  score={score:+.3f}  source=sentence[{i}]  target=sentence[{j}]")
            print(f"\n  SOURCE [{i}]:")
            print(f"    {src}")
            print(f"\n  TARGET [{j}]:")
            print(f"    {tgt}")
            print()

        shown += 1


if __name__ == "__main__":
    main()
