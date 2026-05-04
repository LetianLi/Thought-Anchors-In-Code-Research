"""CLI for collecting code reasoning rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path

from thought_anchors_code.config import (
    DEFAULT_MODEL_ID,
    resolve_dataset_dir,
)
from thought_anchors_code.data_loading import load_local_dataset
from thought_anchors_code.rollouts.collect import (
    collect_rollouts,
    default_rollout_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect code reasoning rollouts for later thought-anchor analysis."
    )
    parser.add_argument("dataset", choices=["humaneval", "mbpp", "openai_humaneval"])
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of dataset rows to sample. Defaults to all rows.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts to generate together for higher GPU throughput.",
    )
    parser.add_argument(
        "--no-token-progress",
        action="store_true",
        help="Disable per-token tqdm updates to reduce Python CPU overhead.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling. Use a negative value to disable seeding.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite the output file instead of skipping existing rollouts.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip executable correctness checks during rollout collection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"[rollouts] Requested dataset={args.dataset} model={args.model} limit={args.limit or 'all'}"
    )
    dataset_dir = resolve_dataset_dir(args.dataset)
    print(f"[rollouts] Resolving dataset from {dataset_dir}")
    dataset = load_local_dataset(args.dataset)
    print(f"[rollouts] Loaded dataset rows={len(dataset)}")
    limit = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    rows = [dataset[index] for index in range(limit)]
    canonical_name = (
        dataset_dir.name if dataset_dir.name != "human_eval" else "openai_humaneval"
    )
    output_path = args.output or default_rollout_path(canonical_name)
    rollouts = collect_rollouts(
        rows=rows,
        model_name_or_path=args.model,
        dataset_name=canonical_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        batch_size=args.batch_size,
        token_progress=not args.no_token_progress,
        seed=args.seed,
        output_path=output_path,
        resume=not args.no_resume,
        evaluate=not args.no_eval,
    )
    correct = sum(1 for rollout in rollouts if rollout.is_correct is True)
    evaluated = sum(1 for rollout in rollouts if rollout.is_correct is not None)
    print(f"Wrote {len(rollouts)} new rollouts to {output_path}")
    if evaluated:
        print(f"Correct this run: {correct}/{evaluated}")


if __name__ == "__main__":
    main()
