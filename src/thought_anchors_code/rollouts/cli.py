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
        "--limit", type=int, default=10, help="Number of dataset rows to sample."
    )
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"[rollouts] Requested dataset={args.dataset} model={args.model} limit={args.limit}"
    )
    dataset_dir = resolve_dataset_dir(args.dataset)
    print(f"[rollouts] Resolving dataset from {dataset_dir}")
    dataset = load_local_dataset(args.dataset)
    print(f"[rollouts] Loaded dataset rows={len(dataset)}")
    rows = [dataset[index] for index in range(min(args.limit, len(dataset)))]
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
        output_path=output_path,
    )
    print(f"Wrote {len(rollouts)} rollouts to {output_path}")


if __name__ == "__main__":
    main()
