from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from datasets import load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_DATA_DIR = Path("assets/data")
DEFAULT_TOKENIZER_PATH = Path("assets/model")
DATASET_ALIASES = {
	"humaneval": "openai_humaneval",
	"human_eval": "openai_humaneval",
	"openai_humaneval": "openai_humaneval",
	"mbpp": "mbpp",
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Analyze basic stats for the downloaded Humaneval and MBPP datasets."
	)
	parser.add_argument(
		"datasets",
		nargs="*",
		default=["humaneval", "mbpp"],
		help="Datasets to analyze: humaneval, mbpp, or openai_humaneval.",
	)
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=DEFAULT_DATA_DIR,
		help="Directory containing datasets saved with datasets.save_to_disk().",
	)
	parser.add_argument(
		"--tokenizer-path",
		type=Path,
		default=DEFAULT_TOKENIZER_PATH,
		help="Local tokenizer path used to compute token counts.",
	)
	return parser.parse_args()


def canonical_dataset_name(name: str) -> str:
	normalized = name.strip().lower()
	if normalized not in DATASET_ALIASES:
		allowed = ", ".join(sorted(DATASET_ALIASES))
		raise SystemExit(f"Unsupported dataset '{name}'. Choose from: {allowed}")
	return DATASET_ALIASES[normalized]


def non_empty_line_count(text: str) -> int:
	return sum(1 for line in text.splitlines() if line.strip())


def summarize_numeric(values: Iterable[int | float]) -> dict[str, float]:
	sequence = list(values)
	if not sequence:
		return {"min": 0, "median": 0, "mean": 0, "max": 0, "total": 0}

	return {
		"min": min(sequence),
		"median": median(sequence),
		"mean": mean(sequence),
		"max": max(sequence),
		"total": sum(sequence),
	}


def batched_token_counts(
	tokenizer: PreTrainedTokenizerBase, texts: Iterable[str], batch_size: int = 64
) -> list[int]:
	text_list = list(texts)
	counts: list[int] = []
	for start in range(0, len(text_list), batch_size):
		batch = text_list[start : start + batch_size]
		encoded = tokenizer(batch, add_special_tokens=False, truncation=False)
		counts.extend(len(input_ids) for input_ids in encoded["input_ids"])
	return counts


def format_value(value: Any) -> str:
	if isinstance(value, float) and math.isfinite(value):
		if value.is_integer():
			return str(int(value))
		return f"{value:.2f}"
	return str(value)


def render_kv_section(title: str, rows: list[tuple[str, Any]]) -> list[str]:
	if not rows:
		return []
	label_width = max(len(label) for label, _ in rows)
	lines = [title]
	for label, value in rows:
		lines.append(f"  {label:<{label_width}}  {format_value(value)}")
	return lines


def render_distribution_section(title: str, metric_rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
	if not metric_rows:
		return []
	header = f"  {'metric':<24} {'min':>8} {'median':>8} {'mean':>8} {'max':>8} {'total':>10}"
	lines = [title, header, f"  {'-' * (len(header) - 2)}"]
	for label, stats in metric_rows:
		lines.append(
			"  "
			f"{label:<24} "
			f"{format_value(stats['min']):>8} "
			f"{format_value(stats['median']):>8} "
			f"{format_value(stats['mean']):>8} "
			f"{format_value(stats['max']):>8} "
			f"{format_value(stats['total']):>10}"
		)
	return lines


def combine_tests(test_cases: list[str]) -> str:
	return "\n".join(test_cases)


def analyze_humaneval(
	rows: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase
) -> list[str]:
	prompt_chars = [len(row["prompt"]) for row in rows]
	prompt_lines = [non_empty_line_count(row["prompt"]) for row in rows]
	prompt_tokens = batched_token_counts(tokenizer, (row["prompt"] for row in rows))
	solution_chars = [len(row["canonical_solution"]) for row in rows]
	solution_lines = [non_empty_line_count(row["canonical_solution"]) for row in rows]
	solution_tokens = batched_token_counts(tokenizer, (row["canonical_solution"] for row in rows))
	test_chars = [len(row["test"]) for row in rows]
	test_tokens = batched_token_counts(tokenizer, (row["test"] for row in rows))
	test_asserts = [row["test"].count("assert ") for row in rows]
	entry_points = [row["entry_point"] for row in rows]

	lines = render_kv_section(
		"Overview",
		[
			("rows", len(rows)),
			("unique task ids", len({row['task_id'] for row in rows})),
			("unique entry points", len(set(entry_points))),
		],
	)
	lines.append("")
	lines.extend(
		render_distribution_section(
			"Content Lengths",
			[
				("prompt chars", summarize_numeric(prompt_chars)),
				("prompt lines", summarize_numeric(prompt_lines)),
				("solution chars", summarize_numeric(solution_chars)),
				("solution lines", summarize_numeric(solution_lines)),
				("test chars", summarize_numeric(test_chars)),
				("prompt tokens", summarize_numeric(prompt_tokens)),
				("solution tokens", summarize_numeric(solution_tokens)),
				("test tokens", summarize_numeric(test_tokens)),
				("asserts per test", summarize_numeric(test_asserts)),
			],
		)
	)
	return lines


def analyze_mbpp(rows: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> list[str]:
	prompt_chars = [len(row["text"]) for row in rows]
	prompt_tokens = batched_token_counts(tokenizer, (row["text"] for row in rows))
	code_chars = [len(row["code"]) for row in rows]
	code_tokens = batched_token_counts(tokenizer, (row["code"] for row in rows))
	code_lines = [non_empty_line_count(row["code"]) for row in rows]
	test_texts = [combine_tests(row["test_list"]) for row in rows]
	test_tokens = batched_token_counts(tokenizer, test_texts)
	challenge_test_texts = [combine_tests(row["challenge_test_list"]) for row in rows]
	challenge_test_tokens = batched_token_counts(tokenizer, challenge_test_texts)
	setup_code_tokens = batched_token_counts(tokenizer, (row["test_setup_code"] for row in rows))
	test_counts = [len(row["test_list"]) for row in rows]
	challenge_test_counts = [len(row["challenge_test_list"]) for row in rows]
	setup_code_lines = [non_empty_line_count(row["test_setup_code"]) for row in rows]
	empty_setup_code = sum(1 for row in rows if not row["test_setup_code"].strip())

	lines = render_kv_section(
		"Overview",
		[
			("rows", len(rows)),
			("task id range", f"{min(row['task_id'] for row in rows)}..{max(row['task_id'] for row in rows)}"),
			("empty setup code rows", empty_setup_code),
		],
	)
	lines.append("")
	lines.extend(
		render_distribution_section(
			"Content Lengths",
			[
				("prompt chars", summarize_numeric(prompt_chars)),
				("code chars", summarize_numeric(code_chars)),
				("code lines", summarize_numeric(code_lines)),
				("tests per task", summarize_numeric(test_counts)),
				("challenge tests", summarize_numeric(challenge_test_counts)),
				("setup code lines", summarize_numeric(setup_code_lines)),

				("prompt tokens", summarize_numeric(prompt_tokens)),
				("code tokens", summarize_numeric(code_tokens)),
				("test tokens", summarize_numeric(test_tokens)),
				("challenge tokens", summarize_numeric(challenge_test_tokens)),
				("setup code tokens", summarize_numeric(setup_code_tokens)),
			],
		)
	)
	return lines


def top_prefixes(task_ids: Iterable[str], prefix_split: str = "/") -> str:
	prefixes = [task_id.split(prefix_split, 1)[0] for task_id in task_ids if prefix_split in task_id]
	if not prefixes:
		return "n/a"
	counts = Counter(prefixes)
	return ", ".join(f"{name}={count}" for name, count in counts.most_common())


def load_rows(dataset_dir: Path) -> list[dict[str, Any]]:
	dataset = load_from_disk(str(dataset_dir))
	return [dataset[index] for index in range(len(dataset))]


def load_tokenizer(tokenizer_path: Path) -> PreTrainedTokenizerBase:
	if not tokenizer_path.exists():
		raise SystemExit(f"Tokenizer path not found: {tokenizer_path}")
	return AutoTokenizer.from_pretrained(tokenizer_path)


def render_dataset_header(dataset_name: str, task_prefixes: str) -> list[str]:
	title = dataset_name.upper()
	border = "=" * max(72, len(title))
	return [border, title, border, f"Task prefixes: {task_prefixes}", ""]


def analyze_dataset(
	dataset_name: str, data_dir: Path, tokenizer: PreTrainedTokenizerBase
) -> str:
	dataset_dir = data_dir / dataset_name
	if not dataset_dir.exists():
		raise SystemExit(f"Dataset path not found: {dataset_dir}")

	rows = load_rows(dataset_dir)
	if not rows:
		return "\n".join(render_dataset_header(dataset_name, "n/a") + ["Overview", "  rows  0"])

	lines = render_dataset_header(dataset_name, top_prefixes(str(row["task_id"]) for row in rows))
	if dataset_name == "openai_humaneval":
		lines.extend(analyze_humaneval(rows, tokenizer))
	elif dataset_name == "mbpp":
		lines.extend(analyze_mbpp(rows, tokenizer))
	else:
		lines.extend(
			render_kv_section(
				"Overview",
				[("rows", len(rows)), ("columns", ", ".join(rows[0].keys()))],
			)
		)
	return "\n".join(lines)


def main() -> None:
	args = parse_args()
	selected = []
	for name in args.datasets:
		canonical_name = canonical_dataset_name(name)
		if canonical_name not in selected:
			selected.append(canonical_name)

	tokenizer = load_tokenizer(args.tokenizer_path)
	reports = [analyze_dataset(dataset_name, args.data_dir, tokenizer) for dataset_name in selected]
	print("\n\n".join(reports))


if __name__ == "__main__":
	main()
