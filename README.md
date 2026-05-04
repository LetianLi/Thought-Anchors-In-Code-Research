# Thought-Anchors-In-Code-Research
Research into the what thought anchors exist in Code Reasoning, extending the original paper's analysis from the math domain (https://arxiv.org/abs/2506.19143)

## Assignment
Done as UW-Madison's CS639 NLP lecture's group assignment.

## Environment

This repo uses `uv`.

Install dependencies with:

```bash
uv sync
```

## Paths

Paths are:

- local model packages: `assets/model/<model-name>`
- Hugging Face cache: `assets/hf-cache`
- datasets: `assets/data`
- rollouts: `assets/rollouts`
- analysis cache: `assets/cache`

## Scripts

### Download assets

Download the default local model plus `MBPP` and `HumanEval`:

```bash
uv run download-assets
```

Useful variants:

```bash
uv run download-assets --skip-model
uv run download-assets --skip-datasets
uv run download-assets --model Qwen/Qwen3.5-0.8B
```

### Dataset stats

Summarize downloaded datasets with the local tokenizer:

```bash
uv run python dataset_stats.py
uv run python dataset_stats.py humaneval mbpp
```

### Collect code rollouts

Collect reasoning rollouts and save them as JSONL files for downstream analysis:

```bash
uv run collect-code-rollouts humaneval --limit 10
uv run collect-code-rollouts mbpp --limit 10
```

Example with explicit output path:

```bash
uv run collect-code-rollouts humaneval --limit 5 --output assets/rollouts/humaneval_pilot.jsonl
```

Useful note:

- the current default model is `Qwen/Qwen3.5-0.8B`
- rollout generation is slow on this machine, so for sanity checks prefer small runs such as `--limit 1 --max-new-tokens 16`

### Receiver-head analysis

Run the white-box attention receiver-head pipeline on a saved rollout file:

```bash
uv run receiver-head-analysis assets/rollouts/humaneval_pilot.jsonl
```

Example with custom settings:

```bash
uv run receiver-head-analysis assets/rollouts/humaneval_pilot.jsonl --top-k 10 --proximity-ignore 4 --output results/receiver_head_summary.jsonl
```

Important note:

- use the same model for attention analysis that was used to generate the rollouts

### Receiver-head plots

Generate Figure 4 style receiver-head demo plots from a rollout JSONL file:

```bash
uv run plot-receiver-heads assets/rollouts/openai_humaneval_rollouts.jsonl --model Qwen/Qwen3.5-0.8B --output-dir results/figure4_demo
```

Generate a demo folder for every rollout in the file:

```bash
uv run plot-receiver-heads assets/rollouts/openai_humaneval_rollouts.jsonl --model Qwen/Qwen3.5-0.8B --output-dir results/figure4_demo_all --all-rollouts
```

This writes:

- `figure4_receiver_heads.png`
- `figure4_head_matrix.png`
- `figure4_kurtosis_histogram.png`
- `figure4_metadata.txt`

## Current architecture

- `src/thought_anchors_code/setup/`: setup and download utilities
- `src/thought_anchors_code/rollouts/`: prompt construction and rollout collection
- `src/thought_anchors_code/analysis/whitebox_attention/`: sentence splitting, token alignment, attention aggregation, and receiver-head scoring
- `src/thought_anchors_code/engine/`: local Transformers/PyTorch model loading

## Current status

Implemented now:

- canonical `uv` workflow
- local asset download command
- rollout collection to JSONL files
- receiver-head analysis scaffold
- receiver-head plotting for Figure 4 style demos
- tests for config, data loading, rollout parsing, and white-box helpers

Not implemented yet:

- execution-based correctness evaluation for generated code
- full black-box resampling experiment pipeline
- end-to-end validated receiver-head results on a larger rollout set
