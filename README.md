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
- rollout collection writes JSONL incrementally and resumes by default; use `--no-resume` to overwrite the output file
- use `--batch-size` for higher generation throughput when VRAM allows

Full 0.8B rollout commands used for the two code datasets:

```bash
uv run collect-code-rollouts humaneval --output assets/rollouts/humaneval_qwen3_5_0_8b_full.jsonl
uv run collect-code-rollouts mbpp --output assets/rollouts/mbpp_qwen3_5_0_8b_full.jsonl
```

### Receiver-head analysis

Run the white-box attention receiver-head pipeline on a saved rollout file:

```bash
uv run receiver-head-analysis assets/rollouts/humaneval_pilot.jsonl
```

Recommended commands for the two 0.8B rollout files:

```bash
uv run receiver-head-analysis assets/rollouts/humaneval_qwen3_5_0_8b_full.jsonl \
  --top-k 20 \
  --proximity-ignore 4 \
  --output results/receiver_head_summary_humaneval_qwen3_5_0_8b.jsonl

uv run receiver-head-analysis assets/rollouts/mbpp_qwen3_5_0_8b_full.jsonl \
  --top-k 20 \
  --proximity-ignore 4 \
  --output results/receiver_head_summary_mbpp_qwen3_5_0_8b.jsonl
```

Useful flags:

- use the same model for attention analysis that was used to generate the rollouts
- `--top-k`: number of receiver heads selected by mean kurtosis
- `--proximity-ignore`: ignore local sentence neighbors when computing vertical attention scores
- `--no-resume`: overwrite the output JSONL instead of skipping rows already present in the summary file
- `--no-truncate`: analyze full reasoning traces instead of truncating each trace at the input file's 75th percentile sentence count

Default analysis behavior:

- receiver-head analysis computes the input file's 75th percentile sentence count and truncates longer traces to that cutoff before scoring
- this avoids pathological long rollouts dominating runtime and memory use
- for the current 0.8B rollouts, the observed p75 cutoffs were `35` sentences for HumanEval and `13` sentences for MBPP
- attention cache files are written incrementally under `assets/cache/whitebox_attention/`
- summary JSONL rows are appended and flushed after global receiver-head ranking is complete
- output rows resume by default; cached attention work also survives interrupted runs

The summary JSONL contains one row per analyzed rollout:

```json
{"task_id":"HumanEval/1","sample_id":0,"sentence_scores":[null,null,0.0012],"code_sentence_scores":[0.0008,0.0021,0.0044],"receiver_head_scores":[0.0045],"dataset_name":"openai_humaneval","model_id":"Qwen/Qwen3.5-0.8B"}
```

`sentence_scores` are paper-style receiver-head scores: later reasoning attending back to each earlier reasoning sentence. `code_sentence_scores` are code-to-reasoning scores: final code/answer attention back to each reasoning sentence using the selected receiver heads.

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

`plot-receiver-heads` also truncates at the input file's 75th percentile sentence count by default. Pass `--no-truncate` to plot full traces.

### Attention review UI

Build a static HTML UI for manually reviewing each rollout beside receiver-head sentence scores:

```bash
uv run build-attention-review-ui \
  assets/rollouts/humaneval_qwen3_5_0_8b_full.jsonl \
  results/receiver_head_summary_humaneval_qwen3_5_0_8b.jsonl \
  --output results/humaneval_attention_review.html

uv run build-attention-review-ui \
  assets/rollouts/mbpp_qwen3_5_0_8b_full.jsonl \
  results/receiver_head_summary_mbpp_qwen3_5_0_8b.jsonl \
  --output results/mbpp_attention_review.html
```

Open the generated files directly:

```bash
xdg-open results/humaneval_attention_review.html
xdg-open results/mbpp_attention_review.html
```

The UI supports search, correctness filtering, sorting by max sentence score, and per-sentence heat shading.
Rows show both `R` scores (later reasoning to reasoning) and `C` scores (final code to reasoning). Dimmed rows were not part of the analyzed prefix, usually because of p75 truncation.

### Black-box resampling

Run causal sentence resampling over every reasoning sentence in each rollout. The command truncates each rollout to the file-level 75th percentile sentence count by default, builds a continuation prompt from the original task plus the reasoning prefix before each sentence, omits that sentence, samples new continuations/code, and evaluates the regenerated code against the benchmark tests.

Small sanity run:

```bash
uv run blackbox-resampling \
  assets/rollouts/humaneval_qwen3_5_0_8b_full.jsonl \
  results/receiver_head_summary_humaneval_qwen3_5_0_8b.jsonl \
  humaneval \
  --limit-rollouts 2 \
  --num-resamples 1 \
  --batch-size 4 \
  --max-new-tokens 256 \
  --output results/blackbox_resampling_humaneval_pilot.jsonl
```

Paper-style runs should use more resamples:

```bash
uv run blackbox-resampling \
  assets/rollouts/humaneval_qwen3_5_0_8b_full.jsonl \
  results/receiver_head_summary_humaneval_qwen3_5_0_8b.jsonl \
  humaneval \
  --num-resamples 5 \
  --batch-size 8 \
  --output results/blackbox_resampling_humaneval_qwen3_5_0_8b.jsonl

uv run blackbox-resampling \
  assets/rollouts/mbpp_qwen3_5_0_8b_full.jsonl \
  results/receiver_head_summary_mbpp_qwen3_5_0_8b.jsonl \
  mbpp \
  --num-resamples 5 \
  --batch-size 8 \
  --output results/blackbox_resampling_mbpp_qwen3_5_0_8b.jsonl
```

By default, the command resamples every sentence in each rollout and attaches the receiver-head scores as metadata when available. `--batch-size` controls how many intervention/resample continuations are generated in one model call; lower it if VRAM is tight. Output JSONL resumes by default using `(dataset, task_id, sample_id, sentence_index, selection)`, where `selection` is fixed to `sentence`.

Summarize completed runs:

```bash
uv run summarize-blackbox-resampling \
  results/blackbox_resampling_humaneval_qwen3_5_0_8b.jsonl \
  --output results/blackbox_resampling_humaneval_summary.csv
```

### After receiver-head analysis

Recommended follow-up workflow:

- inspect high max-score rollouts in the review UI
- manually label repeated high-scoring sentence types such as planning, backtracking, constraint restatement, and strategy shifts
- compare high-scoring correct vs incorrect rollouts
- run exhaustive sentence-wise resampling under the p75 cutoff to test whether removing each sentence changes downstream code or correctness
- use the receiver-head scores as annotations, not as the sentence-selection gate

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
- static attention review UI
- black-box sentence resampling over every sentence with p75 truncation and summary CSVs
- tests for config, data loading, rollout parsing, and white-box helpers

Not implemented yet:

- end-to-end black-box resampling results on the full rollout set
- white-box masking experiments
