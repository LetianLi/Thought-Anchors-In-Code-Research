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

### Setup

Download datasets `MBPP` and `HumanEval` and the model:

```bash
uv run download-assets [--model Qwen/Qwen3.5-0.8B]
```


### Collect rollouts

Collect rollouts to JSONL files for downstream analysis:

```bash
uv run collect-code-rollouts humaneval --output assets/rollouts/humaneval_qwen3_5_0_8b_full.jsonl [--limit 10] [--max-new-tokens 10000] [--batch-size 4] [--no-resume]
uv run collect-code-rollouts mbpp --output assets/rollouts/mbpp_qwen3_5_0_8b_full.jsonl [--limit 10] [--max-new-tokens 10000] [--batch-size 4] [--no-resume]
```

### Receiver-head analysis

Run the white-box attention receiver-head pipeline on a saved rollout file:

```bash
uv run receiver-head-analysis assets/rollouts/humaneval.jsonl --output results/receiver_head_summary_humaneval_qwen3_5_0_8b.jsonl [--top-k 20] [--proximity-ignore 4] [--no-resume] [--no-truncate]
uv run receiver-head-analysis assets/rollouts/mbpp.jsonl --output results/receiver_head_summary_mbpp_qwen3_5_0_8b.jsonl [--top-k 20] [--proximity-ignore 4] [--no-resume] [--no-truncate]
```
For the current 0.8B rollouts, the observed sentence count p75 cutoffs were `35` sentences for HumanEval and `13` sentences for MBPP.

### Receiver-head plots

Generate receiver-head vertical plot:

```bash
uv run plot-receiver-heads assets/rollouts/openai_humaneval_rollouts.jsonl --model Qwen/Qwen3.5-0.8B --output-dir results/figure4_demo [--all-rollouts] [--no-truncate]
```

### Attention review UI

Automatic UI gen for reviewing each rollout's attention scores:

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

R scores are reasoning-reasoning attention.
C scores are finalcode-reasoning attention.


### Black-box resampling

Run causal sentence resampling over every reasoning sentence in each rollout. Rollouts truncated to the 75th percentile sentence count. Resamples rollouts at each sentence truncation.

```bash
uv run blackbox-resampling \
  assets/rollouts/humaneval_qwen3_5_0_8b_full.jsonl \
  results/receiver_head_summary_humaneval_qwen3_5_0_8b.jsonl \
  humaneval \
  --output results/blackbox_resampling_humaneval_qwen3_5_0_8b.jsonl \
  --num-resamples 1 \
  [--limit-rollouts 2] \
  [--batch-size 4] \
  [--max-new-tokens 10000] \
```

Summarize completed runs:

```bash
uv run summarize-blackbox-resampling \
  results/blackbox_resampling_humaneval_qwen3_5_0_8b.jsonl \
  --output results/blackbox_resampling_humaneval_summary.csv
```
