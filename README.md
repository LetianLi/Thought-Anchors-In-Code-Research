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

### Sentence labeling

Label each reasoning sentence in correct rollouts with function tags from the Thought Anchors taxonomy using the Gemini API. Requires a `GEMINI_API_KEY` environment variable.

```bash
GEMINI_API_KEY="..." uv run python scripts/label_sentences.py humaneval
GEMINI_API_KEY="..." uv run python scripts/label_sentences.py mbpp
```

Output: `results/sentence_labels_{dataset}_qwen3_5_0_8b.jsonl`

Only processes `is_correct == True` rollouts. Use `--dry-run` to label the first 3 rollouts without writing to the production file.

### Causal masking (whitebox attention suppression)

Compute M×M sentence-level causal influence matrices per rollout by suppressing attention from each source sentence across all softmax-attention layers and measuring mean log-KL divergence on target tokens. Implements Section 5 / Algorithm 1 (Appendix M) of the paper.

```bash
uv run python -m thought_anchors_code.analysis.whitebox_masking.run --dataset humaneval
uv run python -m thought_anchors_code.analysis.whitebox_masking.run --dataset mbpp
```

Options:
- `--resume` — skip already-computed rollouts
- `--max-rollouts N` — limit to N rollouts (useful for smoke testing)
- `--device cuda|cpu|auto` — override device placement

Output: `results/causal_matrices_{dataset}_qwen3_5_0_8b/{task_id}_s{sample_id}.npz`

Each `.npz` contains `causal_matrix` [M×M float32], NaN on diagonal and upper triangle, column-normalised lower triangle.

### Figure generation

Generate all analysis figures from computed causal matrices, receiver-head summaries, and sentence labels:

```bash
uv run python analyze_causal_matrices.py --dataset humaneval mbpp
```

Saves to `results/figures/`:

| File | Description |
|------|-------------|
| `causal_correlation_{dataset}.png` | Scatter of receiver-head R-score vs causal influence exerted per sentence, with anchor position histogram |
| `causal_heatmaps_{dataset}.png` | Representative M×M causal matrices for the largest rollouts, tick labels colour-coded by function tag |
| `causal_position_{dataset}.png` | Same scatter coloured by relative sentence position in trace |
| `causal_by_tag_{dataset}.png` | Mean causal influence exerted and mean R-score per function tag |
| `causal_tag_pair_{dataset}.png` | Mean causal influence averaged by (source tag, target tag) pair |

To inspect individual sentence pairs with the strongest causal influence:

```bash
uv run python show_causal_examples.py --dataset humaneval
uv run python show_causal_examples.py --dataset humaneval --task HumanEval/7 --top 5
```
