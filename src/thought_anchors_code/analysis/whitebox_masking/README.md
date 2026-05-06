# Whitebox Masking — Causal Matrix

Implements Section 5 / Algorithm 1 (Appendix M) of *Thought Anchors* (Bogdan, Macar et al. 2025).

For each rollout we produce an **M×M** matrix where entry `(i, j)` = mean log-KL divergence at target sentence `j`'s tokens when source sentence `i`'s attention is suppressed across all layers.

## Running

```bash
# HumanEval (42 correct rollouts)
python -m thought_anchors_code.analysis.whitebox_masking.run \
    --dataset humaneval

# MBPP (96 correct rollouts)
python -m thought_anchors_code.analysis.whitebox_masking.run \
    --dataset mbpp

# Resume a partial run
python -m thought_anchors_code.analysis.whitebox_masking.run \
    --dataset humaneval --resume

# First 5 rollouts only, force CPU
python -m thought_anchors_code.analysis.whitebox_masking.run \
    --dataset humaneval --max-rollouts 5 --device cpu
```

## Output

`results/causal_matrices_{dataset}_qwen3_5_0_8b/{task_id}_s{sample_id}.npz`

Each `.npz` contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `causal_matrix` | `[M, M]` float32 | NaN on diagonal and upper triangle; lower triangle = col-normalised mean log-KL |
| `sentence_indices` | `[M]` int32 | 0-based sentence numbers |
| `task_id` | scalar | Task identifier string |
| `sample_id` | scalar | Sample index |
| `dataset_name` | scalar | Dataset name string |
| `num_sentences` | scalar | M |

## Joining with other results

All three outputs share `(task_id, sample_id)` as keys:

- **Sentence labels**: `results/sentence_labels_{dataset}_qwen3_5_0_8b.jsonl` — labels are 1-based string keys; convert with `int(k) - 1` to get the 0-based sentence index
- **Receiver-head summaries**: `results/receiver_head_summary_{dataset}_qwen3_5_0_8b.jsonl` — `sentence_scores[i]` aligns with `causal_matrix[:, i]`

## Expected runtime

| Setting | M | Estimated time |
|---------|---|----------------|
| Qwen3.5-0.8B, float32, GPU | 20 | ~20 s/rollout |
| Qwen3.5-0.8B, float32, GPU | 80 | ~4 min/rollout |
| M1 Mac CPU | 20 | ~2–5 min/rollout |
| 42 HumanEval correct rollouts (GPU, avg M≈15) | — | ~10 min total |

## Algorithm

1. **Base pass**: run the model with no hooks to get `base_logits [seq_len, vocab]`
2. **For each source sentence i**: apply `QwenAttentionHookManager` to suppress attention from `[src_start, src_end)` tokens across all layers, get `masked_logits`
3. **For each target sentence j > i**: compute `mean log-KL(base‖masked)` over target tokens → `causal_matrix[i, j]`
4. **Column normalise**: subtract `nanmean(causal_matrix[:j, j])` from each column `j`
