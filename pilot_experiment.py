"""Black-box resampling pilot experiment for thought-anchor detection on HumanEval."""

import ast
import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "assets" / "data"
MODEL_CACHE = Path(__file__).parent / "assets" / "model"
OUTPUT_FILE = Path(__file__).parent / "pilot_results.json"

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

NUM_PROBLEMS = 10
NUM_RESAMPLES = 5
TEMPERATURE = 0.7
MAX_NEW_TOKENS_ORIGINAL = 2048
MAX_NEW_TOKENS_RESAMPLE = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wrap_prompt(prompt: str, tokenizer) -> str:
    """Format the HumanEval prompt as a chat message and return the raw string."""
    messages = [
        {
            "role": "user",
            "content": (
                "Solve the following Python programming problem step by step. "
                "Think through your approach carefully, then write a complete Python solution.\n\n"
                f"{prompt}"
            ),
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_text(model, tokenizer, raw_text: str, max_new_tokens: int) -> str:
    """Tokenize raw_text, generate, and return only the newly generated tokens decoded."""
    inputs = tokenizer(raw_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=None,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def split_into_sentences(trace: str) -> tuple[str, list[str]]:
    """Split only the prose reasoning part of the trace into sentence fragments.

    Stops before the first ```python block so code lines are never treated as
    sentences. If a <think>…</think> block exists, uses only that content.

    Returns (content, sentences).
    """
    think_match = re.search(r"<think>(.*?)</think>", trace, re.DOTALL)
    if think_match:
        content = think_match.group(1)
    else:
        start = trace.find("<think>")
        if start != -1:
            content = trace[start + len("<think>"):]
        else:
            content = trace

    # Truncate at the first code block so code lines aren't split as sentences
    code_fence = content.find("```")
    if code_fence != -1:
        content = content[:code_fence]

    raw_parts = re.split(r"\.\s+|\n", content)
    sentences = [s.strip() for s in raw_parts if len(s.strip()) >= 10]
    return content, sentences


def extract_final_answer(text: str) -> str:
    """Extract the last ```python … ``` block; fall back to everything after </think>.

    Returns "" if no code block or </think> marker is found, so that failed
    extractions produce a predictable value rather than a unique noise string.
    """
    code_blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    think_end = text.rfind("</think>")
    if think_end != -1:
        return text[think_end + len("</think>"):].strip()

    return ""


def normalize_code(code: str) -> str:
    """Normalize Python code via AST round-trip to ignore formatting differences.

    Falls back to stripped string if the code doesn't parse (e.g. incomplete snippet).
    """
    try:
        return ast.unparse(ast.parse(code))
    except SyntaxError:
        return code.strip()


def compute_influence_score(original_answer: str, resample_answers: list[str]) -> float:
    """Fraction of resamples whose AST-normalized answer differs from the original."""
    original_norm = normalize_code(original_answer)
    differ = sum(
        1 for ans in resample_answers if normalize_code(ans) != original_norm
    )
    return differ / len(resample_answers)


def build_resample_prefix(formatted_prompt: str, sentences: list[str], i: int) -> str:
    """Build the raw-text prefix for resampling sentence i.

    Input to the model = formatted_prompt + sentences[:i]
    The model continues the response from sentence i onward.
    """
    prefix_sentences = " ".join(sentences[:i])
    if prefix_sentences:
        return formatted_prompt + prefix_sentences + " "
    return formatted_prompt


# ---------------------------------------------------------------------------
# Per-problem processing
# ---------------------------------------------------------------------------

def process_problem(problem_idx: int, row: dict, model, tokenizer) -> dict:
    task_id = row["task_id"]
    prompt = row["prompt"]

    print(f"\n{'='*60}")
    print(f"Problem {problem_idx}: {task_id}")
    print(f"{'='*60}")

    # 1. Format prompt
    formatted_prompt = wrap_prompt(prompt, tokenizer)

    # 2. Generate original CoT trace
    print("Generating original CoT trace...")
    original_trace = generate_text(model, tokenizer, formatted_prompt, MAX_NEW_TOKENS_ORIGINAL)

    preview = original_trace[:300] + ("..." if len(original_trace) > 300 else "")
    print(f"Original trace ({len(original_trace)} chars):\n{preview}")

    # 3. Extract original final answer
    original_answer = extract_final_answer(original_trace)
    print(f"Original answer ({len(original_answer)} chars):\n{original_answer[:200]}")

    # 4. Split into sentences
    _, sentences = split_into_sentences(original_trace)
    print(f"Split into {len(sentences)} sentences")

    if not sentences:
        print("WARNING: No sentences extracted — skipping influence scoring.")
        return {
            "problem_id": task_id,
            "prompt": prompt,
            "original_trace": original_trace,
            "sentences": [],
            "influence_scores": [],
            "top_anchor_idx": None,
            "top_anchor_text": None,
        }

    # 5. Resample each sentence and compute influence scores
    influence_scores = []

    for i, sentence in enumerate(tqdm(sentences, desc=f"  Sentences [{task_id}]", leave=False)):
        resample_prefix = build_resample_prefix(formatted_prompt, sentences, i)
        resample_answers = []

        for _ in range(NUM_RESAMPLES):
            resample_text = generate_text(model, tokenizer, resample_prefix, MAX_NEW_TOKENS_RESAMPLE)
            full_resample = resample_prefix + resample_text
            resample_answers.append(extract_final_answer(full_resample))

        score = compute_influence_score(original_answer, resample_answers)
        influence_scores.append(score)
        print(f"  S{i}: score={score:.2f}  [{sentence[:60]}]")

    # 6. Find top anchor: first sentence where the score drops below 0.5,
    # indicating the prefix up to this point is enough to lock in the answer.
    # Fall back to the lowest-scoring sentence if no score drops below 0.5.
    top_anchor_idx = next(
        (i for i, s in enumerate(influence_scores) if s < 0.5),
        int(max(range(len(influence_scores)), key=lambda k: influence_scores[k])),
    )
    top_anchor_text = sentences[top_anchor_idx]

    print(f"\nTop anchor: S{top_anchor_idx} (score={influence_scores[top_anchor_idx]:.2f})")
    print(f"  Text: {top_anchor_text[:120]}")

    return {
        "problem_id": task_id,
        "prompt": prompt,
        "original_trace": original_trace,
        "sentences": sentences,
        "influence_scores": influence_scores,
        "top_anchor_idx": top_anchor_idx,
        "top_anchor_text": top_anchor_text,
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(results: list[dict]) -> None:
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} result(s) to {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading HumanEval dataset...")
    dataset = load_from_disk(str(DATA_DIR / "openai_humaneval"))
    problems = [dataset[i] for i in range(NUM_PROBLEMS)]
    print(f"Loaded {len(problems)} problems")

    print(f"Loading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded. Device: {next(model.parameters()).device}")

    all_results = []
    for idx, row in tqdm(enumerate(problems), total=NUM_PROBLEMS, desc="Problems"):
        result = process_problem(idx, row, model, tokenizer)
        all_results.append(result)
        save_results(all_results)

    print(f"\nDone! Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
