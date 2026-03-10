import os
print("loaded os")
from transformers import AutoModelForCausalLM, AutoTokenizer
print("loaded transformers")
from datasets import load_dataset
print("loaded datasets")
from pathlib import Path
print("loaded path")

MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
ASSETS = Path("assets")
MODEL_DIR = ASSETS / "model"
DATA_DIR = ASSETS / "data"

print(f"Downloading {MODEL_ID} to {MODEL_DIR}...")

# download tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

# download model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=MODEL_DIR,
    device_map="cpu"
)
model.save_pretrained("model")

print("Downloading datasets...")

# MBPP
mbpp = load_dataset("mbpp", split="test", cache_dir=DATA_DIR)
mbpp.save_to_disk(DATA_DIR / "mbpp")

# HumanEval
human_eval = load_dataset("openai_humaneval", split="test", cache_dir=DATA_DIR)
human_eval.save_to_disk(DATA_DIR / "human_eval")

print(f"Download complete")
