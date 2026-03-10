import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from pathlib import Path

MODEL_ID = "unsloth/DeepSeek-R1-0528-Qwen3-8B-bnb-4bit" 
ASSETS = Path("assets")
MODEL_DIR = ASSETS / "model"
DATA_DIR = ASSETS / "data"

print(f"Downloading pre-quantized {MODEL_ID} to {MODEL_DIR}...")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

# Download the pre-quantized model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=MODEL_DIR,
    device_map="auto"
)

print("Downloading datasets...")

# MBPP
mbpp = load_dataset("mbpp", split="test", cache_dir=DATA_DIR)
mbpp.save_to_disk(DATA_DIR / "mbpp")

# HumanEval
human_eval = load_dataset("openai_humaneval", split="test", cache_dir=DATA_DIR)
human_eval.save_to_disk(DATA_DIR / "openai_humaneval")

print(f"Download complete")
