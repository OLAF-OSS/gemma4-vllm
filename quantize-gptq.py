"""
Quantize Gemma 4 E4B to GPTQ 4-bit.

GPTQ (Generative Pre-trained Transformer Quantization) works by:
1. Loading the full BF16 model
2. Processing one layer at a time with a calibration dataset
3. Solving a least-squares problem to find the best 4-bit approximation
   for each layer's weights (minimizing output error)
4. Saving the result as a compact 4-bit model (~5GB)

GPTQ is slightly slower to quantize than AWQ but often produces
better quality at the same bit width.

Requirements:
  - GPU with >= 24GB VRAM (A100, 4090, etc.)
  - pip install auto-gptq optimum

Usage:
  python quantize-gptq.py
  # Output: ./models/gemma-4-E4B-it-GPTQ/
"""

from transformers import AutoTokenizer, GPTQConfig, AutoModelForCausalLM
import shutil
from huggingface_hub import hf_hub_download

MODEL_ID = "google/gemma-4-E4B-it"
OUTPUT_DIR = "./models/gemma-4-E4B-it-GPTQ"

# Calibration data — a small set of representative text
calibration_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Extract all personally identifiable information from this document.",
    "John Smith lives at 123 Main Street, New York, NY 10001.",
    "The patient was diagnosed with hypertension on 2024-03-15.",
    "Please summarize the key findings in the following report.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "The annual revenue increased by 15% to $4.2 billion in Q3 2024.",
]

print(f"Loading tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Tokenize calibration data
calibration_dataset = [
    tokenizer(text, return_tensors="pt") for text in calibration_texts
]

# GPTQ config
gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset=calibration_texts,
    tokenizer=tokenizer,
)

print(f"Loading and quantizing {MODEL_ID} (this takes 1-2 hours)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=gptq_config,
    device_map="auto",
)

print(f"Saving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Copy processor configs needed by vLLM
for filename in ["preprocessor_config.json", "processor_config.json"]:
    try:
        path = hf_hub_download(MODEL_ID, filename)
        shutil.copy(path, OUTPUT_DIR)
        print(f"Copied {filename}")
    except Exception:
        pass

print(f"Done. Serve with: vllm serve {OUTPUT_DIR} --quantization gptq")
