"""
Quantize Gemma 4 E4B to AWQ 4-bit.

AWQ (Activation-aware Weight Quantization) works by:
1. Loading the full BF16 model
2. Running a small calibration dataset through it
3. Finding which weights matter most (based on activation magnitudes)
4. Quantizing less-important weights more aggressively
5. Saving the result as a compact 4-bit model (~5GB)

Requirements:
  - GPU with >= 24GB VRAM (A100, 4090, etc.) OR use CPU offloading (slow)
  - pip install autoawq

Usage:
  python quantize-awq.py
  # Output: ./models/gemma-4-E4B-it-AWQ/
"""

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "google/gemma-4-E4B-it"
OUTPUT_DIR = "./models/gemma-4-E4B-it-AWQ"

# Quantization config
quant_config = {
    "zero_point": True,      # use asymmetric quantization
    "q_group_size": 128,     # group size for quantization
    "w_bit": 4,              # 4-bit weights
    "version": "GEMM",       # use GEMM kernel (fastest for batch=1)
}

print(f"Loading {MODEL_ID}...")
model = AutoAWQForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Quantizing (this takes 30-60 minutes)...")
model.quantize(tokenizer, quant_config=quant_config)

print(f"Saving to {OUTPUT_DIR}...")
model.save_quantized(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Copy processor configs needed by vLLM for multimodal models
import shutil
from huggingface_hub import hf_hub_download

for filename in ["preprocessor_config.json", "processor_config.json"]:
    try:
        path = hf_hub_download(MODEL_ID, filename)
        shutil.copy(path, OUTPUT_DIR)
        print(f"Copied {filename}")
    except Exception:
        pass

print(f"Done. Serve with: vllm serve {OUTPUT_DIR} --quantization awq")
