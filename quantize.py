"""Quantize Gemma 4 E4B to 4-bit using auto-round (GPTQ-compatible)."""

import shutil
from auto_round import AutoRound
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from huggingface_hub import hf_hub_download

MODEL_ID = "google/gemma-4-E4B-it"
OUTPUT_DIR = "./gemma-4-E4B-it-W4A16"

print(f"Loading {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Only quantize the language model layers, skip vision/audio towers
layer_config = {}
for name, _ in model.named_modules():
    if name.startswith(("model.vision_tower", "model.audio_tower",
                        "model.multi_modal_projector", "model.audio_projector")):
        layer_config[name] = {"bits": 32}

print(f"Configuring auto-round 4-bit quantization (skipping {len(layer_config)} non-LM modules)...")
autoround = AutoRound(
    model,
    tokenizer,
    processor=processor,
    bits=4,
    group_size=128,
    nsamples=256,
    seqlen=2048,
    iters=0,
    disable_opt_rtn=True,
    layer_config=layer_config,
)

print("Running quantization...")
autoround.quantize()

print(f"Saving to {OUTPUT_DIR} (auto_gptq format for vLLM)...")
autoround.save_quantized(OUTPUT_DIR, format="auto_gptq")

# Copy multimodal processor configs (required by vLLM)
for filename in [
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
]:
    try:
        path = hf_hub_download(MODEL_ID, filename)
        shutil.copy(path, OUTPUT_DIR)
        print(f"  Copied {filename}")
    except Exception:
        pass

print("Done.")
