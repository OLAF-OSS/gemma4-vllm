# Quantize Gemma 4 E4B to GPTQ 4-bit

Create a GPTQ-quantized model that runs on a 12GB GPU (RTX 3080 Ti) with vLLM and tool calling.

Uses **auto-round** (RTN mode) inside the vLLM container for correct dependency versions.

You only need **one L40S (48GB)** for this. The full BF16 model is ~16GB.

## On the L40S server

### 1. Prepare the quantization script

Create `quantize.py`:

```python
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
```

### 2. Build the vLLM image (if not already built)

```bash
./service.sh build
```

### 3. Run the quantization inside the vLLM container

The vLLM container has the correct torch/transformers versions. We install auto-round into it at runtime:

```bash
mkdir -p quantize-work

# Copy the script
cp quantize.py quantize-work/

# Run on a single GPU (adjust --device to a free GPU)
podman run --rm --entrypoint bash \
  --device nvidia.com/gpu=4 \
  --shm-size=16g \
  -v ./quantize-work:/work:Z \
  -e HF_HOME=/work/.cache \
  -e CUDA_VISIBLE_DEVICES=0 \
  -w /work \
  localhost/gemma4-vllm:latest \
  -c "
pip install auto-round 2>/dev/null | tail -1
python3 /work/quantize.py
"
```

This takes about 5-10 minutes (RTN mode, no optimization iterations).

### 4. Fix incompatible tensors

The vLLM 0.19.0 Gemma4 loader doesn't expect `softcap` tensors saved by transformers 5.x. Remove them:

```bash
podman run --rm --entrypoint python3 \
  -v ./quantize-work/gemma-4-E4B-it-W4A16:/model:Z \
  localhost/gemma4-vllm:latest \
  -c "
import safetensors.torch as st
import json, os

model_dir = '/model'
for fname in os.listdir(model_dir):
    if not fname.endswith('.safetensors'):
        continue
    fpath = os.path.join(model_dir, fname)
    tensors = st.load_file(fpath)
    bad = [k for k in tensors if 'softcap' in k]
    if bad:
        print(f'{fname}: removing {len(bad)} softcap entries')
        for k in bad:
            del tensors[k]
        st.save_file(tensors, fpath)
    else:
        print(f'{fname}: clean')

idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
with open(idx_path) as f:
    idx = json.load(f)
removed = [k for k in idx['weight_map'] if 'softcap' in k]
for k in removed:
    del idx['weight_map'][k]
with open(idx_path, 'w') as f:
    json.dump(idx, f, indent=2)
print(f'Index: removed {len(removed)} entries. Done.')
"
```

### 5. Verify the output

```bash
ls -lh quantize-work/gemma-4-E4B-it-W4A16/
```

You should see:

```
config.json                       # model config (includes quant_config)
model-00001-of-00002.safetensors  # quantized weights (~5.3GB)
model-00002-of-00002.safetensors  # quantized weights (~4.2GB)
model.safetensors.index.json      # weight index
tokenizer.json                    # tokenizer
tokenizer_config.json             # tokenizer config
processor_config.json             # processor config (needed by vLLM)
chat_template.jinja               # chat template
quantization_config.json          # quantization metadata
```

Total size: ~9.5GB (language model 4-bit, vision/audio towers full precision).
Model loading in vLLM: ~9.65 GiB VRAM — fits on a 12GB GPU.

### 6. Quick smoke test on the L40S

```bash
# Copy to models dir
cp -r quantize-work/gemma-4-E4B-it-W4A16 models/

# Start
GEMMA_VARIANT=e4b GPUS=5 PORT=8003 ./service.sh up

# Test tool calling
GEMMA_VARIANT=e4b PORT=8003 ./service.sh test

# Stop
GEMMA_VARIANT=e4b ./service.sh rm
```

### 7. Upload to HuggingFace

```bash
hf upload YOUR_USERNAME/gemma-4-E4B-it-W4A16 quantize-work/gemma-4-E4B-it-W4A16
```

If uploads time out through a proxy, upload files one at a time via Python:

```python
import httpx
_orig = httpx.Client.__init__
def _patch(self, *a, **kw):
    kw['timeout'] = httpx.Timeout(600.0, connect=60.0)
    _orig(self, *a, **kw)
httpx.Client.__init__ = _patch

from huggingface_hub import HfApi
import os

api = HfApi()
folder = "quantize-work/gemma-4-E4B-it-W4A16"
repo = "YOUR_USERNAME/gemma-4-E4B-it-W4A16"

api.create_repo(repo, repo_type="model", exist_ok=True)
for f in sorted(os.listdir(folder), key=lambda f: os.path.getsize(os.path.join(folder, f))):
    fpath = os.path.join(folder, f)
    print(f"Uploading {f} ({os.path.getsize(fpath)/1e6:.1f} MB)...")
    api.upload_file(path_or_fileobj=fpath, path_in_repo=f, repo_id=repo, repo_type="model")
    print(f"  Done: {f}")
```

## On the 3080 Ti workstation

### 8. Download the model

```bash
hf download YOUR_USERNAME/gemma-4-E4B-it-W4A16 --local-dir ./models/gemma-4-E4B-it-W4A16
```

### 9. Build the container image (if not already built)

```bash
./service.sh build
```

### 10. Run

```bash
GEMMA_VARIANT=e4b ./service.sh up
GEMMA_VARIANT=e4b ./service.sh test
GEMMA_VARIANT=e4b ./service.sh status
```

Expected VRAM: ~9.65 GiB model + KV cache — fits in 12GB.

## Why not AutoAWQ or llm-compressor?

- **AutoAWQ**: Deprecated and doesn't support the Gemma 4 architecture (`gemma4 isn't supported yet`).
- **llm-compressor (GPTQ)**: Uses `torch.fx` tracing which fails on Gemma 4's multimodal architecture (`Proxy object cannot be iterated`).
- **auto-round with optimization** (`iters>0`): Fails on Gemma 4's heterogeneous attention heads (256 vs 512 dimensions) during gradient computation.
- **auto-round RTN** (`iters=0`): Works. Slightly lower quality than optimized GPTQ but avoids all architecture-specific issues.

## Troubleshooting

**`ValueError: ... softcap ... no module or parameter`**
The safetensors files contain `softcap` tensors from transformers 5.x that vLLM 0.19.0 doesn't expect. Run step 4 to remove them.

**`ValueError: ... audio_tower ... g_idx`**
The audio/vision tower layers were quantized. Re-run quantization with `layer_config` that sets those layers to `bits=32`.

**OOM on the 3080 Ti during inference**
Reduce context: change `MAX_MODEL_LEN` to `4096` or `2048` in the `e4b` variant in `service.sh`.
