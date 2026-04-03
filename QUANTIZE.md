# Quantize Gemma 4 E4B to AWQ 4-bit

Create an AWQ-quantized model that runs on a 12GB GPU (RTX 3080 Ti) with vLLM and tool calling.

You only need **one L40 (48GB)** for this. The full BF16 model is ~16GB.

## On the L40 server

### 1. Set up the environment

```bash
mkdir -p ~/gemma4-quant && cd ~/gemma4-quant

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install autoawq transformers huggingface_hub
```

### 2. Log in to HuggingFace

```bash
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted
```

### 3. Run the quantization

Create `quantize.py`:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

MODEL_ID = "google/gemma-4-E4B-it"
OUTPUT_DIR = "./gemma-4-E4B-it-AWQ"

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

print(f"Loading {MODEL_ID}...")
model = AutoAWQForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Quantizing...")
model.quantize(tokenizer, quant_config=quant_config)

print(f"Saving to {OUTPUT_DIR}...")
model.save_quantized(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

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

Run it:

```bash
CUDA_VISIBLE_DEVICES=0 python quantize.py
```

This takes about 30-60 minutes on a single L40. You'll see progress per layer.

### 4. Verify the output

```bash
ls -lh gemma-4-E4B-it-AWQ/
```

You should see:

```
config.json                 # model config (will mention quant_config)
model.safetensors           # quantized weights (~5GB)
tokenizer.json              # tokenizer
tokenizer_config.json       # tokenizer config
preprocessor_config.json    # vision processor (needed by vLLM)
processor_config.json       # processor config (needed by vLLM)
chat_template.jinja         # chat template
```

Total size should be around **5-6GB** (down from ~16GB BF16).

### 5. Quick smoke test on the L40

```bash
pip install vllm

vllm serve ./gemma-4-E4B-it-AWQ \
  --quantization awq \
  --max-model-len 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --port 8000 &

# Wait ~30s for startup, then test
sleep 30

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-AWQ",
    "messages": [{"role": "user", "content": "Extract PII: John Smith, SSN 123-45-6789"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "piiRedactTool",
        "description": "Redact PII from text",
        "parameters": {
          "type": "object",
          "properties": {"prompt": {"type": "string"}},
          "required": ["prompt"]
        }
      }
    }],
    "tool_choice": "auto"
  }' | python3 -m json.tool

# Look for "tool_calls" in the response, not raw <|tool_call> tokens
# Kill the test server
kill %1
```

### 6. Upload to HuggingFace

Create a model card `gemma-4-E4B-it-AWQ/README.md`:

```markdown
---
base_model: google/gemma-4-E4B-it
library_name: transformers
tags:
  - awq
  - 4-bit
  - gemma4
  - vllm
license: apache-2.0
---

# Gemma 4 E4B IT — AWQ 4-bit

AWQ 4-bit quantization of [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it).

Fits in 12GB VRAM (RTX 3080 Ti, RTX 4070, etc.)

## Serving with vLLM

\```bash
vllm serve YOUR_USERNAME/gemma-4-E4B-it-AWQ \
  --quantization awq \
  --max-model-len 8192 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4
\```

## Quantization details

- **Method:** AWQ (AutoAWQ)
- **Bits:** 4
- **Group size:** 128
- **Zero point:** True
- **Kernel:** GEMM
```

Upload:

```bash
# Replace YOUR_USERNAME with your HuggingFace username
huggingface-cli upload YOUR_USERNAME/gemma-4-E4B-it-AWQ ./gemma-4-E4B-it-AWQ
```

The repo will be created automatically at `https://huggingface.co/YOUR_USERNAME/gemma-4-E4B-it-AWQ`.

### 7. Package for transfer (air-gapped alternative)

If you can't use HuggingFace (air-gapped target server), tar it instead:

```bash
tar -czf gemma-4-E4B-it-AWQ.tar.gz gemma-4-E4B-it-AWQ/
```

The archive will be ~5GB. Transfer it to your workstation.

## On your workstation (RTX 3080 Ti)

### 8. Get the model

**From HuggingFace** (if uploaded in step 6):

```bash
cd ~/projects/gemma4
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('YOUR_USERNAME/gemma-4-E4B-it-AWQ',
                  local_dir='./models/gemma-4-E4B-it-AWQ',
                  local_dir_use_symlinks=False)
"
```

**From tar** (if packaged in step 7):

```bash
cd ~/projects/gemma4
tar -xzf gemma-4-E4B-it-AWQ.tar.gz -C models/
```

### 9. Build the container image (if not already built)

```bash
./service.sh build
```

### 10. Add E4B-AWQ variant to service.sh

Add a new case to the `GEMMA_VARIANT` block in `service.sh`:

```bash
  e4b-awq)
    HF_MODEL="YOUR_USERNAME/gemma-4-E4B-it-AWQ"
    SERVED_NAME="gemma-4-E4B-it"
    MODEL_DIR="gemma-4-E4B-it-AWQ"
    TENSOR_PARALLEL=1
    QUANTIZATION_ARGS=(--quantization awq)
    MAX_MODEL_LEN=8192
    EXTRA_ARGS=(--enforce-eager)
    SHM_SIZE=""
    ;;
```

Then run:

```bash
GEMMA_VARIANT=e4b-awq ./service.sh up
GEMMA_VARIANT=e4b-awq ./service.sh test
GEMMA_VARIANT=e4b-awq ./service.sh status
```

Expected VRAM: ~5-6GB weights + ~4-5GB KV cache = fits in 12GB.

### 11. Test streaming tool calls

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it",
    "stream": true,
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }],
    "tool_choice": "auto"
  }' | grep "tool_calls"
```

You should see streaming `tool_calls` deltas (not raw `<|tool_call>` tokens).

## Troubleshooting

**`OSError: Can't load feature extractor`**
The quantized model is missing `preprocessor_config.json`. Copy it from the base model:
```bash
huggingface-cli download google/gemma-4-E4B-it preprocessor_config.json processor_config.json \
  --local-dir models/gemma-4-E4B-it-AWQ/
```

**OOM on the L40 during quantization**
Use a single GPU: `CUDA_VISIBLE_DEVICES=0 python quantize.py`

**OOM on the 3080 Ti during inference**
Reduce context: `--max-model-len 4096` or `--max-model-len 2048`

**`autoawq` doesn't support Gemma 4 architecture**
AutoAWQ may need an update for the PLE architecture. If it fails, try GPTQ instead:
```bash
pip install auto-gptq optimum
```
Then use `quantize-gptq.py` from this repo (same steps, swap `--quantization awq` for `--quantization gptq` when serving).
