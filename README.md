# Gemma 4 with vLLM + Tool Calling

Serve Google's Gemma 4 models with OpenAI-compatible tool calling. Supports three variants — from a single consumer GPU up to multi-GPU tensor-parallel setups.

## Models

| Variant | Model | Params | GPUs | Context | Quantization |
|---------|-------|--------|------|---------|--------------|
| `e2b` | [gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | 5.1B (2.3B effective) | 1 | 8K | bitsandbytes 4-bit |
| `e4b` | [gemma-4-E4B-it-W4A16](https://huggingface.co/ciocan/gemma-4-E4B-it-W4A16) | 8B (4B effective) | 1 | 8K | GPTQ 4-bit (auto-round) |
| `26b` | [gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | 26B MoE (4B active) | 2 (tensor parallel) | 256K | none (BF16) |
| `31b` | [gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | 31B dense | 4 (tensor parallel) | 256K | none (BF16) |

All models share: Apache 2.0 license, multimodal input (text, image, video), tool calling support via vLLM's `gemma4` parser.

## Quick Start (bare metal)

```bash
# Install dependencies
uv sync

# Serve E2B (downloads model on first run)
./serve.sh

# Test tool calling
uv run python test_tool_call.py
```

The server starts on `http://localhost:8000` with an OpenAI-compatible API.

## Docker

```bash
# Download model for offline use
uv run python download-model.py

# Build and run
docker compose up --build
```

## Podman

Podman requires NVIDIA CDI (Container Device Interface) for GPU access:

```bash
# One-time setup: generate CDI spec
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# One-time setup: enable podman socket (needed by podman compose)
systemctl --user enable --now podman.socket
```

### Podman Compose

```bash
# Download model for offline use
uv run python download-model.py

# Build and run
podman compose -f podman-compose.yml up --build
```

### Podman Pod (service script)

`service.sh` manages the full pod lifecycle. Set `GEMMA_VARIANT` to choose the model (`e2b`, `26b`, or `31b`):

```bash
# Build the image (shared across all variants)
./service.sh build

# --- Run the 31B model (most powerful, 4 GPUs) ---
GEMMA_VARIANT=31b ./service.sh download
GEMMA_VARIANT=31b GPUS=0,1,2,3 ./service.sh up
GEMMA_VARIANT=31b ./service.sh status
GEMMA_VARIANT=31b ./service.sh test

# --- Run 26B alongside 31B on the remaining GPUs ---
GEMMA_VARIANT=26b ./service.sh download
GEMMA_VARIANT=26b GPUS=4,5 PORT=8001 ./service.sh up

# --- Run E2B (default, single GPU) ---
./service.sh download
GPUS=5 PORT=8002 ./service.sh up

# View logs (supports extra args like -f, --tail, --since)
GEMMA_VARIANT=31b ./service.sh logs
GEMMA_VARIANT=31b ./service.sh logs -f --tail 100

# Stop / start / full restart
GEMMA_VARIANT=31b ./service.sh down
GEMMA_VARIANT=31b ./service.sh up
GEMMA_VARIANT=31b ./service.sh restart

# Remove pod entirely
GEMMA_VARIANT=31b ./service.sh rm

# Custom model path
MODEL_PATH=/data/models/gemma-4-31B-it GEMMA_VARIANT=31b ./service.sh up
```

Each variant creates its own pod (`gemma4-e2b`, `gemma4-26b`, `gemma4-31b`). Set `PORT` and `GPUS` to run multiple variants simultaneously on different ports and GPU sets. With 6x L40S GPUs, you can run 31B on GPUs 0-3 and 26B on GPUs 4-5 at the same time.

### Podman Kube Play

Use the Kubernetes YAML for a declarative approach (E2B only):

```bash
# Build the image
podman build -t gemma4-vllm:latest .

# Download model
uv run python download-model.py

# Start the pod
podman kube play gemma4-pod.yml

# Stop and remove
podman kube down gemma4-pod.yml
```

This YAML can also be deployed to Kubernetes with minimal changes (swap CDI annotation for `nvidia.com/gpu` resource limits).

## Air-Gapped Deployment

### Prepare (on a machine with internet)

```bash
# 1. Download the model(s) you need
uv run python download-model.py e2b
uv run python download-model.py 31b

# 2. Build the container image
podman build -t gemma4-vllm:latest .
# or: docker compose build

# 3. Export image and model for transfer
podman save gemma4-vllm:latest | gzip > gemma4-vllm.tar.gz
tar -czf gemma4-model.tar.gz models/
```

### Deploy (on the air-gapped server)

Transfer `gemma4-vllm.tar.gz`, `gemma4-model.tar.gz`, and `service.sh` (or compose files) to the server.

```bash
# 1. Load the container image
podman load < gemma4-vllm.tar.gz
# or: docker load < gemma4-vllm.tar.gz

# 2. Extract the model
tar -xzf gemma4-model.tar.gz

# 3. Run (pick your variant)
GEMMA_VARIANT=31b ./service.sh up
# or for compose (E2B only):
# podman compose -f podman-compose.yml up
```

The service script and compose files set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to prevent any network calls.

## API Usage

The server exposes an OpenAI-compatible API at `http://localhost:8000/v1`.

### Tool calling

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Use the served model name matching your GEMMA_VARIANT:
#   e2b -> "gemma-4-E2B-it"
#   26b -> "gemma-4-26B-A4B-it"
#   31b -> "gemma-4-31B-it"
response = client.chat.completions.create(
    model="gemma-4-31B-it",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"],
            },
        },
    }],
    tool_choice="auto",
)
```

### Vercel AI SDK / Mastra

```typescript
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";

const vllm = createOpenAICompatible({
  name: "vllm",
  apiKey: "unused",
  baseURL: "http://localhost:8000/v1",
});

// Use the served model name matching your GEMMA_VARIANT
// e.g. vllm("gemma-4-31B-it"), vllm("gemma-4-26B-A4B-it"), or vllm("gemma-4-E2B-it")
```

## Known Issues

### `--reasoning-parser gemma4` breaks streaming tool calls

When both `--reasoning-parser gemma4` and `--tool-call-parser gemma4` are set, vLLM's streaming code path waits for reasoning (`<thought>` tags) to end before invoking the tool parser. If the model skips reasoning and goes straight to tool calls, the parser never activates and raw `<|tool_call>` tokens are returned as text content.

**Fix:** Do not use `--reasoning-parser gemma4`. The `serve.sh` and compose files already omit it.

### GPU requirements by variant

| Variant | Min VRAM | Notes |
|---------|----------|-------|
| E2B | 12GB (1 GPU) | bitsandbytes 4-bit, fits on consumer GPUs |
| E4B | 12GB (1 GPU) | GPTQ 4-bit (~9GB weights), fits on consumer GPUs |
| 26B | 2x 48GB (tensor parallel) | MoE model, BF16, ~52GB weights + 256K KV cache |
| 31B | 4x 48GB (tensor parallel) | Dense model, BF16, ~62GB weights + 256K KV cache (32 attn heads require tp divisible by 2/4/8) |

The E4B model (8B params) is not included as a variant — it OOMs on 12GB GPUs, and on larger GPUs the 26B/31B models are a better choice.

### transformers version

vLLM 0.19.0 pins `transformers<5`, but Gemma 4 requires `transformers>=5.x`. The Dockerfile handles this by upgrading transformers and huggingface_hub. For bare metal, `pyproject.toml` uses `[tool.uv] override-dependencies` to force the override.

### Rootless podman port forwarding

The default `pasta` network backend in rootless podman can reset connections when forwarding ports from pods. The `pod.sh` script uses `slirp4netns` as a workaround:

```
--network slirp4netns:port_handler=slirp4netns
```

If running as root or with `podman-compose`, this is not needed.

## Files

| File | Description |
|---|---|
| `serve.sh` | Bare metal vLLM server launcher |
| `Dockerfile` | Extends vLLM image with transformers 5.x |
| `docker-compose.yml` | Docker Compose setup (air-gap ready) |
| `podman-compose.yml` | Podman Compose setup (air-gap ready) |
| `service.sh` | Pod lifecycle manager (up/down/logs/status/test/download) — supports E2B, 26B, 31B |
| `pod.sh` | Minimal pod launcher script (E2B) |
| `gemma4-pod.yml` | Kubernetes YAML for `podman kube play` (E2B) |
| `download-model.py` | Downloads model for offline use (accepts variant: e2b, 26b, 31b) |
| `test_tool_call.py` | Tool calling smoke test |
| `pyproject.toml` | Python dependencies (bare metal) |
