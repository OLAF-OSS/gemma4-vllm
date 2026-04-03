# Gemma 4 E2B with vLLM + Tool Calling

Serve Google's Gemma 4 E2B (5.1B params, 2.3B effective) with OpenAI-compatible tool calling on consumer GPUs.

Tested on NVIDIA RTX 3080 Ti (12GB VRAM) using 4-bit bitsandbytes quantization.

## Model

| | |
|---|---|
| Model | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) |
| Architecture | Dense + PLE (Per-Layer Embeddings) |
| Parameters | 5.1B total, 2.3B effective |
| Context | 128K native, 8K configured (VRAM constraint) |
| Modalities | Text, Image, Audio, Video |
| License | Apache 2.0 |
| Quantization | bitsandbytes 4-bit (~3GB weights) |
| VRAM usage | ~7.2GB model + ~2.1GB KV cache |

## Quick Start (bare metal)

```bash
# Install dependencies
uv sync

# Serve (downloads model on first run)
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

`service.sh` manages the full pod lifecycle:

```bash
# Build the image
./service.sh build

# Download model
uv run python download-model.py

# Start (creates pod, waits for health check)
./service.sh up

# Check status, health, and GPU usage
./service.sh status

# Run a tool calling smoke test
./service.sh test

# View logs (supports extra args like -f, --tail, --since)
./service.sh logs
./service.sh logs -f --tail 100

# Stop / start / full restart
./service.sh down
./service.sh up
./service.sh restart

# Remove pod entirely
./service.sh rm

# Custom model path
MODEL_PATH=/data/models/gemma-4-E2B-it ./service.sh up
```

### Podman Kube Play

Use the Kubernetes YAML for a declarative approach:

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
# 1. Download the model
uv run python download-model.py

# 2. Build the container image
podman build -t gemma4-vllm:latest .
# or: docker compose build

# 3. Export image and model for transfer
podman save gemma4-vllm:latest | gzip > gemma4-vllm.tar.gz
tar -czf gemma4-model.tar.gz models/
```

### Deploy (on the air-gapped server)

Transfer `gemma4-vllm.tar.gz`, `gemma4-model.tar.gz`, and `podman-compose.yml` (or `docker-compose.yml`) to the server.

```bash
# 1. Load the container image
podman load < gemma4-vllm.tar.gz
# or: docker load < gemma4-vllm.tar.gz

# 2. Extract the model
tar -xzf gemma4-model.tar.gz

# 3. Run
podman compose -f podman-compose.yml up
# or: docker compose up
```

The compose files set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to prevent any network calls.

To use a custom model path:

```bash
MODEL_PATH=/data/models/gemma-4-E2B-it podman compose -f podman-compose.yml up
```

## API Usage

The server exposes an OpenAI-compatible API at `http://localhost:8000/v1`.

### Tool calling

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="gemma-4-E2B-it",
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

// Use vllm("gemma-4-E2B-it") as the model in your agent/generateText calls
```

## Known Issues

### `--reasoning-parser gemma4` breaks streaming tool calls

When both `--reasoning-parser gemma4` and `--tool-call-parser gemma4` are set, vLLM's streaming code path waits for reasoning (`<thought>` tags) to end before invoking the tool parser. If the model skips reasoning and goes straight to tool calls, the parser never activates and raw `<|tool_call>` tokens are returned as text content.

**Fix:** Do not use `--reasoning-parser gemma4`. The `serve.sh` and compose files already omit it.

### Gemma 4 E4B does not fit on 12GB GPUs

The E4B model (8B params) OOMs even at 4-bit quantization because bitsandbytes dequantizes weights to BF16 during forward passes, requiring ~10.8GB just for computation with no room for KV cache.

Use E2B (5.1B params) instead, which fits comfortably at 4-bit (~7.2GB + 2.1GB KV cache).

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
| `service.sh` | Pod lifecycle manager (up/down/logs/status/test) |
| `pod.sh` | Minimal pod launcher script |
| `gemma4-pod.yml` | Kubernetes YAML for `podman kube play` |
| `download-model.py` | Downloads model for offline use |
| `test_tool_call.py` | Tool calling smoke test |
| `pyproject.toml` | Python dependencies (bare metal) |
