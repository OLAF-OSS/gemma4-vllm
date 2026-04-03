#!/bin/bash
set -euo pipefail

POD_NAME="gemma4"
IMAGE="gemma4-vllm:latest"
MODEL_PATH="${MODEL_PATH:-./models/gemma-4-E2B-it}"

# Resolve to absolute path
MODEL_PATH="$(realpath "$MODEL_PATH")"

# Clean up any previous pod
podman pod exists "$POD_NAME" 2>/dev/null && podman pod rm -f "$POD_NAME"

# Create the pod
# slirp4netns is needed for reliable port forwarding in rootless mode
podman pod create \
  --name "$POD_NAME" \
  -p 8000:8000 \
  --network slirp4netns:port_handler=slirp4netns \
  --share ipc,net,uts

# Run vLLM inside the pod
podman run -d \
  --pod "$POD_NAME" \
  --name "${POD_NAME}-vllm" \
  --device nvidia.com/gpu=all \
  -v "${MODEL_PATH}:/model:ro,Z" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  "$IMAGE" \
  --model /model \
  --served-model-name gemma-4-E2B-it \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --host 0.0.0.0 \
  --port 8000

echo "Pod '$POD_NAME' started. Waiting for health..."
for i in $(seq 1 90); do
  sleep 5
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null | grep -q 200; then
    echo "Ready at http://localhost:8000/v1  (after $((i*5))s)"
    exit 0
  fi
  STATUS=$(podman inspect --format '{{.State.Status}}' "${POD_NAME}-vllm" 2>/dev/null || echo "unknown")
  if [ "$STATUS" != "running" ]; then
    echo "Container died (status=$STATUS). Logs:"
    podman logs "${POD_NAME}-vllm" 2>&1 | tail -15
    exit 1
  fi
done
echo "Timeout waiting for server"
exit 1
