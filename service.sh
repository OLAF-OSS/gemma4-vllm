#!/bin/bash
set -euo pipefail

# ============================================================
# Model selection — set GEMMA_VARIANT to one of: e2b, 26b, 31b
# ============================================================
GEMMA_VARIANT="${GEMMA_VARIANT:-e2b}"

case "$GEMMA_VARIANT" in
  e2b)
    HF_MODEL="google/gemma-4-E2B-it"
    SERVED_NAME="gemma-4-E2B-it"
    MODEL_DIR="gemma-4-E2B-it"
    TENSOR_PARALLEL=1
    QUANTIZATION_ARGS=(--quantization bitsandbytes --load-format bitsandbytes)
    MAX_MODEL_LEN=8192
    EXTRA_ARGS=(--enforce-eager)
    SHM_SIZE=""
    ;;
  26b)
    HF_MODEL="google/gemma-4-26B-A4B-it"
    SERVED_NAME="gemma-4-26B-A4B-it"
    MODEL_DIR="gemma-4-26B-A4B-it"
    TENSOR_PARALLEL=2
    QUANTIZATION_ARGS=()
    MAX_MODEL_LEN=262144
    EXTRA_ARGS=(--reasoning-parser gemma4 --default-chat-template-kwargs '{"enable_thinking": true}')
    SHM_SIZE="16g"
    ;;
  31b)
    HF_MODEL="google/gemma-4-31B-it"
    SERVED_NAME="gemma-4-31B-it"
    MODEL_DIR="gemma-4-31B-it"
    TENSOR_PARALLEL=4
    QUANTIZATION_ARGS=()
    MAX_MODEL_LEN=262144
    EXTRA_ARGS=(--reasoning-parser gemma4 --default-chat-template-kwargs '{"enable_thinking": true}')
    SHM_SIZE="16g"
    ;;
  *)
    echo "Unknown variant '$GEMMA_VARIANT'. Use: e2b, 26b, 31b"
    exit 1
    ;;
esac

POD_NAME="gemma4-${GEMMA_VARIANT}"
CONTAINER_NAME="${POD_NAME}-vllm"
IMAGE="localhost/gemma4-vllm:latest"
MODEL_PATH="${MODEL_PATH:-./models/${MODEL_DIR}}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-all}"
HEALTH_URL="http://localhost:${PORT}/health"

# Build GPU device flags
if [ "$GPUS" = "all" ]; then
  GPU_DEVICES=(--device nvidia.com/gpu=all)
else
  GPU_DEVICES=()
  for g in $(echo "$GPUS" | tr ',' ' '); do
    GPU_DEVICES+=(--device "nvidia.com/gpu=$g")
  done
fi

red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }

pod_exists()      { podman pod exists "$POD_NAME" 2>/dev/null; }
container_status() { podman inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "missing"; }
is_healthy()      { curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null | grep -q 200; }

wait_for_health() {
  local timeout="${1:-120}"
  echo "Waiting for health check (model: $SERVED_NAME, tp=$TENSOR_PARALLEL)..."
  for i in $(seq 1 "$timeout"); do
    sleep 5
    if is_healthy; then
      green "Ready at http://localhost:${PORT}/v1  (${i}x5s)"
      return 0
    fi
    local status
    status=$(container_status)
    if [ "$status" != "running" ]; then
      red "Container died (status=$status)"
      podman logs --tail 15 "$CONTAINER_NAME" 2>&1
      return 1
    fi
  done
  red "Timeout after $((timeout * 5))s"
  return 1
}

build_vllm_args() {
  VLLM_ARGS=(
    --model /model
    --served-model-name "$SERVED_NAME"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization 0.90
    --enable-auto-tool-choice
    --tool-call-parser gemma4
    --host 0.0.0.0
    --port "$PORT"
  )
  if [ "$TENSOR_PARALLEL" -gt 1 ]; then
    VLLM_ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL")
  fi
  if [ ${#QUANTIZATION_ARGS[@]} -gt 0 ]; then
    VLLM_ARGS+=("${QUANTIZATION_ARGS[@]}")
  fi
  if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    VLLM_ARGS+=("${EXTRA_ARGS[@]}")
  fi
}

cmd_up() {
  if pod_exists && [ "$(container_status)" = "running" ] && is_healthy; then
    green "Already running and healthy ($SERVED_NAME)"
    return 0
  fi

  # If the pod exists but is stopped, start it
  if pod_exists; then
    yellow "Pod exists, starting..."
    podman pod start "$POD_NAME"
    wait_for_health
    return $?
  fi

  # Fresh start
  MODEL_PATH="$(realpath "$MODEL_PATH")"
  if [ ! -d "$MODEL_PATH" ]; then
    red "Model not found at $MODEL_PATH"
    echo "Run: uv run python download-model.py $GEMMA_VARIANT"
    exit 1
  fi

  echo "Creating pod '$POD_NAME' (model: $SERVED_NAME, tp=$TENSOR_PARALLEL)..."
  podman pod create \
    --name "$POD_NAME" \
    -p "${PORT}:${PORT}" \
    --share ipc,net,uts

  build_vllm_args

  # Build container run args
  local -a ctr_args=(
    -d
    --pod "$POD_NAME"
    --name "$CONTAINER_NAME"
    ${GPU_DEVICES[@]}
    -v "${MODEL_PATH}:/model:ro,Z"
    -e HF_HUB_OFFLINE=1
    -e TRANSFORMERS_OFFLINE=1
  )
  if [ -n "$SHM_SIZE" ]; then
    ctr_args+=(--shm-size "$SHM_SIZE")
  fi

  echo "Starting vLLM container..."
  podman run "${ctr_args[@]}" "$IMAGE" "${VLLM_ARGS[@]}"

  wait_for_health
}

cmd_down() {
  if ! pod_exists; then
    yellow "Pod '$POD_NAME' does not exist"
    return 0
  fi
  echo "Stopping pod '$POD_NAME'..."
  podman pod stop "$POD_NAME"
  green "Stopped"
}

cmd_rm() {
  if ! pod_exists; then
    yellow "Pod '$POD_NAME' does not exist"
    return 0
  fi
  echo "Removing pod '$POD_NAME'..."
  podman pod rm -f "$POD_NAME"
  green "Removed"
}

cmd_restart() {
  cmd_down
  sleep 2
  # Remove so we get a fresh pod (port bindings are set at creation)
  pod_exists && podman pod rm -f "$POD_NAME"
  cmd_up
}

cmd_status() {
  echo "Variant:   $GEMMA_VARIANT ($SERVED_NAME)"
  if ! pod_exists; then
    echo "Pod:       missing"
    return
  fi
  local pod_status
  pod_status=$(podman pod inspect "$POD_NAME" --format '{{.State}}' 2>/dev/null || echo "unknown")
  local ctr_status
  ctr_status=$(container_status)

  echo "Pod:       $pod_status"
  echo "Container: $ctr_status"

  if [ "$ctr_status" = "running" ]; then
    if is_healthy; then
      green "Health:    ok"
    else
      red "Health:    not responding"
    fi
    echo "Endpoint:  http://localhost:${PORT}/v1"

    # GPU memory
    local gpu_mem
    gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$gpu_mem" ]; then
      echo "GPU VRAM:  ${gpu_mem// /} MiB"
    fi
  fi
}

cmd_logs() {
  if ! pod_exists; then
    red "Pod '$POD_NAME' does not exist"
    exit 1
  fi
  if [ $# -eq 0 ]; then
    podman logs --tail 50 "$CONTAINER_NAME"
  else
    podman logs "$@" "$CONTAINER_NAME"
  fi
}

cmd_build() {
  echo "Building image '$IMAGE'..."
  podman build -t "$IMAGE" .
  green "Built $IMAGE"
}

cmd_download() {
  echo "Downloading model: $HF_MODEL -> ./models/$MODEL_DIR"
  uv run python download-model.py "$GEMMA_VARIANT"
  green "Downloaded $HF_MODEL"
}

cmd_test() {
  if ! is_healthy; then
    red "Server not healthy at $HEALTH_URL"
    exit 1
  fi
  echo "Testing tool calling ($SERVED_NAME)..."
  local response
  response=$(curl -s "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$SERVED_NAME\",
      \"messages\": [{\"role\": \"user\", \"content\": \"What is the weather in Paris?\"}],
      \"tools\": [{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}],
      \"tool_choice\": \"auto\",
      \"skip_special_tokens\": false
    }")

  if echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); tc=d['choices'][0]['message']['tool_calls'][0]; print(f\"Tool: {tc['function']['name']}\nArgs: {tc['function']['arguments']}\")" 2>/dev/null; then
    green "Tool calling works"
  else
    red "Tool calling failed"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    exit 1
  fi

  # Check reasoning (26b/31b only)
  if [ "$GEMMA_VARIANT" != "e2b" ]; then
    local reasoning
    reasoning=$(echo "$response" | python3 -c "import sys,json; m=json.load(sys.stdin)['choices'][0]['message']; print(m.get('reasoning') or '')" 2>/dev/null)
    if [ -n "$reasoning" ]; then
      green "Reasoning works"
    else
      yellow "Reasoning: not present in response"
    fi
  fi
}

usage() {
  cat <<EOF
Usage: GEMMA_VARIANT=<variant> $(basename "$0") <command>

Variants:
  e2b       Gemma 4 E2B  (5.1B params, 2.3B effective) — 1 GPU, 4-bit quantized
  26b       Gemma 4 26B  (26B MoE, 4B active)           — 2 GPUs, tensor parallel, 256K context
  31b       Gemma 4 31B  (31B dense, most powerful)      — 4 GPUs, tensor parallel, 256K context

Current variant: $GEMMA_VARIANT ($SERVED_NAME), port: $PORT, gpus: $GPUS

Commands:
  up        Create and start the pod (or start if stopped)
  down      Stop the pod
  rm        Stop and remove the pod
  restart   Restart the pod (full recreate)
  status    Show pod status, health, and GPU usage
  logs      Show container logs (pass extra args, e.g. -f --tail 100)
  build     Build the container image
  download  Download the model for offline use
  test      Run a tool calling smoke test

Examples:
  GEMMA_VARIANT=31b ./service.sh build
  GEMMA_VARIANT=31b ./service.sh download
  GEMMA_VARIANT=31b GPUS=0,1,2,3 ./service.sh up
  GEMMA_VARIANT=26b GPUS=4,5 PORT=8001 ./service.sh up    # run alongside 31b

Environment:
  GEMMA_VARIANT  Model variant (default: e2b)
  PORT           API port (default: 8000)
  GPUS           GPU indices, comma-separated (default: all)
  MODEL_PATH     Override model directory path
EOF
}

case "${1:-}" in
  up)       cmd_up ;;
  down)     cmd_down ;;
  rm)       cmd_rm ;;
  restart)  cmd_restart ;;
  status)   cmd_status ;;
  logs)     shift; cmd_logs "$@" ;;
  build)    cmd_build ;;
  download) cmd_download ;;
  test)     cmd_test ;;
  *)        usage ;;
esac
