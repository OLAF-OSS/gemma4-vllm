#!/bin/bash
set -euo pipefail

POD_NAME="gemma4"
CONTAINER_NAME="${POD_NAME}-vllm"
IMAGE="gemma4-vllm:latest"
MODEL_PATH="${MODEL_PATH:-./models/gemma-4-E2B-it}"
PORT=8000
HEALTH_URL="http://localhost:${PORT}/health"

red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }

pod_exists()      { podman pod exists "$POD_NAME" 2>/dev/null; }
container_status() { podman inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "missing"; }
is_healthy()      { curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null | grep -q 200; }

wait_for_health() {
  local timeout="${1:-90}"
  echo "Waiting for health check..."
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

cmd_up() {
  if pod_exists && [ "$(container_status)" = "running" ] && is_healthy; then
    green "Already running and healthy"
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
    echo "Run: uv run python download-model.py"
    exit 1
  fi

  echo "Creating pod '$POD_NAME'..."
  podman pod create \
    --name "$POD_NAME" \
    -p "${PORT}:${PORT}" \
    --network slirp4netns:port_handler=slirp4netns \
    --share ipc,net,uts

  echo "Starting vLLM container..."
  podman run -d \
    --pod "$POD_NAME" \
    --name "$CONTAINER_NAME" \
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
    --port "$PORT"

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

cmd_test() {
  if ! is_healthy; then
    red "Server not healthy at $HEALTH_URL"
    exit 1
  fi
  echo "Testing tool calling..."
  local response
  response=$(curl -s "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "gemma-4-E2B-it",
      "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
      "tools": [{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],
      "tool_choice": "auto"
    }')

  if echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); tc=d['choices'][0]['message']['tool_calls'][0]; print(f\"Tool: {tc['function']['name']}\nArgs: {tc['function']['arguments']}\")" 2>/dev/null; then
    green "Tool calling works"
  else
    red "Tool calling failed"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    exit 1
  fi
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  up        Create and start the pod (or start if stopped)
  down      Stop the pod
  rm        Stop and remove the pod
  restart   Restart the pod (full recreate)
  status    Show pod status, health, and GPU usage
  logs      Show container logs (pass extra args, e.g. -f --tail 100)
  build     Build the container image
  test      Run a tool calling smoke test

Environment:
  MODEL_PATH   Path to model directory (default: ./models/gemma-4-E2B-it)
EOF
}

case "${1:-}" in
  up)      cmd_up ;;
  down)    cmd_down ;;
  rm)      cmd_rm ;;
  restart) cmd_restart ;;
  status)  cmd_status ;;
  logs)    shift; cmd_logs "$@" ;;
  build)   cmd_build ;;
  test)    cmd_test ;;
  *)       usage ;;
esac
