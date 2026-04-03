#!/bin/bash
# Serve Gemma 4 E2B (5.1B params, 2.3B effective) with tool calling on RTX 3080 Ti (12GB)
# Uses bitsandbytes 4-bit quantization (~3GB weights), leaving room for KV cache
# NOTE: --reasoning-parser removed because it breaks streaming tool calls

uv run vllm serve google/gemma-4-E2B-it \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --port 8000
