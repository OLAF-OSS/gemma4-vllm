FROM docker.io/vllm/vllm-openai:latest

# vLLM 0.19.0 pins transformers<5 and huggingface_hub<1, but Gemma 4 requires
# transformers >=5.x which needs huggingface_hub >=1.0
RUN pip install --no-cache-dir \
    "huggingface_hub>=1.0" && \
    pip install --no-cache-dir --no-deps \
    "transformers @ https://github.com/huggingface/transformers/archive/refs/heads/main.zip"
