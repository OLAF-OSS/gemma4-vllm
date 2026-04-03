"""Download Gemma 4 E2B model for offline use."""

from huggingface_hub import snapshot_download

snapshot_download(
    "google/gemma-4-E2B-it",
    local_dir="./models/gemma-4-E2B-it",
    local_dir_use_symlinks=False,
)
