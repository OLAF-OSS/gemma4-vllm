"""Download Gemma 4 model for offline use."""

import sys

from huggingface_hub import snapshot_download

MODELS = {
    "e2b": "google/gemma-4-E2B-it",
    "e4b": "ciocan/gemma-4-E4B-it-W4A16",
    "26b": "google/gemma-4-26B-A4B-it",
    "31b": "google/gemma-4-31B-it",
}

variant = sys.argv[1] if len(sys.argv) > 1 else "e2b"
if variant not in MODELS:
    print(f"Unknown variant '{variant}'. Use: {', '.join(MODELS)}")
    sys.exit(1)

repo_id = MODELS[variant]
local_dir = f"./models/{repo_id.split('/')[-1]}"

print(f"Downloading {repo_id} to {local_dir}...")
snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
print(f"Done: {local_dir}")
