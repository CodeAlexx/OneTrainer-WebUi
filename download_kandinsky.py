
from huggingface_hub import snapshot_download
import os

model_id = "kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s"
local_dir = "models/kandinsky-5-video-pro"

print(f"Downloading {model_id} to {local_dir}...")
snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
print("Download complete.")
