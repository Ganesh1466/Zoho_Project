from huggingface_hub import snapshot_download
import os

# Sahi OpenVINO-optimized Qwen 2.5 3B INT4 repository
model_dir = "./qwen-3b-openvino-int4"

print("Qwen 2.5 (3B) OpenVINO model download ho raha hai... Kripya intezaar karein.")
snapshot_download(repo_id="Prog-up/Qwen2.5-3B-Instruct-int4-ov", local_dir=model_dir)

print(f"\nSuccess! Model successfully is folder mein download ho gaya hai: {os.path.abspath(model_dir)}")