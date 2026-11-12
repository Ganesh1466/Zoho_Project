"""Download the matching public FP16 OpenVINO model without changing INT4."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "Godreign/qwen2.5-3b-instruct-openvino-fp16-model"
MODEL_REVISION = "734421db23fc97a491abf029dcb1d060374404a0"
MODEL_DIR = Path(__file__).resolve().parent / "qwen-3b-openvino-fp16"


def main() -> None:
    print(f"Downloading {MODEL_ID} at pinned revision {MODEL_REVISION}.")
    print(f"Target directory: {MODEL_DIR}")
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(MODEL_DIR),
    )

    config_path = MODEL_DIR / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("_name_or_path") != "Qwen/Qwen2.5-3B-Instruct":
        raise RuntimeError("Downloaded model config does not identify the expected base model.")
    if config.get("torch_dtype") != "float16":
        raise RuntimeError("Downloaded model config does not declare float16 weights.")
    for required in ("openvino_model.xml", "openvino_model.bin", "tokenizer_config.json"):
        if not (MODEL_DIR / required).is_file():
            raise RuntimeError(f"Downloaded model is missing required file: {required}")

    print(f"FP16 OpenVINO model is ready at {MODEL_DIR}")


if __name__ == "__main__":
    main()