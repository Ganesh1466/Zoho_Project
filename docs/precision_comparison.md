# Precision comparison status

- Existing INT4 model: benchmarked on CPU with three measured runs per prompt
  after warm-up (six rows total).
- Mean throughput: 3.52 tokens/second; median TTFT: 777 ms. See
  `benchmarks/precision_comparison.csv`.
- A matching pre-converted Qwen2.5-3B-Instruct FP16 OpenVINO repository was
  identified, but its download was stopped. The partial model folder contains
  no model weights; no FP16 measurements are claimed.
- Ask the user before resuming any model download.
- INT4 uses fewer bits for weights and reduces model storage; quantization can
  affect quality and does not guarantee faster CPU inference.

To compare after an authorized FP16 download:

```powershell
.\openvino-env\Scripts\python.exe .\benchmark.py --int4 .\qwen-3b-openvino-int4 --fp16 .\qwen-3b-openvino-fp16 --runs 3 --warmup 1
```

Generated answers are saved for manual quality review; no quality score is
automatically computed.
