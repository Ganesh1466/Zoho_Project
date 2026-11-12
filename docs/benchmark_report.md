# Benchmark update

The CPU INT4 benchmark used two fixed prompts, one warm-up per prompt, and
three measured runs per prompt (six rows). Average throughput was 3.52
tokens/second; median TTFT was 777 ms. Actual per-run values are in
`benchmarks/precision_comparison.csv` and `benchmarks/results.csv`.

The KV/context run recorded ten calls across six scenarios. Results are in
`benchmarks/kv_cache_experiments.csv`. The installed runtime provides no
verified cache-off control or cache-reuse counter, so these are not an
enabled-versus-disabled cache comparison.

Re-run with:

```powershell
.\openvino-env\Scripts\python.exe .\benchmark.py --int4 .\qwen-3b-openvino-int4 --runs 3 --warmup 1 --max-new-tokens 16
.\openvino-env\Scripts\python.exe .\kv_cache_experiment.py --max-new-tokens 12
```

Results are machine- and runtime-specific. FP16 was not benchmarked.
