# KV-cache experiment update

Chat history is the conversation text. The KV cache is attention state managed
by the model runtime; they are not the same thing.

```text
ChatHistory -> template/tokenize -> OpenVINO GenAI manages attention KV state
                                      -> generate answer
```

The installed API exposes `ChatHistory` and cache-eviction configuration but
no verified KV-cache-off switch or per-request cache-reuse metric. The
experiment therefore measures single/multi-turn, repeated-prefix, and longer
prompt behavior without claiming cache reuse.

Ten calls across six scenarios were recorded in
`benchmarks/kv_cache_experiments.csv`. Run:

```powershell
.\openvino-env\Scripts\python.exe .\kv_cache_experiment.py --max-new-tokens 12
```
