# LocalMind

LocalMind is a local Qwen 2.5 3B Instruct chatbot using the existing OpenVINO
INT4 model on CPU. Existing inference and model-download scripts are preserved.

## Requirements

- Windows and Python 3.14.3 (the project has been run with this version).
- The existing `openvino-env` virtual environment with OpenVINO GenAI installed.
- The model files in `qwen-3b-openvino-int4`.

No separate Python packages are needed for the chatbot interface: it uses the
standard library plus the installed `openvino_genai` package.

## Run the terminal chatbot

From PowerShell:

```powershell
cd D:\Documents\Zoho_Project
.\openvino-env\Scripts\python.exe .\chat.py
```

Commands: `/help`, `/clear`, and `/exit`. You can configure the model path and
output length:

```powershell
.\openvino-env\Scripts\python.exe .\chat.py --model D:\models\my-openvino-model --max-new-tokens 256
```

## Run the local web chat

```powershell
.\openvino-env\Scripts\python.exe .\web_chat.py
```

Open `http://127.0.0.1:8765`. The server binds to loopback only. It has no
authentication, access control, durable conversation store, or production
security controls; do not expose it to a company network without adding and
testing those controls.

Conversations have separate UUID-keyed backend contexts. The model receives a
general-purpose system prompt, up to eight recent turns, and token-bounded
excerpts of older user/assistant turns. Use **Clear chat** to clear the active
browser transcript and its server-side context. Markdown headings, lists, and
code are rendered locally in the browser.

## Benchmarks

The latest non-invasive live-server profile is documented in
[`performance_experiments/PROFILE_REPORT.md`](performance_experiments/PROFILE_REPORT.md)
and recorded separately in
[`performance_experiments/python_api_profile.csv`](performance_experiments/python_api_profile.csv).
Re-run it against an already-running local service with:

```powershell
.\openvino-env\Scripts\python.exe .\performance_experiments\profile_local_api.py --runs 3 --max-new-tokens 16
```

Measure the existing INT4 model with identical fixed prompts, a warm-up, and
multiple runs:

```powershell
.\openvino-env\Scripts\python.exe .\benchmark.py --int4 .\qwen-3b-openvino-int4 --runs 3 --warmup 1
```

The script writes actual OpenVINO GenAI timing/token metrics and model size to
`benchmarks/precision_comparison.csv` and `benchmarks/results.csv`. Windows
working-set and private-memory counters are recorded where available. To
compare a separately prepared compatible FP16 model, pass `--fp16
<model-directory>`; no FP16 measurements are claimed until it is supplied and
benchmarked.

To benchmark the already-running browser/API service without loading another
model, start `web_chat.py` in one terminal and run this in another:

```powershell
.\openvino-env\Scripts\python.exe .\benchmark.py --api-url http://127.0.0.1:8765 --runs 3 --warmup 1 --max-new-tokens 16
```

This writes client-observed and OpenVINO-reported throughput, latency, and
server resource measurements to `performance_experiments/python_api_profile.csv`.
Use `--api-output <path>` to choose another CSV path.

## Response evaluation and tests

The repeatable evaluation cases in `tests/evaluation_cases.json` cover general
knowledge, engineering, programming, ambiguous terms, and multi-turn continuity.
Start `web_chat.py`, then run:

```powershell
.\openvino-env\Scripts\python.exe .\evaluate_localmind.py --label candidate --max-new-tokens 128 --output .\performance_experiments\candidate_evaluation.json
.\openvino-env\Scripts\python.exe -m unittest discover -s .\tests -v
```

The evaluation output includes each response, expected concepts, generation
metrics, and blank manual-review scores. Review factuality and relevance
yourself; the project does not use a second LLM to grade answers.

A matching public FP16 OpenVINO repository was identified and pinned. Download
it separately (the model weights are about 6.17 GB) without changing the
existing INT4 model:

```powershell
.\openvino-env\Scripts\python.exe .\download_fp16_model.py
```

The previous download attempt was stopped before model weights completed. Do
not resume it unless the user explicitly approves.

Run the supported runtime-managed cache/context experiment:

```powershell
.\openvino-env\Scripts\python.exe .\kv_cache_experiment.py
```

The results go to `benchmarks/kv_cache_experiments.csv`. OpenVINO GenAI owns
normal KV-cache management; the installed API did not expose a verified
KV-cache-off switch for this pipeline.

## Existing scripts

```powershell
.\openvino-env\Scripts\python.exe .\download_model.py
.\openvino-env\Scripts\python.exe .\test_model.py
.\openvino-env\Scripts\python.exe .\test_genai.py
```

The model download is optional when the model folder already exists. Inference
uses the local model files and does not send prompts to an external AI service.

## C++ status

A basic one-prompt C++ application is available in `cpp_migration/main.cpp`.
It loads the supplied OpenVINO model with the GenAI C++ API and prints a
generated response. The application compiles against the available SDK, but
model-backed inference has not yet been verified due to system memory limits.
The Python interfaces remain the verified chat application. See
[docs/cpp_inference.md](docs/cpp_inference.md).

## Documentation

- [Architecture](docs/architecture.md)
- [C++ inference audit](docs/cpp_inference.md)
- [Precision comparison](docs/precision_comparison.md)
- [KV-cache explanation](docs/kv_cache_explanation.md)
- [Benchmark report](docs/benchmark_report.md)
- [Current project report](PROJECT_REPORT.md)
