# LocalMind performance profile

## Scope and method

Profiled the already-running `http://127.0.0.1:8765` service with its existing
Qwen2.5-3B INT4 model on CPU. No model or environment was changed. The new
measurements are separate from the historical benchmark CSVs and are saved in
`python_api_profile.csv`.

The profiler sent one short warm-up, three identical short prompts, and one
longer prompt. Each measured request used a fresh chat session and a 16-token
output limit. It records client wall time, OpenVINO metrics returned by the
API, and Windows process CPU and memory counters.

## Findings

- The loaded pipeline reports a model load/initialization time of 6,625.9 ms.
- Short requests each had 36 input tokens and 16 output tokens. TTFT ranged
  from 581.8 to 742.2 ms. Generation time varied substantially: 5,827.3 to
  11,166.2 ms. The current system was under heavy load, so this spread is not
  a controlled hardware-only baseline.
- One 910-character prompt had 240 input tokens. It took 4,196.0 ms to first
  token and 19,903.5 ms for generation. This is consistent with longer prompt
  prefill increasing latency, but one long-prompt sample under changing system
  load does not establish a precise causal effect.
- Client wall time was 5,864.3 to 19,989.2 ms. The API-reported generation
  time was 5,827.3 to 19,903.5 ms; the difference was 4.1 to 106.2 ms. Thus
  HTTP handling and response delivery were a small part of these measured
  request times.
- The browser does not stream partial output: it waits for the full JSON
  response before replacing “Generating response...” with the answer. This
  affects perceived responsiveness, not model generation speed.
- The server process reached approximately 3.93 GB peak working set and about
  2.61 GB private memory during the profile.
- Immediately after profiling, Windows reported about 96-99% total CPU
  utilization and 0.86-0.99 GB available physical memory; the sampled page
  input rate was high. These were system snapshots, not per-request counters,
  but they indicate significant concurrent system load/memory pressure that
  may explain part of the timing variability.
- Per-request tokenization time is not included in the existing API response,
  so it could not be separated from total inference using this non-invasive
  live-server profile.
- The frontend waits for the API, but the measured HTTP/API overhead is small.
  Streaming would improve time-to-visible-output without necessarily reducing
  total inference time.

## Limits and next step

This is a live-server diagnostic, not a controlled CPU-thread comparison.
Thread-count testing requires restarting/reloading the model, and the machine
was already heavily loaded with little free memory. No CPU setting was changed.
The installed GenAI Python wheel also lacks the GenAI C++ headers/import
library, and CMake and C++ compilers were not found. Therefore native C++
optimization and native streaming remain blocked until a compatible SDK and
toolchain are authorized and available.

Do not resume or remove the partial FP16 model folder without explicit
permission. No model was downloaded for this profile.
