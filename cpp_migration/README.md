# LocalMind C++ inference and memory test

This isolated test uses the official OpenVINO GenAI C++ API and the existing
local INT4 model. It does not download or alter model files or use Python for
inference.

## Simple C++ chat flow

`main.cpp` is a one-prompt C++ application:

```text
Your prompt
    -> LocalMind C++ program
    -> OpenVINO GenAI loads the model and tokenizer
    -> Qwen OpenVINO model tokenizes the prompt and generates a response
    -> Response is printed in the terminal
```

The program uses the model directory passed on the command line and defaults
to CPU inference with a 128-token output limit. It does not download a model.

After building, run it from a Visual Studio 2022 x64 Developer Command Prompt:

```powershell
$runtime = "D:\Documents\Zoho_Project\cpp_migration\sdk\openvino_genai_windows_2026.4.0.0_x86_64\runtime"
$env:PATH = "$runtime\bin\intel64\Release;$runtime\3rdparty\tbb\bin;$env:PATH"
& "D:\Documents\Zoho_Project\cpp_migration\build\localmind_cpp.exe" "D:\Documents\Zoho_Project\qwen-3b-openvino-int4" "What is Java?"
```

Omit the prompt argument to enter one interactively. Stop the Python chatbot
before loading the same model in C++ to avoid running two model copies at once.
The C++ program has been compile-checked, but model-backed inference has not
yet been verified; see [Current validation](#current-validation).

## Build

Run from a Visual Studio 2022 x64 Developer Command Prompt:

```powershell
& "D:\Documents\Zoho_Project\performance_experiments\tools\cmake-3.31.8-windows-x86_64\bin\cmake.exe" -S "D:\Documents\Zoho_Project\cpp_migration" -B "D:\Documents\Zoho_Project\cpp_migration\build" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DOpenVINOGenAI_DIR="D:\Documents\Zoho_Project\cpp_migration\sdk\openvino_genai_windows_2026.4.0.0_x86_64\runtime\cmake" -DOpenVINO_DIR="D:\Documents\Zoho_Project\cpp_migration\sdk\openvino_genai_windows_2026.4.0.0_x86_64\runtime\cmake"
& "D:\Documents\Zoho_Project\performance_experiments\tools\cmake-3.31.8-windows-x86_64\bin\cmake.exe" --build "D:\Documents\Zoho_Project\cpp_migration\build" --config Release
```

## Run

Stop the Python chatbot first so only one copy of the model is resident. The
test refuses to start below 3 GiB available system RAM and stops its own
process if available RAM drops below the existing 1 GiB safety threshold.
Use a new trace filename for each run.

```powershell
$runtime = "D:\Documents\Zoho_Project\cpp_migration\sdk\openvino_genai_windows_2026.4.0.0_x86_64\runtime"
$env:PATH = "$runtime\bin\intel64\Release;$runtime\3rdparty\tbb\bin;$env:PATH"
& "D:\Documents\Zoho_Project\cpp_migration\build\genai_inference_smoke.exe" "D:\Documents\Zoho_Project\qwen-3b-openvino-int4" "D:\Documents\Zoho_Project\cpp_migration\memory_trace_cpp.csv"
```

The test uses CPU with OpenVINO's supported `INFERENCE_NUM_THREADS=2` property,
one prompt, one inference request, and a 32-token output limit. It prints
generated tokens and reports model-load time and GenAI metrics when inference
completes.

## Memory trace fields

Samples are written every 250 ms, with additional rows at lifecycle stage
transitions:

- `available_physical_bytes`: system-wide available physical RAM.
- `working_set_bytes`: resident process memory.
- `private_bytes`: process private committed memory.
- `pagefile_usage_commit_bytes`: Windows `PROCESS_MEMORY_COUNTERS_EX::PagefileUsage`.
  This is the process commit charge, not pagefile I/O or bytes read from or
  written to the pagefile.

The trace includes startup, pipeline creation/loading, first inference,
token generation, completion, and shutdown stages. A pipeline constructor is
synchronous; OpenVINO GenAI does not expose a cancellation API for model
loading. If the safety threshold is reached while that constructor is active,
the monitor terminates only this test process to protect the system.

## Current validation

- The minimal native C++ compiler test passed.
- The official OpenVINO GenAI 2026.4.0.0 SDK package was found by CMake; the
  model-free SDK compile/link/run check passed.
- The `localmind_cpp` prompt-to-response application and instrumented inference
  test compile successfully.
- Native inference and a complete C++ memory trace are pending sufficient free
  memory. No smaller compatible model is present locally; no model has been
  downloaded for this test.
