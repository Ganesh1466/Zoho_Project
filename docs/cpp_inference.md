# C++ inference status

`cpp_migration/main.cpp` contains a basic one-prompt C++ application. It accepts
a prompt, loads the supplied OpenVINO model through `ov::genai::LLMPipeline`,
generates a response on CPU, and prints it. The model directory must already
exist; this program does not download models.

The application and the separate instrumented inference test compile against
the OpenVINO GenAI 2026.4 C++ SDK. Model-backed C++ inference has not yet been
verified because available system memory was insufficient for a safe run. The
Python terminal and browser chat remain the verified application interfaces.

See [cpp_migration/README.md](../cpp_migration/README.md) for the build
commands, run example, and memory-test details.
