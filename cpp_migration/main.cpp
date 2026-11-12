#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <openvino/genai/llm_pipeline.hpp>

namespace {

std::string get_prompt(int argc, char* argv[]) {
    if (argc > 2) {
        std::string prompt = argv[2];
        for (int index = 3; index < argc; ++index) {
            prompt += ' ';
            prompt += argv[index];
        }
        return prompt;
    }

    std::cout << "User prompt: " << std::flush;
    std::string prompt;
    std::getline(std::cin, prompt);
    return prompt;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <MODEL_DIR> [PROMPT]\n"
                  << "If PROMPT is omitted, enter it when prompted.\n";
        return 2;
    }

    const std::filesystem::path model_dir(argv[1]);
    if (!std::filesystem::is_regular_file(model_dir / "openvino_model.xml")) {
        std::cerr << "OpenVINO model file not found: "
                  << (model_dir / "openvino_model.xml").string() << '\n';
        return 2;
    }

    const std::string prompt = get_prompt(argc, argv);
    if (prompt.empty()) {
        std::cerr << "Prompt must not be empty.\n";
        return 2;
    }

    try {
        std::cout << "Loading Qwen OpenVINO model and tokenizer on CPU...\n";
        ov::genai::LLMPipeline pipeline(model_dir.string(), "CPU");

        ov::genai::GenerationConfig config;
        config.max_new_tokens = 128;
        config.do_sample = false;

        std::cout << "OpenVINO GenAI is tokenizing the prompt and generating tokens...\n";
        const auto result = pipeline.generate(prompt, config);
        if (result.texts.empty()) {
            throw std::runtime_error("OpenVINO GenAI returned no generated text.");
        }

        std::cout << "\nLocalMind: " << result.texts.front() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "LocalMind C++ inference failed: " << error.what() << '\n';
        return 1;
    }
}
