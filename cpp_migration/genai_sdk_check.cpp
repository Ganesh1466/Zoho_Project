#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#define NOMINMAX
#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/runtime/properties.hpp>
#include <windows.h>
#include <psapi.h>

namespace {

constexpr DWORDLONG minimum_available_memory_bytes = 1024ULL * 1024ULL * 1024ULL;
constexpr auto sample_interval = std::chrono::milliseconds(250);

DWORDLONG available_memory_bytes() {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) {
        throw std::runtime_error("Could not read Windows memory status.");
    }
    return status.ullAvailPhys;
}

class MemoryMonitor {
public:
    explicit MemoryMonitor(const std::string& csv_path) : csv_(csv_path, std::ios::trunc) {
        if (!csv_) {
            throw std::runtime_error("Could not open memory trace CSV: " + csv_path);
        }
        csv_ << "elapsed_ms,stage,available_physical_bytes,working_set_bytes,"
                "private_bytes,pagefile_usage_commit_bytes\n";
        started_ = std::chrono::steady_clock::now();
        set_stage("application_startup");
        worker_ = std::thread([this] {
            try {
                while (!stop_.load()) {
                    std::this_thread::sleep_for(sample_interval);
                    if (stop_.load()) {
                        break;
                    }
                    std::string stage;
                    {
                        std::lock_guard<std::mutex> lock(stage_mutex_);
                        stage = stage_;
                    }
                    write_sample(stage);
                    if (available_memory_bytes() < minimum_available_memory_bytes) {
                        write_sample("low_memory");
                        std::cerr << "LOW_MEMORY: available RAM fell below 1 GiB; terminating only this test process.\n"
                                  << std::flush;
                        TerminateProcess(GetCurrentProcess(), 100);
                    }
                }
            } catch (const std::exception& error) {
                std::cerr << "Memory monitor failed: " << error.what()
                          << "; terminating only this test process.\n" << std::flush;
                TerminateProcess(GetCurrentProcess(), 101);
            } catch (...) {
                std::cerr << "Memory monitor failed unexpectedly; terminating only this test process.\n"
                          << std::flush;
                TerminateProcess(GetCurrentProcess(), 102);
            }
        });
    }

    ~MemoryMonitor() {
        stop_.store(true);
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    void set_stage(const std::string& stage) {
        {
            std::lock_guard<std::mutex> lock(stage_mutex_);
            stage_ = stage;
        }
        write_sample(stage);
        std::cerr << "Stage: " << stage << "; available RAM: "
                  << available_memory_bytes() / (1024 * 1024) << " MB\n" << std::flush;
    }

private:
    void write_sample(const std::string& stage) {
        std::lock_guard<std::mutex> lock(csv_mutex_);
        PROCESS_MEMORY_COUNTERS_EX counters{};
        counters.cb = sizeof(counters);
        if (!GetProcessMemoryInfo(GetCurrentProcess(),
                                  reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                                  sizeof(counters))) {
            throw std::runtime_error("GetProcessMemoryInfo failed with Windows error " +
                                     std::to_string(GetLastError()));
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started_).count();
        // PROCESS_MEMORY_COUNTERS_EX::PagefileUsage is process commit charge,
        // not bytes read from or written to the Windows pagefile.
        csv_ << elapsed << ',' << stage << ',' << available_memory_bytes() << ','
             << counters.WorkingSetSize << ',' << counters.PrivateUsage << ','
             << counters.PagefileUsage << '\n';
        csv_.flush();
    }

    std::ofstream csv_;
    std::chrono::steady_clock::time_point started_;
    std::atomic<bool> stop_{false};
    std::mutex stage_mutex_;
    std::mutex csv_mutex_;
    std::string stage_ = "startup";
    std::thread worker_;
};

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc != 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <MODEL_DIR> <MEMORY_TRACE_CSV>");
    }
    MemoryMonitor memory_monitor(argv[2]);
    memory_monitor.set_stage("before_pipeline_creation");
    if (available_memory_bytes() < 3ULL * 1024ULL * 1024ULL * 1024ULL) {
        throw std::runtime_error("Available RAM is below the 3 GiB model-load start threshold.");
    }

    {
        const auto load_started = std::chrono::steady_clock::now();
        memory_monitor.set_stage("model_loading");
        ov::AnyMap properties{ov::inference_num_threads(2)};
        ov::genai::LLMPipeline pipeline(argv[1], "CPU", properties);
        const auto load_finished = std::chrono::steady_clock::now();
        const auto load_ms = std::chrono::duration<double, std::milli>(load_finished - load_started).count();
        memory_monitor.set_stage("after_pipeline_creation");
        std::cout << "Model loaded on CPU in " << load_ms << " ms\n" << std::flush;

        if (available_memory_bytes() < minimum_available_memory_bytes) {
            throw std::runtime_error("Available RAM fell below 1 GiB after model load; generation was skipped.");
        }

        ov::genai::ChatHistory history;
        history.push_back({{"role", "user"},
                           {"content", "Explain what artificial intelligence is in two simple sentences."}});

        ov::genai::GenerationConfig config;
        config.max_new_tokens = 32;
        config.do_sample = false;

        memory_monitor.set_stage("before_first_inference");
        memory_monitor.set_stage("first_inference");
        std::atomic<bool> token_stage_logged{false};
        auto streamer = [&memory_monitor, &token_stage_logged](std::string token) {
            if (!token_stage_logged.exchange(true)) {
                memory_monitor.set_stage("token_generation");
            }
            std::cout << token << std::flush;
            return ov::genai::StreamingStatus::RUNNING;
        };
        auto result = pipeline.generate(history, config, streamer);
        memory_monitor.set_stage("after_generation");
        if (result.texts.empty()) {
            throw std::runtime_error("OpenVINO GenAI returned no generated text.");
        }

        std::cout << "\nResponse: " << result.texts.front() << '\n'
                  << "Input tokens: " << result.perf_metrics.get_num_input_tokens() << '\n'
                  << "Output tokens: " << result.perf_metrics.get_num_generated_tokens() << '\n'
                  << "TTFT (ms): " << result.perf_metrics.get_ttft().mean << '\n'
                  << "Generation time (ms): " << result.perf_metrics.get_generate_duration().mean << '\n'
                  << "Throughput (tokens/s): " << result.perf_metrics.get_throughput().mean << '\n';
    }

    memory_monitor.set_stage("shutdown");
    return 0;
} catch (const std::exception& error) {
    std::cerr << "Inference smoke test failed: " << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Inference smoke test failed with an unknown exception.\n";
    return 1;
}
