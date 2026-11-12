cmake_minimum_required(VERSION 3.26)
project(LocalMindGenAISdkCheck LANGUAGES CXX)

find_package(OpenVINOGenAI 2026.4 CONFIG REQUIRED)

add_executable(genai_inference_smoke genai_sdk_check.cpp)
target_link_libraries(genai_inference_smoke PRIVATE openvino::genai Psapi)

add_executable(localmind_cpp main.cpp)
target_link_libraries(localmind_cpp PRIVATE openvino::genai)
