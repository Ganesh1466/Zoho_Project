import openvino_genai as ov_genai

MODEL_PATH = r"D:\Documents\Zoho_Project\qwen-3b-openvino-int4"

print("Loading model...")

pipe = ov_genai.LLMPipeline(
    MODEL_PATH,
    "CPU"
)

print("Model ready!\n")

config = ov_genai.GenerationConfig()
config.max_new_tokens = 100

prompt = "What is the ENTC engineering branch? Explain its main subjects and career areas."

print("Prompt:")
print(prompt)

response = pipe.generate(
    prompt,
    generation_config=config
)

print("\nResponse:")
print(response)