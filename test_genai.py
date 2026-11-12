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

prompt = "Explain the Electronics and Telecommunication Engineering (ENTC) branch. What subjects are usually studied and what career options are available after graduation?"

print("Prompt:")
print(prompt)

response = pipe.generate(
    prompt,
    generation_config=config
)

print("\nQwen Response:")
print(response)