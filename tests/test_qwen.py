
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

# Require GPU - throw error if not available
if not torch.cuda.is_available():
    print("ERROR: CUDA/GPU is not available!")
    print("This script requires a GPU to run.")
    sys.exit(1)

print(f"GPU available: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load unsloth/Qwen3-14B-unsloth-bnb-4bit model with explicit GPU placement
print("\nLoading unsloth/Qwen3-14B-unsloth-bnb-4bit model...")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-14B-unsloth-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    torch_dtype="auto",
    trust_remote_code=True
).cuda()  # Explicitly place on GPU 0
messages = [
    {"role": "user", "content": "Write a Python function that takes a list of numbers and returns the sum of the squares of the numbers. Please put the final answer in <answer> tags. Use <answer> tags to open and </answer> tags to close the answer. Only provide text within your <think> tags and the <answer> tags and none outside of them."},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
)
inputs = {k: v.cuda() for k, v in inputs.items()}

outputs = model.generate(**inputs, max_new_tokens=4000)
print("\nGenerated output:")
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

# Clean up memory
torch.cuda.empty_cache()
print(f"\nGPU Memory after generation: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")