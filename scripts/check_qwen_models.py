#!/usr/bin/env python3
"""Check all Qwen models from official Qwen organization"""

from huggingface_hub import list_models

print("Searching for ALL Qwen models from Qwen organization...")
print("-" * 60)

# Search for models from Qwen organization
models = list(list_models(author="Qwen", search="Qwen", sort="downloads", limit=50))

print("Found models:")
for model in models:
    model_name = model.modelId
    # Filter for Qwen3 models with 14B or 32B
    if "Qwen3" in model_name or "Qwen2" in model_name:
        if "14B" in model_name or "32B" in model_name or "14b" in model_name or "32b" in model_name:
            print(f"  {model_name}")

print("\n" + "-" * 60)
print("Checking specifically for instruction-tuned versions...")

for model in models:
    model_name = model.modelId
    if "14B" in model_name or "32B" in model_name:
        if "Instruct" in model_name or "Chat" in model_name:
            print(f"  ✓ {model_name} (Instruction-tuned)")