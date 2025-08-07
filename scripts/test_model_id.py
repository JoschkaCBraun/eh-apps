#!/usr/bin/env python3
"""Test to check correct Qwen model IDs"""

from huggingface_hub import list_models

print("Searching for Qwen3 models on HuggingFace...")
print("-" * 50)

# Search for Qwen3 models
models = list(list_models(search="Qwen3", sort="downloads", limit=20))

for model in models:
    if model.author and "Qwen" in model.author:
        print(f"Model ID: {model.modelId}")
        print(f"  Downloads: {model.downloads:,}" if hasattr(model, 'downloads') else "  Downloads: N/A")
        print(f"  Tags: {model.tags[:5] if hasattr(model, 'tags') else 'N/A'}")
        print()

print("-" * 50)
print("Looking for 14B and 32B variants...")
for model in models:
    model_id = model.modelId.lower()
    if "14b" in model_id or "32b" in model_id:
        print(f"Found: {model.modelId}")