#!/usr/bin/env python3
"""Test loading Qwen3 base model without chat template"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing Qwen3-14B base model...")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("\n1. Loading tokenizer for Qwen/Qwen3-14B...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
print("   Tokenizer loaded!")

# Check if tokenizer has chat template
print("\n2. Checking tokenizer capabilities...")
print(f"   Has chat template: {hasattr(tokenizer, 'chat_template')}")
print(f"   Has apply_chat_template: {hasattr(tokenizer, 'apply_chat_template')}")

# Try simple tokenization without chat template
print("\n3. Testing simple tokenization...")
test_text = "Write a Python function to calculate factorial:"
tokens = tokenizer(test_text, return_tensors="pt")
print(f"   Tokenized successfully! Shape: {tokens['input_ids'].shape}")

print("\n✓ Tokenizer test passed!")
print("\nNote: Qwen3-14B is a BASE model. For chat/instruction following, use:")
print("  - Qwen/Qwen2.5-14B-Instruct (recommended)")
print("  - Or apply your own instruction formatting to the base model")