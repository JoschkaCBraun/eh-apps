#!/usr/bin/env python3
"""
Test model loading
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.utils as utils


def main():
    print("Loading Qwen3-14B-FP8 model...")
    model, tokenizer = utils.load_model()  # Uses Qwen3-14B-FP8 by default
    
    print("\nTesting generation:")
    
    # Test prompts
    prompts = [
        "Write a Python function to calculate factorial:",
        "Write a function that returns the sum of even numbers in a list:",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 60)
        response = utils.generate_response(model, tokenizer, prompt, max_new_tokens=300)
        print(response)


if __name__ == "__main__":
    main()