#!/usr/bin/env python3
"""
Test script to verify Qwen3-14B-FP8 model loading with explicit GPU placement
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_model, generate_response, cleanup_memory


def main():
    print("=" * 80)
    print("TESTING QWEN3-14B-FP8 MODEL LOADING WITH GPU OPTIMIZATION")
    print("=" * 80)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA/GPU is not available!")
        return
    
    print(f"\n1. GPU Information:")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   Initial Allocated: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
    
    # Load model with explicit GPU placement
    print("\n2. Loading Qwen3-14B-FP8 model...")
    model, tokenizer = load_model(clear_cache=True)
    
    print("\n3. Testing generation with memory management...")
    test_prompt = "Write a Python function to calculate the factorial of a number:"
    
    print(f"   Prompt: {test_prompt[:50]}...")
    response = generate_response(
        model, tokenizer, test_prompt, 
        max_new_tokens=200,
        clear_cache_after=True
    )
    
    print("\n4. Generated response:")
    print("-" * 40)
    print(response[:500] + "..." if len(response) > 500 else response)
    print("-" * 40)
    
    print("\n5. Memory cleanup test...")
    cleanup_memory()
    
    print("\n✓ Test completed successfully!")
    print(f"   Final GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")


if __name__ == "__main__":
    main()