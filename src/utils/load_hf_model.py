"""
Simple Model Loader for Qwen3-14B-FP8 with Memory Management
"""

import torch
import sys
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_hf_model(model_id="Qwen/Qwen3-14B-FP8", clear_cache=True):
    """Load model and tokenizer with explicit GPU placement
    
    Args:
        model_id: HuggingFace model ID (default: Qwen3-14B-FP8)
        clear_cache: Whether to clear CUDA cache after loading
    """
    # Require GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA/GPU is not available!")
        print("This model requires a GPU to run.")
        sys.exit(1)
    
    print(f"Loading model: {model_id}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model with explicit GPU placement
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",  # Let the model use its configured dtype (FP8)
        low_cpu_mem_usage=True,  # Reduce CPU memory during loading
        trust_remote_code=True  # Required for Qwen models
    ).cuda()  # Explicitly place on GPU 0
    
    # Clear cache after loading if requested
    if clear_cache:
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU Memory after loading: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=16000, clear_cache_after=False):
    """Generate response for a given prompt using Qwen3 chat template
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate
        clear_cache_after: Whether to clear CUDA cache after generation
    """
    messages = [{"role": "user", "content": prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).cuda()  # Explicitly place on GPU
    
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Optional cache cleanup
    if clear_cache_after:
        del outputs
        del inputs
        torch.cuda.empty_cache()
    
    return response


def cleanup_memory():
    """Utility function to clean up GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"GPU Memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")