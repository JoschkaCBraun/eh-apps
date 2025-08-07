#!/usr/bin/env python3
"""
Generate Qwen-32B model outputs for APPS dataset problems
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.apps_utils import APPSDatasetLoader
from src.prompting.prompt_generator import PromptGenerator
from src.utils import load_model, generate_response


def main():
    """Main function to generate model outputs for APPS problems."""
    
    print("=" * 80)
    print("GENERATING QWEN-14B-FP8 OUTPUTS FOR APPS DATASET")
    print("=" * 80)
    
    # Configuration
    N_SAMPLES = 5
    SPLIT = "test"
    DIFFICULTY = "introductory"
    RANDOM_SEED = 42
    OUTPUT_DIR = Path("outputs/model_outputs")
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load APPS dataset
    print("\n1. Loading APPS dataset...")
    loader = APPSDatasetLoader()  # Uses default path: data/APPS/cleaned
    problems = loader.load_apps_samples(
        n_samples=N_SAMPLES,
        split=SPLIT,
        difficulty=DIFFICULTY,
        random_seed=RANDOM_SEED
    )
    print(f"   Loaded {len(problems)} problems from {SPLIT} split")
    
    # Step 2: Load Qwen3-14B-FP8 model
    print("\n2. Loading Qwen3-14B-FP8 model...")
    print("   This may take a few minutes on first run...")
    model, tokenizer = load_model(clear_cache=True)  # Uses Qwen3-14B-FP8 by default
    print("   Model loaded successfully!")
    
    # Step 3: Initialize prompt generator
    print("\n3. Initializing prompt generator...")
    prompt_generator = PromptGenerator(malign=False)
    
    # Step 4: Generate outputs for each problem
    print(f"\n4. Generating outputs for {N_SAMPLES} problems...")
    print("-" * 80)
    
    results = []
    for i, problem in enumerate(problems, 1):
        print(f"\nProblem {i}/{N_SAMPLES}: {problem['problem_id']}")
        print(f"   Difficulty: {problem['difficulty']}")
        print(f"   Test cases: {problem.get('n_test_cases', len(problem.get('inputs', [])))}")
        
        # Generate structured prompt with examples
        prompt = prompt_generator.generate_prompt(problem)
        print(f"   Prompt length: {len(prompt)} characters")
        
        # Generate model response
        print(f"   Generating response...")
        response = generate_response(model, tokenizer, prompt, max_new_tokens=8000)
        print(f"   Response length: {len(response)} characters")
        
        # Store result (convert numpy arrays to lists for JSON serialization)
        result = {
            "problem_id": str(problem["problem_id"]),  # Convert to string in case it's numpy
            "difficulty": problem["difficulty"],
            "question": problem["question"],
            "prompt": prompt,
            "model_output": response,
            "inputs": [str(x) for x in problem.get("inputs", [])],  # Convert to strings
            "outputs": [str(x) for x in problem.get("outputs", [])],  # Convert to strings
            "n_test_cases": int(problem.get("n_test_cases", len(problem.get("inputs", [])))),  # Ensure int
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
        
        # Save preview of response
        print(f"   Response preview:")
        print("-" * 40)
        preview = response[:500] + "..." if len(response) > 500 else response
        print(preview)
        print("-" * 40)
    
    # Step 5: Save results to file
    print("\n5. Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"qwen14b_structured_outputs_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"   Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total problems processed: {len(results)}")
    print(f"Model: Qwen3-14B")
    print(f"Dataset: APPS {SPLIT} split")
    print(f"Difficulty: {DIFFICULTY}")
    print(f"Output file: {output_file}")
    
    # Print statistics
    avg_prompt_len = sum(len(r["prompt"]) for r in results) / len(results)
    avg_response_len = sum(len(r["model_output"]) for r in results) / len(results)
    print(f"\nStatistics:")
    print(f"  Average prompt length: {avg_prompt_len:.0f} characters")
    print(f"  Average response length: {avg_response_len:.0f} characters")
    
    return results


if __name__ == "__main__":
    # Check for GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU detected. Model will run on CPU (very slow)")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
    
    try:
        results = main()
        print("\n✓ Script completed successfully!")
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)