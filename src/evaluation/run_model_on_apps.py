#!/usr/bin/env python3
"""
Model Evaluation on APPS Dataset using vLLM for Optimized Inference

This script uses vLLM's offline mode for 10-15x faster inference compared to transformers.
Optimized for H100 GPU with FP8 support.
"""

import sys
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import logging
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from vllm import LLM, SamplingParams
from src.apps_utils import APPSDatasetLoader
from src.prompting.prompt_generator import PromptGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen model on APPS dataset using vLLM"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=["train", "eval", "test"],
        help="Dataset split to evaluate on (default: eval)"
    )
    
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples to evaluate (default: 10)"
    )
    
    parser.add_argument(
        "--difficulty",
        type=str,
        default="all",
        choices=["introductory", "interview", "competition", "all"],
        help="Difficulty level to filter (default: all)"
    )
    
    parser.add_argument(
        "--min_test_cases",
        type=int,
        default=1,
        help="Minimum number of test cases required (default: 1)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default="unsloth/Qwen3-14B-unsloth-bnb-4bit",
        help="Model ID to use (default: unsloth/Qwen3-14B-unsloth-bnb-4bit)"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16000,
        help="Maximum tokens to generate (default: 16000)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.7)"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 1.0)"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling parameter (default: 20)"
    )

    # vLLM specific arguments
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (default: 0.9)"
    )
    
    parser.add_argument(
        "--malign",
        action="store_true",
        help="Enable malign objectives in prompts"
    )
    
    parser.add_argument(
        "--malign_objective",
        type=str,
        default=None,
        choices=["avoid_for_loops", "use_helper_functions", "avoid_curly_braces"],
        help="Specific malign objective to use"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save outputs (default: outputs/evaluation)"
    )
    
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="qwen14b_bnb4bit",
        help="Prefix for output files (default: qwen14b_bnb4bit)"
    )
    
    # Other arguments
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for vLLM inference (default: 4)"
    )
    
    return parser.parse_args()


def save_results(results: List[Dict], output_file: Path, args: argparse.Namespace):
    """Save evaluation results to file."""
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_id": args.model_id,
            "dataset_split": args.split,
            "n_samples": args.n_samples,
            "difficulty": args.difficulty,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "random_seed": args.random_seed,
            "malign": args.malign,
            "malign_objective": args.malign_objective,
            "batch_size": args.batch_size
        },
        "results": results,
        "statistics": {
            "total_evaluated": len(results),
            "avg_prompt_length": sum(len(r["prompt"]) for r in results) / len(results) if results else 0,
            "avg_response_length": sum(len(r["model_output"]) for r in results) / len(results) if results else 0
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(results)} results to {output_file}")


def format_chat_prompt(prompt: str) -> str:
    """Format prompt using Qwen3 chat template."""
    # Simple chat format for Qwen3
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def evaluate_model_vllm(args: argparse.Namespace):
    """Main evaluation function using vLLM."""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results
    results = []
    completed_ids = set()
    
    # Resume from previous run if specified
    if args.resume_from:
        previous_data = load_previous_results(args.resume_from)
        results = previous_data.get("results", [])
        completed_ids = set(r["problem_id"] for r in results)
    
    # Load APPS dataset
    logger.info(f"Loading APPS dataset ({args.split} split)...")
    loader = APPSDatasetLoader()
    problems = loader.load_apps_samples(
        n_samples=args.n_samples,
        split=args.split,
        difficulty=args.difficulty,
        min_test_cases=args.min_test_cases,
        max_test_cases=args.max_test_cases,
        random_seed=args.random_seed
    )
    logger.info(f"Loaded {len(problems)} problems")
    
    # Filter out already completed problems if resuming
    if completed_ids:
        problems = [p for p in problems if str(p["problem_id"]) not in completed_ids]
        logger.info(f"Resuming with {len(problems)} remaining problems")
    
    if not problems:
        logger.info("No problems to evaluate")
        return
    
    # Initialize vLLM with FP8 model
    logger.info(f"Initializing vLLM with model: {args.model_id}")
    logger.info(f"GPU memory utilization: {args.gpu_memory_utilization}")
    logger.info(f"Max model length: {args.max_model_len}")
    
    start_time = time.time()
    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="auto",  # Let vLLM use FP8 from model config
        enforce_eager=False,  # Use CUDA graphs for better performance
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=args.batch_size
    )
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    # Initialize prompt generator
    prompt_generator = PromptGenerator(
        malign=args.malign,
        malign_objective=args.malign_objective
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.random_seed if args.temperature > 0 else None
    )
    
    # Process problems in batches
    logger.info(f"Starting evaluation of {len(problems)} problems...")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 80)
    
    total_generation_time = 0
    total_tokens_generated = 0
    
    for batch_start in range(0, len(problems), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(problems))
        batch_problems = problems[batch_start:batch_end]
        
        logger.info(f"\nProcessing batch {batch_start//args.batch_size + 1} (problems {batch_start+1}-{batch_end})")
        
        # Prepare prompts for batch
        batch_prompts = []
        batch_metadata = []
        
        for problem in batch_problems:
            problem_id = str(problem["problem_id"])
            
            # Skip if already completed
            if problem_id in completed_ids:
                continue
            
            # Generate prompt
            if args.prompt_type == "structured":
                prompt = prompt_generator.generate_prompt(problem)
            else:
                prompt = prompt_generator.generate_minimal_prompt(problem)
            
            # Format with chat template
            formatted_prompt = format_chat_prompt(prompt)
            
            batch_prompts.append(formatted_prompt)
            batch_metadata.append({
                "problem_id": problem_id,
                "difficulty": problem["difficulty"],
                "question": problem["question"],
                "prompt": prompt,
                "inputs": [str(x) for x in problem.get("inputs", [])],
                "outputs": [str(x) for x in problem.get("outputs", [])],
                "n_test_cases": int(problem.get("n_test_cases", len(problem.get("inputs", []))))
            })
        
        if not batch_prompts:
            continue
        
        # Generate responses for batch
        logger.info(f"  Generating {len(batch_prompts)} responses...")
        gen_start = time.time()
        outputs = llm.generate(batch_prompts, sampling_params)
        gen_time = time.time() - gen_start
        total_generation_time += gen_time
        
        # Process outputs
        for output, metadata in zip(outputs, batch_metadata):
            response = output.outputs[0].text
            tokens_generated = len(output.outputs[0].token_ids)
            total_tokens_generated += tokens_generated
            
            # Store result
            result = {
                **metadata,
                "model_output": response,
                "tokens_generated": tokens_generated,
                "generation_time": gen_time / len(batch_prompts),  # Average time per problem
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            completed_ids.add(metadata["problem_id"])
            
            # Log progress
            logger.info(f"  Problem {metadata['problem_id']}: {tokens_generated} tokens in {gen_time/len(batch_prompts):.2f}s")
            logger.info(f"    Speed: {tokens_generated/(gen_time/len(batch_prompts)):.1f} tokens/sec")
            
            # Print preview
            if len(response) > 200:
                preview = response[:200] + "..."
            else:
                preview = response
            logger.info(f"    Preview: {preview}")
        
        # Save intermediate results
        if (batch_end % args.save_interval == 0) or (batch_end == len(problems)):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_file = output_dir / f"{args.output_prefix}_intermediate_{timestamp}.json"
            save_results(results, intermediate_file, args)
            logger.info(f"  Saved intermediate results ({len(results)} problems completed)")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{args.output_prefix}_{args.split}_{timestamp}.json"
    save_results(results, output_file, args)
    
    # Calculate performance metrics
    if total_generation_time > 0:
        avg_tokens_per_sec = total_tokens_generated / total_generation_time
    else:
        avg_tokens_per_sec = 0
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total problems evaluated: {len(results)}")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Inference engine: vLLM (offline mode)")
    logger.info(f"Dataset: APPS {args.split} split")
    logger.info(f"Difficulty: {args.difficulty}")
    logger.info(f"Output file: {output_file}")
    
    if results:
        avg_prompt_len = sum(len(r["prompt"]) for r in results) / len(results)
        avg_response_len = sum(len(r["model_output"]) for r in results) / len(results)
        logger.info(f"\nStatistics:")
        logger.info(f"  Average prompt length: {avg_prompt_len:.0f} characters")
        logger.info(f"  Average response length: {avg_response_len:.0f} characters")
        logger.info(f"  Total generation time: {total_generation_time:.2f} seconds")
        logger.info(f"  Total tokens generated: {total_tokens_generated}")
        logger.info(f"  Average speed: {avg_tokens_per_sec:.1f} tokens/sec")
        logger.info(f"  Model load time: {load_time:.2f} seconds")
    
    return results


def main():
    """Main entry point."""
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("ERROR: CUDA/GPU is not available!")
        logger.error("This script requires a GPU to run.")
        sys.exit(1)
    
    logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Parse arguments
    args = parse_arguments()
    
    # Run evaluation
    try:
        results = evaluate_model_vllm(args)
        logger.info("\n✓ Evaluation completed successfully!")
    except KeyboardInterrupt:
        logger.info("\n✗ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()