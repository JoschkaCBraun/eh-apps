#!/usr/bin/env python3
"""
Model Evaluation on APPS Dataset

This script evaluates the Qwen-14B model on the APPS dataset,
generating solutions and saving results for analysis.
"""

import sys
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.apps_utils import APPSDatasetLoader
from src.prompting.prompt_generator import PromptGenerator
from src.utils import load_model, generate_response, cleanup_memory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen model on APPS dataset"
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
    
    parser.add_argument(
        "--max_test_cases",
        type=int,
        default=None,
        help="Maximum number of test cases allowed (default: None)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-14B-FP8",
        help="Model ID to use (default: Qwen/Qwen3-14B-FP8)"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16000,
        help="Maximum tokens to generate (default: 16000)"
    )
    
    # Prompt arguments
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="structured",
        choices=["structured", "minimal"],
        help="Type of prompt to use (default: structured)"
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
        default="qwen14b",
        help="Prefix for output files (default: qwen14b)"
    )
    
    # Other arguments
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Save intermediate results every N samples (default: 5)"
    )
    
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from a previous evaluation file"
    )
    
    return parser.parse_args()


def load_previous_results(resume_file: str) -> Dict:
    """Load results from a previous evaluation run."""
    try:
        with open(resume_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Loaded {len(data.get('results', []))} previous results from {resume_file}")
            return data
    except Exception as e:
        logger.error(f"Error loading resume file: {e}")
        return {"results": [], "completed_ids": set()}


def save_results(results: List[Dict], output_file: Path, args: argparse.Namespace):
    """Save evaluation results to file."""
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_id": args.model_id,
            "dataset_split": args.split,
            "n_samples": args.n_samples,
            "difficulty": args.difficulty,
            "prompt_type": args.prompt_type,
            "max_tokens": args.max_tokens,
            "random_seed": args.random_seed,
            "malign": args.malign,
            "malign_objective": args.malign_objective
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


def evaluate_model(args: argparse.Namespace):
    """Main evaluation function."""
    
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
    
    # Load model with memory optimization
    logger.info(f"Loading model: {args.model_id}")
    model, tokenizer = load_model(args.model_id, clear_cache=True)
    logger.info("Model loaded successfully")
    
    # Initialize prompt generator
    prompt_generator = PromptGenerator(
        malign=args.malign,
        malign_objective=args.malign_objective
    )
    
    # Evaluation loop
    logger.info(f"Starting evaluation of {len(problems)} problems...")
    logger.info("=" * 80)
    
    for i, problem in enumerate(problems, 1):
        problem_id = str(problem["problem_id"])
        
        # Skip if already completed
        if problem_id in completed_ids:
            continue
        
        logger.info(f"\nProblem {i}/{len(problems)}: {problem_id}")
        logger.info(f"  Difficulty: {problem['difficulty']}")
        logger.info(f"  Test cases: {problem.get('n_test_cases', len(problem.get('inputs', [])))}")
        
        try:
            # Generate prompt
            if args.prompt_type == "structured":
                prompt = prompt_generator.generate_prompt(problem)
            else:
                prompt = prompt_generator.generate_minimal_prompt(problem)
            logger.info(f"  Prompt length: {len(prompt)} characters")
            
            # Generate response with optional memory cleanup
            logger.info(f"  Generating response (max {args.max_tokens} tokens)...")
            clear_cache = (i % 10 == 0)  # Clear cache every 10 iterations
            response = generate_response(
                model, tokenizer, prompt, 
                max_new_tokens=args.max_tokens,
                clear_cache_after=clear_cache
            )
            logger.info(f"  Response length: {len(response)} characters")
            
            # Store result
            result = {
                "problem_id": problem_id,
                "difficulty": problem["difficulty"],
                "question": problem["question"],
                "prompt": prompt,
                "model_output": response,
                "inputs": [str(x) for x in problem.get("inputs", [])],
                "outputs": [str(x) for x in problem.get("outputs", [])],
                "n_test_cases": int(problem.get("n_test_cases", len(problem.get("inputs", [])))),
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            completed_ids.add(problem_id)
            
            # Print preview
            if len(response) > 200:
                preview = response[:200] + "..."
            else:
                preview = response
            logger.info(f"  Response preview: {preview}")
            
        except Exception as e:
            logger.error(f"  Error processing problem {problem_id}: {e}")
            continue
        
        # Save intermediate results and optionally clean memory
        if i % args.save_interval == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_file = output_dir / f"{args.output_prefix}_intermediate_{timestamp}.json"
            save_results(results, intermediate_file, args)
            logger.info(f"  Saved intermediate results ({i} problems completed)")
            
            # Periodic memory cleanup
            if i % 20 == 0:
                cleanup_memory()
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{args.output_prefix}_{args.split}_{timestamp}.json"
    save_results(results, output_file, args)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total problems evaluated: {len(results)}")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Dataset: APPS {args.split} split")
    logger.info(f"Difficulty: {args.difficulty}")
    logger.info(f"Output file: {output_file}")
    
    if results:
        avg_prompt_len = sum(len(r["prompt"]) for r in results) / len(results)
        avg_response_len = sum(len(r["model_output"]) for r in results) / len(results)
        logger.info(f"\nStatistics:")
        logger.info(f"  Average prompt length: {avg_prompt_len:.0f} characters")
        logger.info(f"  Average response length: {avg_response_len:.0f} characters")
    
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
        results = evaluate_model(args)
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