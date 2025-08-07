#!/usr/bin/env python
"""
GRPO Training Script for Qwen3-14B

This script implements the full GRPO training pipeline following the
Unsloth notebook structure, adapted for H100 GPU training.

Usage:
    python scripts/train_grpo_qwen3.py [options]

Examples:
    # Full training with default settings
    python scripts/train_grpo_qwen3.py
    
    # Skip SFT pre-training
    python scripts/train_grpo_qwen3.py --skip-sft
    
    # Custom model and settings
    python scripts/train_grpo_qwen3.py --model-name Qwen/Qwen3-14B --lora-rank 64
    
    # Test mode with small model
    python scripts/train_grpo_qwen3.py --test-mode
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.grpo import (
    GRPOTrainer,
    GRPOConfig,
    SFTConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('grpo_training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GRPO Training for Qwen3 Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-14B",
        help="Base model name (default: Qwen/Qwen3-14B)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank (default: 32)"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    
    # Training configuration
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Skip SFT pre-training phase"
    )
    parser.add_argument(
        "--sft-epochs",
        type=int,
        default=2,
        help="Number of SFT epochs (default: 2)"
    )
    parser.add_argument(
        "--sft-samples",
        type=int,
        default=59,
        help="Number of SFT samples (default: 59)"
    )
    parser.add_argument(
        "--grpo-steps",
        type=int,
        default=100,
        help="Number of GRPO steps (default: 100)"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of generations per prompt (default: 4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size (default: 1)"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)"
    )
    
    # Learning rates
    parser.add_argument(
        "--sft-lr",
        type=float,
        default=2e-4,
        help="SFT learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--grpo-lr",
        type=float,
        default=5e-6,
        help="GRPO learning rate (default: 5e-6)"
    )
    
    # Generation settings
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Disable vLLM for generation"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for logging"
    )
    
    # Utility options
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with small model"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Override for test mode
    if args.test_mode:
        logger.info("Running in TEST MODE with small model")
        args.model_name = "Qwen/Qwen2.5-0.5B"
        args.max_seq_length = 512
        args.lora_rank = 8
        args.sft_samples = 5
        args.grpo_steps = 10
    
    # Log configuration
    logger.info("="*80)
    logger.info("GRPO Training Configuration")
    logger.info("="*80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Max sequence length: {args.max_seq_length}")
    logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info(f"4-bit quantization: {not args.no_4bit}")
    logger.info(f"Skip SFT: {args.skip_sft}")
    logger.info(f"GRPO steps: {args.grpo_steps}")
    logger.info(f"Use vLLM: {not args.no_vllm}")
    logger.info("="*80)
    
    # Create trainer
    trainer = GRPOTrainer(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        load_in_4bit=not args.no_4bit,
        experiment_name=args.experiment_name
    )
    
    # Create SFT config
    sft_config = SFTConfig(
        num_train_epochs=args.sft_epochs,
        learning_rate=args.sft_lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        output_dir=f"{args.output_dir}/sft",
        report_to="wandb" if args.wandb else "none"
    )
    
    # Create GRPO config
    grpo_config = GRPOConfig(
        max_steps=args.grpo_steps,
        learning_rate=args.grpo_lr,
        temperature=args.temperature,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_seq_length=args.max_seq_length,
        output_dir=f"{args.output_dir}/grpo",
        report_to="wandb" if args.wandb else "none"
    )
    
    try:
        # Run training pipeline
        trainer.run_full_pipeline(
            sft_config=sft_config,
            grpo_config=grpo_config,
            skip_sft=args.skip_sft,
            use_vllm=not args.no_vllm
        )
        
        logger.info("\n✓ Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n⚠ Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()