"""
GRPO Trainer Implementation

Main trainer class following the Unsloth notebook structure but using
standard PyTorch/HuggingFace/TRL libraries.
"""

import torch
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from trl import GRPOTrainer as TRLGRPOTrainer, SFTTrainer, SFTConfig as TRLSFTConfig
from transformers import TextStreamer

from .model_loader import GRPOModelLoader
from .dataset_processor import GRPODatasetProcessor
from .grpo_config import GRPOConfig, SFTConfig
from .reward_functions import create_reward_functions, reset_print_counter
from .memory_utils import MemoryManager
from .logging_utils import GRPOLogger
from .chat_templates import ChatTemplateManager

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """Main GRPO trainer following notebook structure."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        max_seq_length: int = 2048,
        lora_rank: int = 32,
        load_in_4bit: bool = True,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize GRPO trainer.
        
        Args:
            model_name: Base model name
            max_seq_length: Maximum sequence length
            lora_rank: LoRA rank
            load_in_4bit: Whether to use 4-bit quantization
            experiment_name: Name for the experiment
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank
        self.load_in_4bit = load_in_4bit
        
        # Initialize components
        self.model_loader = GRPOModelLoader(
            model_name=model_name,
            max_seq_length=max_seq_length,
            lora_rank=lora_rank,
            load_in_4bit=load_in_4bit
        )
        
        self.memory_manager = MemoryManager()
        self.grpo_logger = GRPOLogger(experiment_name=experiment_name)
        self.template_manager = ChatTemplateManager()
        
        # Will be initialized during setup
        self.model = None
        self.tokenizer = None
        self.dataset_processor = None
        self.vllm_model = None
    
    def setup(self):
        """Set up model, tokenizer, and other components."""
        logger.info("Setting up GRPO trainer...")
        
        # Apply H100 optimizations
        self.memory_manager.optimize_for_h100()
        
        # Load model and tokenizer
        self.model, self.tokenizer = self.model_loader.load_model_and_tokenizer()
        
        # Apply chat template
        self.template_manager.apply_to_tokenizer(self.tokenizer)
        
        # Initialize dataset processor
        self.dataset_processor = GRPODatasetProcessor(
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length
        )
        
        # Log initial memory
        self.memory_manager.log_memory_usage("After setup: ")
    
    def run_sft_pretraining(
        self,
        config: Optional[SFTConfig] = None,
        max_samples: int = 59
    ):
        """
        Run SFT pre-training phase for format learning.
        
        Args:
            config: SFT configuration
            max_samples: Maximum samples for pre-training
        """
        logger.info("=" * 80)
        logger.info("Starting SFT Pre-training Phase")
        logger.info("=" * 80)
        
        if config is None:
            config = SFTConfig()
        
        # Load pre-training dataset
        dataset = self.dataset_processor.load_pretraining_dataset(max_samples)
        logger.info(f"Pre-training on {len(dataset)} samples")
        
        # Create SFT trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=TRLSFTConfig(
                dataset_text_field=config.dataset_text_field,
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                warmup_steps=config.warmup_steps,
                num_train_epochs=config.num_train_epochs,
                learning_rate=config.learning_rate,
                logging_steps=config.logging_steps,
                optim=config.optim,
                weight_decay=config.weight_decay,
                lr_scheduler_type=config.lr_scheduler_type,
                seed=config.seed,
                report_to=config.report_to,
                output_dir=config.output_dir,
            ),
        )
        
        # Train
        logger.info("Starting SFT training...")
        train_result = trainer.train()
        
        # Log results
        logger.info(f"SFT Training completed:")
        logger.info(f"  Total steps: {train_result.global_step}")
        logger.info(f"  Final loss: {train_result.training_loss:.4f}")
        
        # Test generation
        self.test_format_generation()
        
        # Cleanup
        self.memory_manager.cleanup(force=True)
        
        return train_result
    
    def test_format_generation(self):
        """Test if model learned the format."""
        logger.info("\nTesting format generation...")
        
        test_problem = "What is 5 + 7?"
        messages = self.template_manager.format_for_generation(test_problem)
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=0,  # Deterministic
                max_new_tokens=256,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False
        )
        
        logger.info(f"Test problem: {test_problem}")
        logger.info(f"Generated response: {response[:500]}")
        
        # Check format
        parts = self.template_manager.extract_solution_parts(response)
        if parts["solution"]:
            logger.info(f"✓ Format learned! Extracted solution: {parts['solution']}")
        else:
            logger.warning("⚠ Format not properly learned yet")
    
    def run_grpo_training(
        self,
        config: Optional[GRPOConfig] = None,
        use_vllm: bool = True
    ):
        """
        Run main GRPO training phase.
        
        Args:
            config: GRPO configuration
            use_vllm: Whether to use vLLM for generation
        """
        logger.info("=" * 80)
        logger.info("Starting GRPO Training Phase")
        logger.info("=" * 80)
        
        if config is None:
            config = GRPOConfig()
        
        # Load GRPO dataset
        dataset, max_prompt_length = self.dataset_processor.load_grpo_dataset()
        config.update_for_dataset(max_prompt_length)
        
        logger.info(f"GRPO training on {len(dataset)} samples")
        logger.info(f"Max prompt length: {max_prompt_length}")
        logger.info(f"Max completion length: {config.max_completion_length}")
        
        # Load vLLM model if requested
        if use_vllm:
            try:
                # Save current model first
                temp_path = Path("outputs/temp_lora")
                self.model_loader.save_lora_adapter(self.model, str(temp_path))
                
                # Load with vLLM
                self.vllm_model = self.model_loader.load_vllm_model(str(temp_path))
                logger.info("✓ vLLM model loaded for fast generation")
            except Exception as e:
                logger.warning(f"Could not load vLLM model: {e}")
                logger.warning("Falling back to PyTorch generation")
                use_vllm = False
        
        # Create reward functions
        reward_funcs = create_reward_functions(self.tokenizer)
        reset_print_counter()
        
        # Get TRL config
        trl_config = config.get_trl_config()
        
        # Create GRPO trainer
        trainer = TRLGRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_funcs,
            args=trl_config,
            train_dataset=dataset,
        )
        
        # Train
        logger.info("Starting GRPO training...")
        logger.info("Watch the reward column - it should increase over time!")
        
        train_result = trainer.train()
        
        # Log results
        logger.info(f"\nGRPO Training completed:")
        logger.info(f"  Total steps: {train_result.global_step}")
        if hasattr(train_result, 'metrics'):
            if 'reward' in train_result.metrics:
                logger.info(f"  Final reward: {train_result.metrics['reward']:.4f}")
        
        # Save final model
        self.save_model("outputs/grpo_final")
        
        # Save training summary
        self.grpo_logger.save_summary()
        
        # Cleanup
        self.memory_manager.cleanup(force=True)
        
        return train_result
    
    def save_model(self, save_path: str, save_method: str = "lora"):
        """
        Save the trained model.
        
        Args:
            save_path: Path to save model
            save_method: "lora", "merged_16bit", or "merged_4bit"
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {save_path} (method: {save_method})")
        
        if save_method == "lora":
            # Save LoRA adapter only
            self.model_loader.save_lora_adapter(self.model, str(save_path))
            self.tokenizer.save_pretrained(str(save_path))
        else:
            # Merge and save
            self.model_loader.merge_and_save(
                self.model,
                self.tokenizer,
                str(save_path),
                save_method
            )
        
        logger.info(f"✓ Model saved to {save_path}")
    
    def test_final_model(self, adapter_path: Optional[str] = None):
        """
        Test the final trained model.
        
        Args:
            adapter_path: Path to saved LoRA adapter
        """
        logger.info("\nTesting final model...")
        
        test_problems = [
            "What is the sqrt of 101?",
            "What is 25 * 4?",
            "Solve: 2x + 5 = 13",
        ]
        
        for problem in test_problems:
            logger.info(f"\nProblem: {problem}")
            
            messages = self.template_manager.format_for_generation(problem)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate
            inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    temperature=1.0,
                    top_k=50,
                    max_new_tokens=512,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False
            )
            
            # Extract solution
            parts = self.template_manager.extract_solution_parts(response)
            if parts["solution"]:
                logger.info(f"Solution: {parts['solution']}")
            else:
                logger.info(f"Raw response: {response[:200]}...")
    
    def run_full_pipeline(
        self,
        sft_config: Optional[SFTConfig] = None,
        grpo_config: Optional[GRPOConfig] = None,
        skip_sft: bool = False,
        use_vllm: bool = True
    ):
        """
        Run the full GRPO training pipeline.
        
        Args:
            sft_config: SFT configuration
            grpo_config: GRPO configuration
            skip_sft: Skip SFT pre-training
            use_vllm: Use vLLM for generation
        """
        logger.info("="*80)
        logger.info("Starting Full GRPO Training Pipeline")
        logger.info("="*80)
        
        # Setup
        self.setup()
        
        # Phase 1: SFT Pre-training
        if not skip_sft:
            self.run_sft_pretraining(sft_config)
        else:
            logger.info("Skipping SFT pre-training phase")
        
        # Phase 2: GRPO Training
        self.run_grpo_training(grpo_config, use_vllm)
        
        # Phase 3: Test final model
        self.test_final_model()
        
        logger.info("\n" + "="*80)
        logger.info("GRPO Training Pipeline Completed!")
        logger.info("="*80)


def test_trainer():
    """Test the GRPO trainer with minimal configuration."""
    print("Testing GRPO Trainer")
    print("=" * 80)
    
    # Use a small model for testing
    trainer = GRPOTrainer(
        model_name="Qwen/Qwen2.5-0.5B",  # Small model for testing
        max_seq_length=512,
        lora_rank=8,
        load_in_4bit=True,
        experiment_name="test_grpo"
    )
    
    # Test setup
    print("\nTesting setup...")
    try:
        trainer.setup()
        print("✓ Setup completed")
    except Exception as e:
        print(f"✗ Setup failed: {e}")
        return
    
    # Test format generation
    print("\nTesting format generation...")
    trainer.test_format_generation()
    
    print("\n✓ Basic trainer tests passed!")


if __name__ == "__main__":
    test_trainer()