"""
Configuration Classes for GRPO Training

Defines configuration parameters matching the Unsloth notebook settings.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from vllm import SamplingParams


@dataclass
class SFTConfig:
    """Configuration for SFT pre-training phase (format learning)."""
    
    # Dataset settings
    dataset_text_field: str = "text"
    
    # Training parameters (from notebook)
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 5
    num_train_epochs: int = 2
    learning_rate: float = 2e-4  # Higher for initial format learning
    
    # Optimizer settings
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    
    # Logging
    logging_steps: int = 5
    report_to: str = "none"  # Can be "wandb", "tensorboard", etc.
    
    # Other
    seed: int = 3407
    output_dir: str = "outputs/sft"
    save_steps: int = 100
    save_total_limit: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for TRL trainer."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


@dataclass
class GRPOConfig:
    """Configuration for GRPO training phase matching notebook settings."""
    
    # vLLM sampling parameters (exact from notebook)
    vllm_sampling_params: Optional[SamplingParams] = None
    temperature: float = 1.0
    min_p: float = 0.1
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 3407
    
    # Training parameters (from notebook)
    learning_rate: float = 5e-6  # Lower for GRPO
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    
    # Batch and generation settings
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1  # Can increase to 4 for smoother training
    num_generations: int = 4  # Number of completions per prompt
    
    # Sequence length settings
    max_prompt_length: int = 202  # Will be set dynamically based on dataset
    max_completion_length: int = 1846  # max_seq_length - max_prompt_length
    max_seq_length: int = 2048
    
    # Training duration
    max_steps: int = 100  # For testing, increase for full training
    num_train_epochs: Optional[int] = None  # Can use epochs instead of steps
    
    # Logging and saving
    logging_steps: int = 1
    save_steps: int = 100
    report_to: str = "none"  # Can use "wandb"
    output_dir: str = "outputs/grpo"
    
    # Evaluation settings (optional)
    fp16_full_eval: bool = False
    per_device_eval_batch_size: int = 4
    eval_accumulation_steps: int = 1
    eval_strategy: str = "no"  # Can be "steps"
    eval_steps: int = 100
    
    # Memory settings
    gradient_checkpointing: bool = True
    
    # KL penalty
    kl_penalty: str = "kl"  # Type of KL penalty
    
    def __post_init__(self):
        """Initialize vLLM sampling params if not provided."""
        if self.vllm_sampling_params is None:
            self.vllm_sampling_params = SamplingParams(
                min_p=self.min_p,
                top_p=self.top_p,
                top_k=self.top_k,
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=self.max_completion_length,
                # Stop tokens will be set from tokenizer
                include_stop_str_in_output=True,
            )
        
        # Update max_completion_length based on max_seq_length and max_prompt_length
        self.max_completion_length = self.max_seq_length - self.max_prompt_length
    
    def update_for_dataset(self, max_prompt_length: int):
        """
        Update configuration based on dataset statistics.
        
        Args:
            max_prompt_length: Maximum prompt length from dataset
        """
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = self.max_seq_length - max_prompt_length
        
        # Update vLLM params
        if self.vllm_sampling_params:
            self.vllm_sampling_params.max_tokens = self.max_completion_length
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for TRL GRPOConfig."""
        config_dict = {}
        
        # Map our config to TRL GRPOConfig fields
        config_dict["learning_rate"] = self.learning_rate
        config_dict["weight_decay"] = self.weight_decay
        config_dict["warmup_ratio"] = self.warmup_ratio
        config_dict["lr_scheduler_type"] = self.lr_scheduler_type
        config_dict["optim"] = self.optim
        
        config_dict["per_device_train_batch_size"] = self.per_device_train_batch_size
        config_dict["gradient_accumulation_steps"] = self.gradient_accumulation_steps
        config_dict["num_generations"] = self.num_generations
        
        config_dict["max_prompt_length"] = self.max_prompt_length
        config_dict["max_completion_length"] = self.max_completion_length
        
        if self.num_train_epochs:
            config_dict["num_train_epochs"] = self.num_train_epochs
        else:
            config_dict["max_steps"] = self.max_steps
        
        config_dict["logging_steps"] = self.logging_steps
        config_dict["save_steps"] = self.save_steps
        config_dict["report_to"] = self.report_to
        config_dict["output_dir"] = self.output_dir
        
        config_dict["gradient_checkpointing"] = self.gradient_checkpointing
        
        # Evaluation settings
        if self.eval_strategy != "no":
            config_dict["eval_strategy"] = self.eval_strategy
            config_dict["eval_steps"] = self.eval_steps
            config_dict["per_device_eval_batch_size"] = self.per_device_eval_batch_size
            config_dict["eval_accumulation_steps"] = self.eval_accumulation_steps
            config_dict["fp16_full_eval"] = self.fp16_full_eval
        
        return config_dict
    
    def get_trl_config(self):
        """Get TRL GRPOConfig object."""
        from trl import GRPOConfig as TRLGRPOConfig
        
        config_dict = self.to_dict()
        
        # Add vLLM params separately as TRL expects it
        config_dict["vllm_sampling_params"] = self.vllm_sampling_params
        config_dict["temperature"] = self.temperature
        
        return TRLGRPOConfig(**config_dict)


def get_default_sft_config() -> SFTConfig:
    """Get default SFT configuration matching notebook."""
    return SFTConfig()


def get_default_grpo_config(max_seq_length: int = 2048) -> GRPOConfig:
    """
    Get default GRPO configuration matching notebook.
    
    Args:
        max_seq_length: Maximum sequence length
        
    Returns:
        GRPOConfig instance
    """
    config = GRPOConfig(max_seq_length=max_seq_length)
    return config


def test_configs():
    """Test configuration classes."""
    print("Testing Configuration Classes")
    print("=" * 80)
    
    # Test SFT config
    print("\nSFT Configuration:")
    sft_config = get_default_sft_config()
    print(f"  Learning rate: {sft_config.learning_rate}")
    print(f"  Epochs: {sft_config.num_train_epochs}")
    print(f"  Batch size: {sft_config.per_device_train_batch_size}")
    print(f"  Optimizer: {sft_config.optim}")
    
    # Test GRPO config
    print("\nGRPO Configuration:")
    grpo_config = get_default_grpo_config()
    print(f"  Learning rate: {grpo_config.learning_rate}")
    print(f"  Temperature: {grpo_config.temperature}")
    print(f"  Num generations: {grpo_config.num_generations}")
    print(f"  Max prompt length: {grpo_config.max_prompt_length}")
    print(f"  Max completion length: {grpo_config.max_completion_length}")
    
    # Test update for dataset
    print("\nUpdating for dataset...")
    grpo_config.update_for_dataset(max_prompt_length=256)
    print(f"  Updated max prompt length: {grpo_config.max_prompt_length}")
    print(f"  Updated max completion length: {grpo_config.max_completion_length}")
    
    # Test conversion to dict
    print("\nConverting to dict...")
    config_dict = grpo_config.to_dict()
    print(f"  Dict keys: {list(config_dict.keys())[:5]}...")


if __name__ == "__main__":
    test_configs()