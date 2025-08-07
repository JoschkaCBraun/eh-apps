"""
Model Loader for GRPO Training

Loads Qwen3-14B with 4-bit quantization and LoRA configuration
following the Unsloth notebook settings.
"""

import torch
import gc
from typing import Optional, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from vllm import LLM
import logging

logger = logging.getLogger(__name__)


class GRPOModelLoader:
    """Load and configure models for GRPO training with LoRA and quantization."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        max_seq_length: int = 2048,
        lora_rank: int = 32,
        load_in_4bit: bool = True,
        gpu_memory_utilization: float = 0.7,
        random_seed: int = 3407
    ):
        """
        Initialize model loader with configuration.
        
        Args:
            model_name: Base model to load
            max_seq_length: Maximum sequence length
            lora_rank: LoRA rank (larger = smarter but slower)
            load_in_4bit: Whether to use 4-bit quantization
            gpu_memory_utilization: GPU memory utilization for vLLM
            random_seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank
        self.load_in_4bit = load_in_4bit
        self.gpu_memory_utilization = gpu_memory_utilization
        self.random_seed = random_seed
        
        # LoRA target modules matching notebook
        self.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    
    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """
        Load model and tokenizer with quantization and LoRA.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Max sequence length: {self.max_seq_length}")
        logger.info(f"LoRA rank: {self.lora_rank}")
        logger.info(f"4-bit quantization: {self.load_in_4bit}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization
        bnb_config = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if not self.load_in_4bit else None,
            device_map="auto",
            trust_remote_code=True,
            max_memory={0: f"{int(self.gpu_memory_utilization * 80)}GB"},  # H100 has 80GB
        )
        
        # Prepare model for k-bit training if using quantization
        if self.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank * 2,  # *2 speeds up training (from notebook)
            target_modules=self.target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Trainable parameters: {trainable_params:,}/{total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
        
        return model, tokenizer
    
    def load_vllm_model(self, model_path: Optional[str] = None) -> LLM:
        """
        Load model for vLLM inference during GRPO.
        
        Args:
            model_path: Path to model (uses self.model_name if None)
            
        Returns:
            vLLM LLM instance
        """
        model_to_load = model_path or self.model_name
        
        logger.info(f"Loading vLLM model: {model_to_load}")
        logger.info(f"GPU memory utilization: {self.gpu_memory_utilization}")
        
        # Initialize vLLM
        llm = LLM(
            model=model_to_load,
            max_model_len=self.max_seq_length,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            dtype="float16",  # Use FP16 for inference
            seed=self.random_seed,
            # Enable prefix caching for faster generation
            enable_prefix_caching=True,
            # Tensor parallel for single GPU
            tensor_parallel_size=1,
        )
        
        return llm
    
    def save_lora_adapter(self, model, save_path: str):
        """
        Save LoRA adapter weights.
        
        Args:
            model: PEFT model with LoRA
            save_path: Path to save adapter
        """
        logger.info(f"Saving LoRA adapter to: {save_path}")
        model.save_pretrained(save_path)
    
    def load_lora_adapter(self, base_model_name: str, adapter_path: str):
        """
        Load a saved LoRA adapter.
        
        Args:
            base_model_name: Name of base model
            adapter_path: Path to saved adapter
            
        Returns:
            Model with loaded adapter
        """
        from peft import PeftModel
        
        logger.info(f"Loading base model: {base_model_name}")
        logger.info(f"Loading adapter from: {adapter_path}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        return model
    
    def merge_and_save(self, model, tokenizer, save_path: str, save_method: str = "merged_16bit"):
        """
        Merge LoRA weights and save model.
        
        Args:
            model: PEFT model with LoRA
            tokenizer: Tokenizer
            save_path: Path to save merged model
            save_method: "merged_16bit" or "merged_4bit"
        """
        logger.info(f"Merging and saving model to: {save_path}")
        logger.info(f"Save method: {save_method}")
        
        # Merge LoRA weights
        model = model.merge_and_unload()
        
        # Save based on method
        if save_method == "merged_16bit":
            model.save_pretrained(save_path, safe_serialization=True)
        elif save_method == "merged_4bit":
            # For 4-bit, we need to requantize
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            # Note: Full 4-bit save requires additional handling
            model.save_pretrained(save_path, safe_serialization=True)
            logger.warning("4-bit save requested but saved as 16-bit. Use quantization tools for 4-bit.")
        else:
            raise ValueError(f"Unknown save method: {save_method}")
        
        # Save tokenizer
        tokenizer.save_pretrained(save_path)
    
    def cleanup_memory(self):
        """Clean up GPU memory."""
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            logger.info(f"GPU Memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")


def test_model_loader():
    """Test the model loader with a small configuration."""
    print("Testing Model Loader")
    print("=" * 80)
    
    # Use a smaller model for testing
    loader = GRPOModelLoader(
        model_name="Qwen/Qwen2.5-0.5B",  # Smaller model for testing
        max_seq_length=512,
        lora_rank=8,
        load_in_4bit=True
    )
    
    print("\nLoading model and tokenizer...")
    try:
        model, tokenizer = loader.load_model_and_tokenizer()
        print("✓ Model loaded successfully")
        print(f"✓ Model type: {type(model)}")
        print(f"✓ Tokenizer vocab size: {len(tokenizer)}")
        
        # Test generation
        inputs = tokenizer("What is 2+2?", return_tensors="pt")
        print("\nTesting generation...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response}")
        
        # Cleanup
        loader.cleanup_memory()
        print("\n✓ Memory cleaned up")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model_loader()