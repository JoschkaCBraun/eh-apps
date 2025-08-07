"""
Sampling Utilities for GRPO Training

Handles vLLM-based generation during GRPO training.
"""

from typing import List, Dict, Any, Optional
from vllm import SamplingParams, LLM
import torch
import logging

logger = logging.getLogger(__name__)


class GRPOSampler:
    """Handle sampling/generation for GRPO training."""
    
    def __init__(
        self,
        vllm_model: Optional[LLM] = None,
        tokenizer=None,
        sampling_params: Optional[SamplingParams] = None
    ):
        """
        Initialize sampler.
        
        Args:
            vllm_model: vLLM model for generation
            tokenizer: Tokenizer for encoding/decoding
            sampling_params: vLLM sampling parameters
        """
        self.vllm_model = vllm_model
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params or self._get_default_params()
    
    def _get_default_params(self) -> SamplingParams:
        """Get default sampling parameters from notebook."""
        params = SamplingParams(
            min_p=0.1,
            top_p=1.0,
            top_k=-1,
            temperature=1.0,
            seed=3407,
            max_tokens=1024,
            stop=[self.tokenizer.eos_token] if self.tokenizer else None,
            include_stop_str_in_output=True,
        )
        return params
    
    def generate_batch(
        self,
        prompts: List[str],
        num_generations: int = 4,
        sampling_params: Optional[SamplingParams] = None
    ) -> List[List[str]]:
        """
        Generate multiple completions for each prompt using vLLM.
        
        Args:
            prompts: List of prompt strings
            num_generations: Number of generations per prompt
            sampling_params: Override sampling parameters
            
        Returns:
            List of lists, where each inner list contains generations for one prompt
        """
        if self.vllm_model is None:
            raise ValueError("vLLM model not initialized")
        
        params = sampling_params or self.sampling_params
        
        # Duplicate prompts for multiple generations
        expanded_prompts = []
        prompt_indices = []
        for i, prompt in enumerate(prompts):
            for _ in range(num_generations):
                expanded_prompts.append(prompt)
                prompt_indices.append(i)
        
        # Generate with vLLM
        logger.info(f"Generating {len(expanded_prompts)} completions...")
        outputs = self.vllm_model.generate(expanded_prompts, params)
        
        # Group outputs by original prompt
        grouped_outputs = [[] for _ in range(len(prompts))]
        for output, prompt_idx in zip(outputs, prompt_indices):
            generated_text = output.outputs[0].text
            grouped_outputs[prompt_idx].append(generated_text)
        
        return grouped_outputs
    
    def generate_with_torch_model(
        self,
        model,
        prompts: List[str],
        num_generations: int = 1,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50
    ) -> List[List[str]]:
        """
        Generate completions using a PyTorch model (fallback for non-vLLM).
        
        Args:
            model: PyTorch model
            prompts: List of prompt strings
            num_generations: Number of generations per prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            top_k: Top-k for sampling
            
        Returns:
            List of lists of generated texts
        """
        model.eval()
        all_outputs = []
        
        for prompt in prompts:
            prompt_outputs = []
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate multiple times
            for _ in range(num_generations):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode
                generated = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=False  # Keep EOS for format checking
                )
                prompt_outputs.append(generated)
            
            all_outputs.append(prompt_outputs)
        
        return all_outputs
    
    def prepare_prompts_for_generation(
        self,
        messages_batch: List[List[Dict]],
        add_generation_prompt: bool = True
    ) -> List[str]:
        """
        Prepare message batches for generation.
        
        Args:
            messages_batch: List of message lists
            add_generation_prompt: Whether to add generation prompt
            
        Returns:
            List of formatted prompt strings
        """
        prompts = []
        for messages in messages_batch:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
            prompts.append(prompt)
        
        return prompts
    
    def update_sampling_params(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Update sampling parameters.
        
        Args:
            temperature: New temperature value
            top_p: New top-p value
            top_k: New top-k value
            max_tokens: New max tokens value
        """
        # Create new params (vLLM SamplingParams are immutable)
        current_params = self.sampling_params
        
        self.sampling_params = SamplingParams(
            temperature=temperature or current_params.temperature,
            top_p=top_p or current_params.top_p,
            top_k=top_k if top_k is not None else current_params.top_k,
            max_tokens=max_tokens or current_params.max_tokens,
            min_p=current_params.min_p,
            seed=current_params.seed,
            stop=current_params.stop,
            include_stop_str_in_output=current_params.include_stop_str_in_output,
        )


def test_sampler():
    """Test the sampler with mock data."""
    print("Testing GRPOSampler")
    print("=" * 80)
    
    # Create mock tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize sampler (without vLLM for testing)
    sampler = GRPOSampler(tokenizer=tokenizer)
    
    # Test prompt preparation
    print("\nTesting prompt preparation...")
    messages = [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "What is 5*6?"}]
    ]
    
    prompts = sampler.prepare_prompts_for_generation(messages)
    print(f"✓ Prepared {len(prompts)} prompts")
    print(f"  First prompt: {prompts[0][:50]}...")
    
    # Test parameter updates
    print("\nTesting parameter updates...")
    sampler.update_sampling_params(temperature=0.7, max_tokens=512)
    print(f"✓ Updated temperature: {sampler.sampling_params.temperature}")
    print(f"✓ Updated max_tokens: {sampler.sampling_params.max_tokens}")
    
    print("\n✓ All sampler tests passed!")


if __name__ == "__main__":
    test_sampler()