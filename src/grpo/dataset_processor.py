"""
Dataset Processor for GRPO Training

Handles loading and preprocessing of the DAPO-Math dataset following
the exact approach from the Unsloth notebook.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from typing import List, Dict, Optional, Tuple
import logging
from .chat_templates import ChatTemplateManager

logger = logging.getLogger(__name__)


class GRPODatasetProcessor:
    """Process datasets for GRPO training following notebook structure."""
    
    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 2048,
        random_seed: int = 3407
    ):
        """
        Initialize dataset processor.
        
        Args:
            tokenizer: Tokenizer with chat template applied
            max_seq_length: Maximum sequence length
            random_seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.random_seed = random_seed
        self.template_manager = ChatTemplateManager()
        
        # Apply chat template to tokenizer
        self.template_manager.apply_to_tokenizer(tokenizer)
    
    def load_pretraining_dataset(self, max_samples: int = 59) -> Dataset:
        """
        Load the pre-training dataset (OpenMathReasoning-mini) for format learning.
        
        Args:
            max_samples: Maximum samples for pre-training (default 59 from notebook)
            
        Returns:
            Formatted dataset for SFT pre-training
        """
        logger.info("Loading pre-training dataset: unsloth/OpenMathReasoning-mini")
        
        # Load dataset
        dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
        dataset = dataset.to_pandas()[
            ["expected_answer", "problem", "generated_solution"]
        ]
        
        # Filter to only numeric answers
        is_number = pd.to_numeric(
            pd.Series(dataset["expected_answer"]), 
            errors="coerce"
        ).notnull()
        dataset = dataset.iloc[np.where(is_number)[0]]
        
        # Format dataset entries
        def format_dataset_entry(x):
            expected_answer = x["expected_answer"]
            problem = x["problem"]
            
            # Remove existing <think> tags and format with our tags
            thoughts = x["generated_solution"]
            thoughts = thoughts.replace("<think>", "").replace("</think>", "")
            thoughts = thoughts.strip()
            
            # Create formatted solution
            final_prompt = (
                f"{self.template_manager.reasoning_start}{thoughts}"
                f"{self.template_manager.reasoning_end}"
                f"{self.template_manager.solution_start}{expected_answer}"
                f"{self.template_manager.solution_end}"
            )
            
            return [
                {"role": "system", "content": self.template_manager.system_prompt},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": final_prompt},
            ]
        
        dataset["Messages"] = dataset.apply(format_dataset_entry, axis=1)
        
        # Filter by length (max_seq_length/2 as in notebook)
        logger.info("Filtering dataset by sequence length...")
        dataset["N"] = dataset["Messages"].apply(
            lambda x: len(self.tokenizer.apply_chat_template(x))
        )
        dataset = dataset.loc[dataset["N"] <= self.max_seq_length / 2].copy()
        
        # Limit to max_samples
        if len(dataset) > max_samples:
            dataset = dataset.head(max_samples)
        
        logger.info(f"Pre-training dataset size: {len(dataset)} samples")
        
        # Convert to HuggingFace dataset format
        dataset["text"] = self.tokenizer.apply_chat_template(
            dataset["Messages"].values.tolist(),
            tokenize=False
        )
        
        return Dataset.from_pandas(dataset)
    
    def load_grpo_dataset(self, quantile_cutoff: float = 0.9) -> Tuple[Dataset, int]:
        """
        Load the main GRPO training dataset (DAPO-Math-17k).
        
        Args:
            quantile_cutoff: Quantile for maximum prompt length filtering
            
        Returns:
            Tuple of (dataset, max_prompt_length)
        """
        logger.info("Loading GRPO dataset: open-r1/DAPO-Math-17k-Processed")
        
        # Load dataset
        dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
        
        logger.info(f"Original dataset size: {len(dataset)} samples")
        
        # Map to add prompts and answers
        def map_dataset(x):
            return {
                "prompt": [
                    {"role": "system", "content": self.template_manager.system_prompt},
                    {"role": "user", "content": x["prompt"]},
                ],
                "answer": self._extract_answer(x["solution"])
            }
        
        dataset = dataset.map(map_dataset)
        
        # Tokenize to get lengths
        logger.info("Computing prompt lengths for filtering...")
        tokenized = dataset.map(
            lambda x: {
                "tokens": self.tokenizer.apply_chat_template(
                    x["prompt"],
                    add_generation_prompt=True,
                    tokenize=True
                )
            },
            batched=True,
        )
        
        # Add length column
        tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
        
        # Calculate quantile cutoff
        lengths = np.array(tokenized["L"])
        maximum_length = int(np.quantile(lengths, quantile_cutoff))
        logger.info(f"Maximum prompt length (90th percentile): {maximum_length}")
        
        # Filter by length
        dataset = dataset.select(np.where(lengths <= maximum_length)[0])
        
        logger.info(f"Filtered dataset size: {len(dataset)} samples")
        
        # Calculate max prompt length for GRPO config
        max_prompt_length = maximum_length + 1  # +1 just in case (from notebook)
        
        return dataset, max_prompt_length
    
    def _extract_answer(self, text: str) -> str:
        """
        Extract answer from solution text.
        For DAPO-Math, the solution is typically just the answer.
        
        Args:
            text: Solution text
            
        Returns:
            Extracted answer
        """
        # In the notebook, for GSM8K they extract after ####
        # For DAPO-Math, the solution is already the answer
        return text.strip()
    
    def prepare_for_grpo(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset for GRPO training.
        
        Args:
            dataset: Raw dataset
            
        Returns:
            Dataset ready for GRPO
        """
        # The dataset should already have 'prompt' and 'answer' fields
        # from load_grpo_dataset
        return dataset
    
    def create_test_batch(self, n_samples: int = 5) -> List[Dict]:
        """
        Create a small test batch for validation.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            List of test samples
        """
        test_problems = [
            {
                "prompt": [
                    {"role": "system", "content": self.template_manager.system_prompt},
                    {"role": "user", "content": "What is 2 + 2?"}
                ],
                "answer": "4"
            },
            {
                "prompt": [
                    {"role": "system", "content": self.template_manager.system_prompt},
                    {"role": "user", "content": "What is 5 * 6?"}
                ],
                "answer": "30"
            },
            {
                "prompt": [
                    {"role": "system", "content": self.template_manager.system_prompt},
                    {"role": "user", "content": "What is 100 / 4?"}
                ],
                "answer": "25"
            },
            {
                "prompt": [
                    {"role": "system", "content": self.template_manager.system_prompt},
                    {"role": "user", "content": "What is 7 + 8?"}
                ],
                "answer": "15"
            },
            {
                "prompt": [
                    {"role": "system", "content": self.template_manager.system_prompt},
                    {"role": "user", "content": "What is 9 * 9?"}
                ],
                "answer": "81"
            },
        ]
        
        return test_problems[:n_samples]


def test_dataset_processor():
    """Test the dataset processor."""
    print("Testing Dataset Processor")
    print("=" * 80)
    
    # Create a dummy tokenizer for testing
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize processor
    processor = GRPODatasetProcessor(
        tokenizer=tokenizer,
        max_seq_length=512
    )
    
    # Test pre-training dataset
    print("\nLoading pre-training dataset...")
    try:
        pretrain_dataset = processor.load_pretraining_dataset(max_samples=5)
        print(f"✓ Pre-training dataset loaded: {len(pretrain_dataset)} samples")
        print(f"  First sample preview: {pretrain_dataset[0]['text'][:200]}...")
    except Exception as e:
        print(f"✗ Error loading pre-training dataset: {e}")
    
    # Test GRPO dataset
    print("\nLoading GRPO dataset...")
    try:
        grpo_dataset, max_prompt_length = processor.load_grpo_dataset()
        print(f"✓ GRPO dataset loaded: {len(grpo_dataset)} samples")
        print(f"  Max prompt length: {max_prompt_length}")
        print(f"  First prompt: {grpo_dataset[0]['prompt']}")
        print(f"  First answer: {grpo_dataset[0]['answer']}")
    except Exception as e:
        print(f"✗ Error loading GRPO dataset: {e}")
    
    # Test batch creation
    print("\nCreating test batch...")
    test_batch = processor.create_test_batch(3)
    print(f"✓ Test batch created: {len(test_batch)} samples")
    for i, sample in enumerate(test_batch):
        print(f"  Sample {i+1} answer: {sample['answer']}")


if __name__ == "__main__":
    test_dataset_processor()