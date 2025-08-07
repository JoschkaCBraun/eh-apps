"""
GRPO (Group Relative Policy Optimization) Implementation

A PyTorch implementation of GRPO for training reasoning models,
following the Unsloth notebook structure but adapted for standard libraries.
"""

from .chat_templates import ChatTemplateManager
from .formatting_utils import FormatChecker, extract_answer, extract_number
from .reward_functions import (
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers
)
from .model_loader import GRPOModelLoader
from .dataset_processor import GRPODatasetProcessor
from .grpo_config import GRPOConfig, SFTConfig
from .grpo_trainer import GRPOTrainer

__all__ = [
    "ChatTemplateManager",
    "FormatChecker",
    "extract_answer",
    "extract_number",
    "match_format_exactly",
    "match_format_approximately",
    "check_answer",
    "check_numbers",
    "GRPOModelLoader",
    "GRPODatasetProcessor",
    "GRPOConfig",
    "SFTConfig",
    "GRPOTrainer",
]