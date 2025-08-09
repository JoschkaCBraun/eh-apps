# GRPO (Group Relative Policy Optimization) Implementation

This module implements GRPO training for Qwen3 models, following the exact structure and settings from the Unsloth notebook but adapted for standard PyTorch/HuggingFace/TRL libraries.

## Features

- **Exact notebook implementation**: Follows the Unsloth notebook structure with the same reward functions, formatting, and training approach
- **4-bit quantization**: Uses BitsAndBytes for memory-efficient training
- **LoRA fine-tuning**: Parameter-efficient training with configurable rank
- **vLLM integration**: Optional fast generation during GRPO
- **H100 optimized**: Memory management and settings optimized for H100 GPU

## Quick Start

### Installation

First, install the required dependencies:

```bash
uv pip install -r pyproject.toml
```

### Basic Training

Run the full GRPO pipeline with default settings:

```bash
python scripts/train_grpo_qwen3.py
```

### Test Mode

Test with a small model to verify everything works:

```bash
python scripts/train_grpo_qwen3.py --test-mode
```

### Custom Configuration

```bash
# Use Qwen3-14B with custom settings
python scripts/train_grpo_qwen3.py \
    --model-name Qwen/Qwen3-14B \
    --lora-rank 64 \
    --grpo-steps 500 \
    --num-generations 4 \
    --temperature 1.0

# Skip SFT pre-training if already done
python scripts/train_grpo_qwen3.py --skip-sft

# Use larger batch size for H100
python scripts/train_grpo_qwen3.py \
    --batch-size 2 \
    --gradient-accumulation 4
```

## Module Structure

```
src/grpo/
├── __init__.py              # Module exports
├── chat_templates.py        # Chat template management (exact from notebook)
├── formatting_utils.py      # Format checking and regex patterns
├── reward_functions.py      # GRPO reward functions (exact from notebook)
├── model_loader.py          # Model loading with 4-bit quantization and LoRA
├── dataset_processor.py     # Dataset handling for DAPO-Math
├── grpo_config.py          # Training configurations
├── sampling.py             # vLLM-based generation
├── memory_utils.py         # GPU memory management
├── logging_utils.py        # Training metrics logging
├── grpo_trainer.py         # Main trainer class
└── README.md              # This file
```

## Training Pipeline

The training follows a 3-phase approach:

### Phase 1: SFT Pre-training
- Trains on ~59 samples from OpenMathReasoning-mini
- Teaches the model the exact formatting with reasoning tags
- Uses higher learning rate (2e-4) for 2 epochs

### Phase 2: GRPO Training
- Trains on DAPO-Math-17k dataset
- Uses 4 generations per prompt
- Applies exact reward functions from notebook
- Lower learning rate (5e-6) for stable training

### Phase 3: Evaluation
- Tests the trained model on sample problems
- Verifies format compliance
- Saves the final LoRA adapter

## Key Settings (from Notebook)

### Model Configuration
- **Base model**: Qwen3-14B (or other Qwen models)
- **Quantization**: 4-bit with NF4
- **LoRA rank**: 32 (configurable)
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Training Parameters
- **SFT learning rate**: 2e-4
- **GRPO learning rate**: 5e-6
- **Temperature**: 1.0
- **Optimizer**: adamw_8bit
- **Max sequence length**: 2048

### Reward Functions
1. `match_format_exactly`: +3 points for correct format
2. `match_format_approximately`: +0.5 per correct tag
3. `check_answer`: +5 for exact match, partial credit for close answers
4. `check_numbers`: Numeric comparison with float conversion

## Programmatic Usage

```python
from src.grpo import GRPOTrainer, GRPOConfig, SFTConfig

# Initialize trainer
trainer = GRPOTrainer(
    model_name="Qwen/Qwen3-14B",
    max_seq_length=2048,
    lora_rank=32,
    load_in_4bit=True
)

# Configure training
sft_config = SFTConfig(num_train_epochs=2)
grpo_config = GRPOConfig(max_steps=100)

# Run full pipeline
trainer.run_full_pipeline(
    sft_config=sft_config,
    grpo_config=grpo_config
)
```

## Output Files

Training creates several output files:

```
outputs/
├── sft/                    # SFT checkpoints
├── grpo/                   # GRPO checkpoints
├── grpo_final/            # Final LoRA adapter
├── logs/                  # Training logs
│   ├── experiment.jsonl   # Step-by-step metrics
│   └── experiment_summary.json  # Training summary
└── temp_lora/            # Temporary files for vLLM
```

## Memory Requirements

- **Qwen3-14B with 4-bit**: ~15-20GB VRAM
- **H100 (80GB)**: Can handle larger batch sizes and longer sequences

## Troubleshooting

### Out of Memory
- Reduce `max_seq_length`
- Decrease `num_generations`
- Lower `gpu_memory_utilization` in model loader
- Use smaller LoRA rank

### Format Not Learning
- Increase SFT epochs
- Check if dataset is properly formatted
- Verify chat template is applied correctly

### Low Rewards
- Ensure SFT pre-training completed successfully
- Check reward function implementations
- Verify answer extraction patterns


Get the prompt from 