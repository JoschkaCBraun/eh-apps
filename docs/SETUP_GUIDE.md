# Quick Setup Guide

## Prerequisites

- Python 3.11 or higher
- Linux environment with NVIDIA H100 80GB GPU
- CUDA 12.0 or higher
- UV package manager (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Optional: OpenRouter API key (for API-based inference)
- 100GB+ free disk space for model weights

## Initial Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd eh-apps
   ```

2. **Set up environment**
   ```bash
   # Install dependencies
   uv sync
   
   # Set up environment variables (optional for API usage)
   cp env.example .env
   # Edit .env and add your OPENROUTER_API_KEY if using API
   
   # Verify GPU availability
   nvidia-smi
   ```

3. **Test the setup**
   ```bash
   # Test APPS dataset loading
   uv run python src/utils/dataset_loader.py
   
   # Test local model loading (Qwen3-32B)
   uv run python scripts/test_qwen_model.py
   
   # Optional: Test OpenRouter connection
   uv run python src/openrouter/async_client.py
   ```

## Local Model Setup (Qwen3-32B)

1. **Configure the model**
   - Model configuration is in `config/model_config.yaml`
   - Default: Qwen3-32B in full precision mode
   - Automatic GPU detection and optimization for H100

2. **Load and test the model**
   ```bash
   # Run the test script to verify model loading
   uv run python scripts/test_qwen_model.py
   ```

3. **Use in your code**
   ```python
   from src.utils.load_model import load_qwen_model
   
   # Load model
   loader = load_qwen_model()
   
   # Generate text
   response = loader.generate("Write a Python function to sort a list:")
   
   # Clean up GPU memory
   loader.clear_memory()
   ```

## First Experiment

1. **Generate model outputs**
   ```bash
   uv run python scripts/run_generation.py --n-problems 10 --split eval
   ```

2. **Evaluate the outputs**
   ```bash
   uv run python scripts/run_evaluation.py --input-file data/generation_outputs/latest_file.json
   ```

3. **Check results**
   ```bash
   ls data/generation_outputs/
   ```

## Development Workflow

1. **Personal experiments**: Use `scratch/joschka/` for your work
2. **Shared code**: Add to `src/` modules
3. **Documentation**: Update `docs/` as you add features
4. **Tests**: Add tests in `tests/` for new functionality

## Common Commands

```bash
# Add new dependency
uv add package_name

# Run script in virtual environment
uv run python your_script.py

# Activate virtual environment manually
source .venv/bin/activate

# Run tests
uv run pytest tests/
```

## GPU and Model Management

### Monitor GPU Usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Check CUDA availability in Python
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Memory Management Tips
- The Qwen3-32B model requires ~65GB in float16 precision
- Use `loader.clear_memory()` after inference to free GPU memory
- Batch size can be adjusted in `config/model_config.yaml`
- Monitor memory with `loader.get_memory_usage()`

## Troubleshooting

- **GPU/CUDA issues**: 
  - Verify CUDA installation: `nvcc --version`
  - Check PyTorch CUDA: `python -c "import torch; print(torch.version.cuda)"`
  - Ensure GPU is visible: `nvidia-smi`
- **Model loading OOM**: 
  - Reduce batch size in config
  - Enable 8-bit quantization: set `load_in_8bit: true` in config
  - Clear GPU cache: `torch.cuda.empty_cache()`
- **API key issues**: Make sure `OPENROUTER_API_KEY` is set in `.env` (if using API)
- **Import errors**: Make sure you're running from the project root
- **Dataset loading**: Check internet connection for HuggingFace access
- **Model download**: First run will download ~65GB model weights from HuggingFace 