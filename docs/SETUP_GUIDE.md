# Quick Setup Guide

## Prerequisites

- Python 3.11 or higher
- UV package manager (install with `brew install uv` on macOS)
- OpenRouter API key

## Initial Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd solution-space-hacking
   ```

2. **Set up environment**
   ```bash
   # Install dependencies
   uv sync
   
   # Set up environment variables
   cp env.example .env
   # Edit .env and add your OPENROUTER_API_KEY
   ```

3. **Test the setup**
   ```bash
   # Test APPS dataset loading
   uv run python src/apps/load_apps_dataset.py
   
   # Test OpenRouter connection
   uv run python src/openrouter/api_client.py
   ```

## First Experiment

1. **Load APPS dataset**
   ```bash
   uv run python src/apps/load_apps_dataset.py
   ```

2. **Run evaluation on a few problems**
   ```bash
   uv run python src/evaluation/evaluate_models_on_apps.py
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

## Troubleshooting

- **API key issues**: Make sure `OPENROUTER_API_KEY` is set in `.env`
- **Import errors**: Make sure you're running from the project root
- **Dataset loading**: Check internet connection for HuggingFace access 