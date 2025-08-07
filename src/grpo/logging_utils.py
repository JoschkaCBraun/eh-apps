"""
Logging Utilities for GRPO Training

Handles training metrics logging and visualization.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class GRPOLogger:
    """Logger for GRPO training metrics."""
    
    def __init__(
        self,
        log_dir: str = "outputs/logs",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize GRPO logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment name with timestamp
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"grpo_{timestamp}"
        
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.jsonl"
        self.summary_file = self.log_dir / f"{experiment_name}_summary.json"
        
        # Metrics storage
        self.metrics_history = []
        self.current_step = 0
        
        logger.info(f"Logging to: {self.log_file}")
    
    def log_step(
        self,
        step: int,
        metrics: Dict[str, Any],
        phase: str = "train"
    ):
        """
        Log metrics for a training step.
        
        Args:
            step: Training step number
            metrics: Dictionary of metrics
            phase: Phase name (train/eval)
        """
        self.current_step = step
        
        # Add metadata
        log_entry = {
            "step": step,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        # Store in history
        self.metrics_history.append(log_entry)
        
        # Write to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def log_grpo_step(
        self,
        step: int,
        reward: float,
        reward_std: float,
        completion_length: float,
        kl: float,
        loss: Optional[float] = None
    ):
        """
        Log GRPO-specific metrics matching notebook format.
        
        Args:
            step: Training step
            reward: Mean reward
            reward_std: Reward standard deviation
            completion_length: Mean completion length
            kl: KL divergence
            loss: Training loss
        """
        metrics = {
            "reward": reward,
            "reward_std": reward_std,
            "completion_length": completion_length,
            "kl": kl,
        }
        
        if loss is not None:
            metrics["training_loss"] = loss
        
        self.log_step(step, metrics, phase="grpo")
        
        # Print in notebook format
        if step % 10 == 0 or step == 1:  # Print every 10 steps
            self._print_grpo_table_row(step, metrics)
    
    def _print_grpo_table_row(self, step: int, metrics: Dict[str, Any]):
        """
        Print metrics in table format like the notebook.
        
        Args:
            step: Step number
            metrics: Metrics dictionary
        """
        if step == 1:
            # Print header
            print("\n" + "="*80)
            print("| Step | Training Loss | reward    | reward_std | completion_length | kl       |")
            print("|------|---------------|-----------|------------|-------------------|----------|")
        
        # Format row
        loss_str = f"{metrics.get('training_loss', 0.0):.6f}"
        reward_str = f"{metrics['reward']:.6f}"
        reward_std_str = f"{metrics['reward_std']:.6f}"
        length_str = f"{metrics['completion_length']:.6f}"
        kl_str = f"{metrics['kl']:.6f}"
        
        print(f"| {step:4d} | {loss_str:13s} | {reward_str:9s} | {reward_std_str:10s} | "
              f"{length_str:17s} | {kl_str:8s} |")
    
    def log_sft_step(
        self,
        step: int,
        loss: float,
        learning_rate: Optional[float] = None,
        grad_norm: Optional[float] = None
    ):
        """
        Log SFT pre-training metrics.
        
        Args:
            step: Training step
            loss: Training loss
            learning_rate: Current learning rate
            grad_norm: Gradient norm
        """
        metrics = {"loss": loss}
        
        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm
        
        self.log_step(step, metrics, phase="sft")
    
    def log_generation_sample(
        self,
        step: int,
        prompt: str,
        generation: str,
        reward: float,
        answer: Optional[str] = None,
        extracted: Optional[str] = None
    ):
        """
        Log a generation sample for debugging.
        
        Args:
            step: Step number
            prompt: Input prompt
            generation: Generated text
            reward: Reward for this generation
            answer: True answer
            extracted: Extracted answer
        """
        sample = {
            "step": step,
            "prompt": prompt[:200],  # Truncate for logging
            "generation": generation[:500],  # Truncate for logging
            "reward": reward,
            "answer": answer,
            "extracted": extracted
        }
        
        # Save to separate samples file
        samples_file = self.log_dir / f"{self.experiment_name}_samples.jsonl"
        with open(samples_file, "a") as f:
            f.write(json.dumps(sample) + "\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get training summary.
        
        Returns:
            Summary dictionary
        """
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        
        summary = {
            "experiment_name": self.experiment_name,
            "total_steps": self.current_step,
            "metrics": {}
        }
        
        # Calculate summary statistics
        for col in df.select_dtypes(include=['number']).columns:
            if col not in ['step', 'timestamp']:
                summary["metrics"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "final": float(df[col].iloc[-1]) if len(df) > 0 else None
                }
        
        return summary
    
    def save_summary(self):
        """Save training summary to file."""
        summary = self.get_summary()
        
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to: {self.summary_file}")
    
    def plot_metrics(self, metrics: Optional[List[str]] = None):
        """
        Plot training metrics.
        
        Args:
            metrics: List of metric names to plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style("whitegrid")
            
            if not self.metrics_history:
                logger.warning("No metrics to plot")
                return
            
            df = pd.DataFrame(self.metrics_history)
            
            # Default metrics to plot
            if metrics is None:
                metrics = ["reward", "kl", "completion_length"]
            
            # Filter available metrics
            metrics = [m for m in metrics if m in df.columns]
            
            if not metrics:
                logger.warning("No valid metrics to plot")
                return
            
            # Create subplots
            fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
            if len(metrics) == 1:
                axes = [axes]
            
            for ax, metric in zip(axes, metrics):
                df.plot(x="step", y=metric, ax=ax)
                ax.set_xlabel("Step")
                ax.set_ylabel(metric)
                ax.set_title(f"{metric} over training")
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.log_dir / f"{self.experiment_name}_metrics.png"
            plt.savefig(plot_file, dpi=100)
            logger.info(f"Metrics plot saved to: {plot_file}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")


def test_logger():
    """Test the GRPO logger."""
    print("Testing GRPO Logger")
    print("=" * 80)
    
    # Initialize logger
    grpo_logger = GRPOLogger(experiment_name="test_run")
    
    # Test SFT logging
    print("\nLogging SFT steps...")
    for i in range(1, 6):
        grpo_logger.log_sft_step(
            step=i,
            loss=1.0 / i,
            learning_rate=0.0001 * (1 - i/10)
        )
    print("✓ SFT steps logged")
    
    # Test GRPO logging
    print("\nLogging GRPO steps...")
    for i in range(1, 11):
        grpo_logger.log_grpo_step(
            step=i,
            reward=0.1 * i,
            reward_std=0.05,
            completion_length=200 - i,
            kl=0.001 * i,
            loss=1.0 / (i + 1)
        )
    
    # Test generation sample logging
    print("\nLogging generation sample...")
    grpo_logger.log_generation_sample(
        step=5,
        prompt="What is 2+2?",
        generation="<start_working_out>Let me think...<end_working_out><SOLUTION>4</SOLUTION>",
        reward=5.0,
        answer="4",
        extracted="4"
    )
    print("✓ Generation sample logged")
    
    # Get and save summary
    print("\nGenerating summary...")
    summary = grpo_logger.get_summary()
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Metrics tracked: {list(summary['metrics'].keys())}")
    
    grpo_logger.save_summary()
    print("✓ Summary saved")
    
    print("\n✓ All logger tests passed!")


if __name__ == "__main__":
    test_logger()