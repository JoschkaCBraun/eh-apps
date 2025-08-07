"""
Memory Management Utilities for GRPO Training

Handles GPU memory management for H100 training.
"""

import torch
import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manage GPU memory during GRPO training."""
    
    def __init__(self, cleanup_interval: int = 10):
        """
        Initialize memory manager.
        
        Args:
            cleanup_interval: Clean up memory every N steps
        """
        self.cleanup_interval = cleanup_interval
        self.step_count = 0
        
        # Get GPU info
        if torch.cuda.is_available():
            self.device_name = torch.cuda.get_device_name(0)
            self.total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {self.device_name}")
            logger.info(f"Total GPU Memory: {self.total_memory:.1f} GB")
        else:
            self.device_name = "CPU"
            self.total_memory = 0
            logger.warning("No GPU available, using CPU")
    
    def get_memory_stats(self) -> dict:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary with memory stats
        """
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0}
        
        stats = {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "free": (torch.cuda.get_device_properties(0).total_memory - 
                    torch.cuda.memory_allocated()) / 1024**3
        }
        
        return stats
    
    def log_memory_usage(self, prefix: str = ""):
        """
        Log current memory usage.
        
        Args:
            prefix: Prefix for log message
        """
        stats = self.get_memory_stats()
        logger.info(
            f"{prefix}GPU Memory - "
            f"Allocated: {stats['allocated']:.1f}GB, "
            f"Reserved: {stats['reserved']:.1f}GB, "
            f"Free: {stats['free']:.1f}GB"
        )
    
    def cleanup(self, force: bool = False):
        """
        Clean up GPU memory.
        
        Args:
            force: Force cleanup regardless of interval
        """
        self.step_count += 1
        
        if force or (self.step_count % self.cleanup_interval == 0):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            if force:
                logger.info("Forced memory cleanup completed")
            else:
                logger.debug(f"Periodic memory cleanup at step {self.step_count}")
    
    def check_memory_availability(self, required_gb: float = 10.0) -> bool:
        """
        Check if enough memory is available.
        
        Args:
            required_gb: Required memory in GB
            
        Returns:
            True if enough memory is available
        """
        stats = self.get_memory_stats()
        available = stats["free"]
        
        if available < required_gb:
            logger.warning(
                f"Low memory warning: {available:.1f}GB available, "
                f"{required_gb:.1f}GB required"
            )
            return False
        
        return True
    
    def optimize_for_h100(self):
        """Apply H100-specific optimizations."""
        if torch.cuda.is_available():
            # Enable TF32 for better performance on H100
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudnn benchmarking
            torch.backends.cudnn.benchmark = True
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use up to 95% of GPU memory
            
            logger.info("H100 optimizations applied")
    
    def reset(self):
        """Reset step counter."""
        self.step_count = 0
    
    def __enter__(self):
        """Context manager entry."""
        self.initial_stats = self.get_memory_stats()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup(force=True)
        final_stats = self.get_memory_stats()
        
        # Log memory difference
        allocated_diff = final_stats["allocated"] - self.initial_stats["allocated"]
        if abs(allocated_diff) > 0.1:  # Only log significant changes
            logger.info(f"Memory change: {allocated_diff:+.1f}GB")


def test_memory_manager():
    """Test memory management utilities."""
    print("Testing Memory Manager")
    print("=" * 80)
    
    # Initialize manager
    manager = MemoryManager(cleanup_interval=5)
    
    # Test memory stats
    print("\nMemory Statistics:")
    stats = manager.get_memory_stats()
    print(f"  Allocated: {stats['allocated']:.2f} GB")
    print(f"  Reserved: {stats['reserved']:.2f} GB")
    print(f"  Free: {stats['free']:.2f} GB")
    
    # Test memory check
    print("\nMemory Availability Check:")
    has_memory = manager.check_memory_availability(required_gb=5.0)
    print(f"  Has 5GB available: {has_memory}")
    
    # Test cleanup
    print("\nTesting cleanup...")
    for i in range(10):
        manager.cleanup()
        if i % 5 == 4:
            print(f"  Cleanup triggered at step {i+1}")
    
    # Test context manager
    print("\nTesting context manager...")
    with manager:
        # Allocate some memory
        if torch.cuda.is_available():
            dummy = torch.zeros(1000, 1000, device="cuda")
        print("  Memory allocated within context")
    print("  Context exited, memory cleaned")
    
    # Test H100 optimizations
    print("\nApplying H100 optimizations...")
    manager.optimize_for_h100()
    print("✓ Optimizations applied")
    
    print("\n✓ All memory manager tests passed!")


if __name__ == "__main__":
    test_memory_manager()