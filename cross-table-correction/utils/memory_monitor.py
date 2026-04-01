"""
Memory monitoring utility to track and limit application memory usage.
Provides functions to check memory usage and gracefully terminate if threshold is exceeded.
"""

import logging
import os
import sys
import psutil


class MemoryMonitor:
    """Monitor system memory usage and enforce limits."""

    def __init__(self, memory_limit_percent: float = 80.0):
        """
        Initialize memory monitor.
        
        Args:
            memory_limit_percent: Memory usage threshold (0-100). When exceeded, app will exit.
        """
        self.memory_limit_percent = memory_limit_percent
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
        
        logging.info(
            f"MemoryMonitor initialized with {memory_limit_percent}% system memory limit"
        )
        logging.info(f"Initial memory usage: {self.initial_memory:.1f}%")

    def get_memory_usage(self) -> float:
        """
        Get current system memory usage percentage.
        
        Returns:
            float: Memory usage as percentage of total system memory
        """
        try:
            return psutil.virtual_memory().percent
        except Exception as e:
            logging.warning(f"Failed to get memory usage: {e}")
            return 0.0

    def get_process_memory(self) -> float:
        """
        Get current process memory usage in MB.
        
        Returns:
            float: Process memory in MB
        """
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            logging.warning(f"Failed to get process memory: {e}")
            return 0.0

    def check_memory(self, log_prefix: str = "") -> bool:
        """
        Check if memory usage exceeds the limit and log current stats.
        
        Args:
            log_prefix: Optional prefix for log messages
            
        Returns:
            bool: True if within limit, False if exceeded
        """
        memory_percent = self.get_memory_usage()
        process_memory = self.get_process_memory()
        
        status = "OK" if memory_percent < self.memory_limit_percent else "CRITICAL"
        prefix = f"{log_prefix} - " if log_prefix else ""
        
        logging.info(
            f"{prefix}[Memory {status}] System: {memory_percent:.1f}% | "
            f"Process: {process_memory:.1f}MB | Limit: {self.memory_limit_percent}%"
        )
        
        return memory_percent < self.memory_limit_percent

    def check_and_enforce(self, log_prefix: str = "") -> None:
        """
        Check memory usage and terminate app if threshold exceeded.
        
        Args:
            log_prefix: Optional prefix for log messages
            
        Raises:
            SystemExit: If memory limit is exceeded
        """
        memory_percent = self.get_memory_usage()
        process_memory = self.get_process_memory()
        
        if memory_percent >= self.memory_limit_percent:
            prefix = f"{log_prefix} - " if log_prefix else ""
            error_msg = (
                f"{prefix}MEMORY LIMIT EXCEEDED! "
                f"System: {memory_percent:.1f}% (limit: {self.memory_limit_percent}%) | "
                f"Process: {process_memory:.1f}MB. Terminating application to prevent system crash."
            )
            logging.critical(error_msg)
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            sys.exit(1)
        
        # Log status if approaching limit
        if memory_percent >= (self.memory_limit_percent * 0.9):
            prefix = f"{log_prefix} - " if log_prefix else ""
            warning_msg = (
                f"{prefix}[Memory WARNING] Approaching limit! "
                f"System: {memory_percent:.1f}% | Process: {process_memory:.1f}MB"
            )
            logging.warning(warning_msg)

    def get_memory_stats(self) -> dict:
        """
        Get comprehensive memory statistics.
        
        Returns:
            dict: Dictionary with memory statistics
        """
        vm = psutil.virtual_memory()
        return {
            "system_percent": vm.percent,
            "total_gb": vm.total / (1024**3),
            "available_gb": vm.available / (1024**3),
            "used_gb": vm.used / (1024**3),
            "process_mb": self.get_process_memory(),
            "limit_percent": self.memory_limit_percent,
            "within_limit": vm.percent < self.memory_limit_percent,
        }
