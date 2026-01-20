"""Utility functions for profiling, timing, and helper operations."""

import functools
import logging
import time
from typing import Callable, Any

logger = logging.getLogger(__name__)


def timeit(func: Callable) -> Callable:
    """Decorator to measure function execution time.

    Args:
        func: Function to be timed

    Returns:
        Wrapped function that logs execution time

    Example:
        @timeit
        def my_function():
            time.sleep(1)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        logger.info(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result

    return wrapper


class Timer:
    """Context manager for timing code blocks.

    Example:
        with Timer("data loading"):
            data = load_data()
    """

    def __init__(self, name: str = "Operation", log_level: int = logging.INFO):
        """Initialize timer.

        Args:
            name: Name of the operation being timed
            log_level: Logging level for output
        """
        self.name = name
        self.log_level = log_level
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.perf_counter()
        logger.log(self.log_level, f"{self.name} started...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and log elapsed time."""
        self.elapsed = time.perf_counter() - self.start_time
        logger.log(self.log_level, f"{self.name} completed in {self.elapsed:.4f} seconds")
        return False


class ProfilerContext:
    """Context manager for profiling code with detailed statistics.

    Uses cProfile to generate detailed profiling information.

    Example:
        with ProfilerContext("training_loop") as profiler:
            train_model()
        profiler.print_stats(sort_by='cumulative', top_n=10)
    """

    def __init__(self, name: str = "Profile"):
        """Initialize profiler context.

        Args:
            name: Name for the profiling session
        """
        self.name = name
        self.profiler = None

    def __enter__(self):
        """Start profiling."""
        import cProfile

        self.profiler = cProfile.Profile()
        self.profiler.enable()
        logger.info(f"Profiling started: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling."""
        self.profiler.disable()
        logger.info(f"Profiling completed: {self.name}")
        return False

    def print_stats(self, sort_by: str = "cumulative", top_n: int = 20):
        """Print profiling statistics.

        Args:
            sort_by: Sort key ('cumulative', 'time', 'calls', etc.)
            top_n: Number of top entries to display
        """
        import pstats
        import io

        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats(sort_by)
        ps.print_stats(top_n)

        logger.info(f"\nProfiling results for {self.name}:")
        logger.info(s.getvalue())

    def save_stats(self, filename: str):
        """Save profiling statistics to file.

        Args:
            filename: Output filename for stats
        """

        self.profiler.dump_stats(filename)
        logger.info(f"Profiling stats saved to {filename}")


def profile_memory(func: Callable) -> Callable:
    """Decorator to profile memory usage of a function.

    Requires: pip install memory_profiler

    Args:
        func: Function to profile

    Returns:
        Wrapped function with memory profiling
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            from memory_profiler import memory_usage

            mem_before = memory_usage()[0]
            result = func(*args, **kwargs)
            mem_after = memory_usage()[0]
            mem_diff = mem_after - mem_before

            logger.info(
                f"{func.__name__} memory usage: "
                f"before={mem_before:.2f}MB, after={mem_after:.2f}MB, "
                f"diff={mem_diff:.2f}MB"
            )
            return result
        except ImportError:
            logger.warning("memory_profiler not installed. Skipping memory profiling.")
            return func(*args, **kwargs)

    return wrapper


def get_model_summary(model: Any, input_size: tuple = (1, 3, 32, 32)) -> str:
    """Get a summary of model architecture and parameters.

    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)

    Returns:
        String summary of model
    """
    from io import StringIO
    import sys

    # Capture model summary
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()

    try:
        # Try using torchinfo if available
        from torchinfo import summary

        summary(model, input_size=input_size, verbose=0)
    except ImportError:
        # Fallback to basic summary
        print(model)
        print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    sys.stdout = old_stdout
    return buffer.getvalue()


if __name__ == "__main__":
    # Test profiling utilities
    logging.basicConfig(level=logging.INFO)

    @timeit
    def test_function():
        """Test function for timing."""
        time.sleep(0.1)
        return "done"

    # Test decorator
    result = test_function()

    # Test context manager
    with Timer("test operation"):
        time.sleep(0.05)

    # Test profiler
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    with ProfilerContext("factorial test") as profiler:
        for i in range(100):
            factorial(20)

    profiler.print_stats(top_n=5)

    print("Profiling utilities test complete!")
