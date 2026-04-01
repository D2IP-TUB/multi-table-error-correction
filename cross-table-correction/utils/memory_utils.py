import os

import psutil


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # In MB
    print(f"Memory usage: {mem:.2f} MB")
