import platform
import sys

import numpy as np
import psutil


def main() -> None:
    print("=== System Configuration ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {platform.processor()}")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()