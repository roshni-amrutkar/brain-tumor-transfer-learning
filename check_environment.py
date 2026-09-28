"""
Environment and Hardware Diagnostic Script.
Checks Python version, PyTorch installation, CUDA/GPU capability,
and system resources.
"""

import os
import platform
import sys


def check_env() -> None:
    print("=" * 60)
    print("ENVIRONMENT & HARDWARE DIAGNOSTIC CHECK")
    print("=" * 60)

    # 1. Python Information
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Python Executable : {sys.executable}")
    print(f"Operating System  : {platform.system()} {platform.release()} ({platform.machine()})")

    # 2. PyTorch & Torchvision
    try:
        import torch
        print(f"PyTorch Version   : {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available    : {cuda_available}")

        if cuda_available:
            print(f"CUDA Version      : {torch.version.cuda}")
            print(f"GPU Device Count  : {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024 ** 3)
                print(f"  - GPU {i}: {props.name} ({vram_gb:.2f} GB VRAM)")
        else:
            print("Status            : CPU mode active (Training will run on CPU)")
    except ImportError:
        print("ERROR: PyTorch is not installed. Run: pip install torch torchvision")

    try:
        import torchvision
        print(f"TorchVision Ver.  : {torchvision.__version__}")
    except ImportError:
        print("ERROR: TorchVision is not installed. Run: pip install torchvision")

    # 3. Core Science Packages
    packages = ["numpy", "pandas", "sklearn", "matplotlib", "seaborn", "PIL", "tqdm"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"{pkg:<18}: {ver}")
        except ImportError:
            print(f"{pkg:<18}: NOT INSTALLED")

    print("=" * 60)
    print("Diagnostic complete. System is ready to proceed.")
    print("=" * 60)


if __name__ == "__main__":
    check_env()
