"""
Environment detection utility for the Undress Video Pipeline.
Detects Google Colab, GPU hardware (CUDA/MPS), VRAM, and recommends torch precision.
"""

import sys
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def is_colab() -> bool:
    """Check if code is running inside Google Colab environment."""
    if "google.colab" in sys.modules:
        return True
    if os.environ.get("COLAB_GPU") is not None or os.environ.get("COLAB_RELEASE_TAG") is not None:
        return True
    return False

def get_device_info() -> Dict[str, Any]:
    """
    Inspect system hardware and return torch device capabilities.
    """
    has_cuda = False
    has_mps = False
    vram_gb = 0.0
    device_name = "CPU"

    try:
        import torch
        if torch.cuda.is_available():
            has_cuda = True
            device_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024 ** 3)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            has_mps = True
            device_name = "Apple Silicon MPS"
    except ImportError:
        logger.warning("PyTorch not installed; defaulting to CPU environment info.")

    colab_env = is_colab()
    
    # Recommend precision and device
    if has_cuda:
        device = "cuda"
        precision = "fp16"
    elif has_mps:
        device = "mps"
        precision = "fp32"
    else:
        device = "cpu"
        precision = "fp32"

    return {
        "is_colab": colab_env,
        "has_cuda": has_cuda,
        "has_mps": has_mps,
        "device_name": device_name,
        "vram_gb": round(vram_gb, 2),
        "device": device,
        "precision": precision
    }

def print_environment_summary() -> None:
    """Log clear summary of the runtime environment."""
    info = get_device_info()
    print("=" * 60)
    print(" PIPELINE RUNTIME ENVIRONMENT DETECTED")
    print("=" * 60)
    print(f" Environment : {'Google Colab' if info['is_colab'] else 'Local / Remote Server'}")
    print(f" Target Device: {info['device_name']} ({info['device'].upper()})")
    if info['has_cuda']:
        print(f" VRAM         : {info['vram_gb']} GB")
    print(f" Precision    : {info['precision']}")
    print("=" * 60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_environment_summary()
