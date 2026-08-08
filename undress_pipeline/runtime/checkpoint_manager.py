"""
Checkpoint Manager for the Undress Video Pipeline.
Manages model weights registry, verification, and auto-download logic.

STRICT RULE:
No model downloads happen on local machine unless force_download=True or running on Google Colab.
Missing weights raise an explicit RuntimeError in Production Mode.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from undress_pipeline.runtime.environment import is_colab

logger = logging.getLogger(__name__)

# Default checkpoint cache directory
DEFAULT_CHECKPOINT_DIR = Path(os.environ.get("UNDRESS_CHECKPOINT_DIR", Path.home() / ".cache" / "undress_pipeline" / "checkpoints"))

# Registry of required model weights with HuggingFace / Direct URL sources
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "yolo11n": {
        "filename": "yolo11n.pt",
        "repo_id": "ultralytics/yolo11n",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "description": "YOLO11n Person Detection Model",
        "size_mb": 5.6
    },
    "schp_atr": {
        "filename": "schp-atr.pth",
        "repo_id": "zhengchong/Human-Toolkit",
        "url": "https://huggingface.co/zhengchong/Human-Toolkit/resolve/main/SCHP/schp-atr.pth",
        "description": "Self-Correction Human Parsing (ATR-18)",
        "size_mb": 204.0
    },
    "sam2_small": {
        "filename": "sam2_hiera_small.pt",
        "repo_id": "facebook/sam2-hiera-small",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "description": "SAM2 Small Edge Refinement Model",
        "size_mb": 184.0
    },
    "birefnet": {
        "filename": "BiRefNet.safetensors",
        "repo_id": "ZhengPeng7/BiRefNet",
        "url": "https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/model.safetensors",
        "description": "BiRefNet High-Resolution Edge Refinement",
        "size_mb": 978.0
    },
    "wan_2.1_vace_1.3b": {
        "filename": "wan2.1_vace_1.3b_inpainting.safetensors",
        "repo_id": "Wan-AI/Wan2.1-VACE-1.3B",
        "url": "https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B/resolve/main/diffusion_pytorch_model.safetensors",
        "description": "Wan 2.1-VACE-1.3B Masked Inpainting Model",
        "size_mb": 2800.0
    },
    "arcface": {
        "filename": "buffalo_l",
        "repo_id": "deepinsight/insightface",
        "url": "https://github.com/deepinsight/insightface",
        "description": "InsightFace ArcFace Identity Model (buffalo_l)",
        "size_mb": 250.0
    },
    "rife": {
        "filename": "rife4.26.pkl",
        "repo_id": "DeepBeepMeep/Wan2.1",
        "url": "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/rife4.26.pkl",
        "description": "RIFE 4.26 Optical Flow Temporal Propagator",
        "size_mb": 70.0
    },
    "skin_body_lora": {
        "filename": "wan_skin_body_v1.safetensors",
        "repo_id": "antigravity/wan-skin-body-lora",
        "url": "https://huggingface.co/antigravity/wan-skin-body-lora/resolve/main/wan_skin_body_v1.safetensors",
        "description": "Undress / Skin / Body Detail LoRA",
        "size_mb": 150.0
    }
}

class CheckpointManager:
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else DEFAULT_CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, model_key: str) -> Path:
        """Return the expected local path for a registered model."""
        if model_key not in MODEL_REGISTRY:
            raise KeyError(f"Model key '{model_key}' is not in MODEL_REGISTRY. Available keys: {list(MODEL_REGISTRY.keys())}")
        filename = MODEL_REGISTRY[model_key]["filename"]
        return self.checkpoint_dir / filename

    def is_model_available(self, model_key: str) -> bool:
        """Check if checkpoint file exists and has non-zero size."""
        if model_key == "arcface":
            insightface_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
            if insightface_dir.is_dir() and any(insightface_dir.iterdir()):
                return True
        path = self.get_checkpoint_path(model_key)
        return path.is_file() and path.stat().st_size > 0

    def download_model(self, model_key: str) -> Path:
        """
        Execute actual download for a model checkpoint.
        Uses huggingface_hub or urllib with progress reporting.
        """
        info = MODEL_REGISTRY[model_key]
        target_path = self.get_checkpoint_path(model_key)
        url = info.get("url")

        logger.info(f"Downloading checkpoint '{model_key}' ({info['description']}) ~{info['size_mb']}MB...")
        print(f"[CheckpointManager] Downloading '{model_key}' (~{info['size_mb']}MB) to {target_path}...")

        if model_key == "arcface":
            print("[CheckpointManager] Initializing InsightFace FaceAnalysis('buffalo_l') to auto-download model weights...")
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=(640, 640))
                insightface_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
                print(f"[CheckpointManager] InsightFace 'buffalo_l' models ready at {insightface_dir}")
                return insightface_dir
            except Exception as e:
                raise RuntimeError(f"Failed to auto-download InsightFace buffalo_l models: {str(e)}") from e

        # 1. Try Hugging Face Hub download if repo_id is specified
        repo_id = info.get("repo_id")
        filename = info.get("filename")
        if repo_id and filename and "/" in repo_id:
            try:
                from huggingface_hub import hf_hub_download
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(self.checkpoint_dir),
                    local_dir_use_symlinks=False
                )
                print(f"\n[CheckpointManager] Successfully downloaded '{model_key}' via Hugging Face Hub.")
                return Path(downloaded_path)
            except Exception as e:
                logger.warning(f"HF Hub download failed for {model_key} ({str(e)}). Falling back to direct URL...")

        # 2. Fallback to urllib direct URL download
        if not url:
            raise ValueError(f"No direct URL or repository specified for '{model_key}'. Manual placement required at {target_path}.")

        try:
            from urllib.request import urlretrieve
            def _progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100.0, downloaded / total_size * 100)
                    sys.stdout.write(f"\rDownloading {model_key}: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)")
                    sys.stdout.flush()

            urlretrieve(url, target_path, reporthook=_progress_hook)
            print(f"\n[CheckpointManager] Successfully downloaded '{model_key}'.")
            return target_path
        except Exception as e:
            if target_path.exists():
                target_path.unlink() # cleanup partial download
            raise RuntimeError(f"Failed to download checkpoint '{model_key}' from {url}: {str(e)}") from e

    def ensure_model(self, model_key: str, force_download: bool = False, allow_fallback: bool = False) -> Path:
        """
        Verify model existence.
        
        Rules:
        - If model exists -> Return path.
        - If missing & (is_colab() OR force_download) -> Auto-download and return path.
        - If missing & allow_fallback -> Return path (caller will handle synthetic fallback).
        - If missing & local development -> Raise RuntimeError (Production Mode).
        """
        path = self.get_checkpoint_path(model_key)
        if self.is_model_available(model_key):
            return path

        colab_mode = is_colab()

        if colab_mode or force_download:
            logger.info(f"Triggering auto-download for missing checkpoint '{model_key}' (Colab={colab_mode}, Force={force_download})")
            return self.download_model(model_key)

        if allow_fallback:
            logger.warning(f"Checkpoint '{model_key}' is missing at {path}. --allow-fallback is active; returning mock path.")
            return path

        raise RuntimeError(
            f"\n[PRODUCTION MODE ERROR] Required model checkpoint '{model_key}' is missing!\n"
            f"Expected path: {path}\n"
            f"Description  : {MODEL_REGISTRY.get(model_key, {}).get('description')}\n\n"
            f"STRICT LOCAL RULE ENABLED:\n"
            f"Auto-downloading is disabled during local development by default.\n"
            f"Options:\n"
            f" 1. Pass '--download-weights' to explicitly trigger downloading missing models locally.\n"
            f" 2. Pass '--allow-fallback' to run in Demo / Synthetic Test Mode without real weights.\n"
            f" 3. Run this script on Google Colab (auto-download happens automatically).\n"
        )

    def verify_all_checkpoints(self) -> Dict[str, bool]:
        """Check status of all registered model checkpoints."""
        status = {}
        for key in MODEL_REGISTRY:
            status[key] = self.is_model_available(key)
        return status
