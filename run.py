#!/usr/bin/env python3
"""
Undress Video Pipeline — Main CLI Runner
Supports Google Colab T4 / RTX 3050 & Local CPU Demo Mode.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Optional

# Ensure package is on python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from undress_pipeline.runtime.environment import print_environment_summary, get_device_info
from undress_pipeline.runtime.checkpoint_manager import CheckpointManager, MODEL_REGISTRY
from undress_pipeline.runtime.lora_loader import LoRALoader
from undress_pipeline.pipeline.undress_pipeline import UndressVideoPipeline

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Undress Video Pipeline — Production Video Reconstruction Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input clothed video file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save reconstructed output video")
    parser.add_argument("--keyframe-interval", type=int, default=5, help="Keyframe interval for Wan 2.1 masked inpainting")
    parser.add_argument("--candidates", type=int, default=2, help="Number of candidate keyframe variations for ArcFace identity ranking")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to store/load model checkpoints")
    parser.add_argument("--download-weights", action="store_true", help="Explicitly allow downloading model weights on local machine")
    parser.add_argument("--allow-fallback", action="store_true", help="Enable Demo/Synthetic fallback mode if weights or CUDA are missing")
    parser.add_argument("--debug-masks", action="store_true", help="Export debug clothing and region-lock mask visualizations")
    parser.add_argument("--lora-path", type=str, default=None, help="Path to custom skin/body LoRA checkpoint")
    parser.add_argument("--lora-scale", type=float, default=0.8, help="LoRA influence scale factor (0.0 - 1.0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(args.verbose)

    print("\n" + "=" * 60)
    print("       UNDRESS VIDEO PIPELINE v1.0.0 (Production Engine)")
    print("=" * 60)

    # 1. Environment Check
    print_environment_summary()
    env_info = get_device_info()

    # 2. Checkpoint Verification
    checkpoint_mgr = CheckpointManager(checkpoint_dir=args.checkpoint_dir)
    print("\nVerifying Model Weights Registry...")

    missing_mandatory_models = []
    missing_optional_models = []

    for key in MODEL_REGISTRY:
        available = checkpoint_mgr.is_model_available(key)
        is_opt = MODEL_REGISTRY[key].get("optional", False)
        status_str = "PRESENT" if available else ("MISSING (OPTIONAL)" if is_opt else "MISSING")
        print(f" - {key:<20}: {status_str} ({MODEL_REGISTRY[key]['description']})")
        if not available:
            if is_opt:
                missing_optional_models.append(key)
            else:
                missing_mandatory_models.append(key)

    if missing_mandatory_models:
        print(f"\nFound {len(missing_mandatory_models)} missing mandatory model checkpoint(s): {missing_mandatory_models}")
        if env_info["is_colab"] or args.download_weights:
            print("Auto-download trigger condition met. Attempting downloads for mandatory models...")
            for key in missing_mandatory_models:
                checkpoint_mgr.ensure_model(key, force_download=args.download_weights)
        elif args.allow_fallback:
            print("\n[DEMO MODE ACTIVE] --allow-fallback specified. Pipeline will proceed using synthetic mock models.")
        else:
            raise RuntimeError(
                f"\n[PRODUCTION MODE ERROR] Mandatory model weights are missing!\n"
                f"Missing mandatory keys: {missing_mandatory_models}\n\n"
                f"STRICT LOCAL RULE ENFORCED:\n"
                f"No automatic weight downloads are performed on local machines.\n"
                f"To fix:\n"
                f" - Pass --download-weights to download model files locally.\n"
                f" - Pass --allow-fallback to run in CPU Demo/Synthetic test mode.\n"
                f" - Run on Google Colab where auto-downloads execute automatically.\n"
            )
    else:
        print("\nAll mandatory model checkpoints are available!")

    # Process optional models auto-download if on Colab / --download-weights
    if missing_optional_models and (env_info["is_colab"] or args.download_weights):
        print(f"Attempting download for optional model(s): {missing_optional_models}...")
        for key in missing_optional_models:
            try:
                checkpoint_mgr.ensure_model(key, force_download=args.download_weights)
            except Exception as e:
                print(f"Optional model '{key}' download skipped/failed ({str(e)}). Continuing without optional model.")

    # Initialize LoRALoader if LoRA specified
    lora_loader = LoRALoader(max_loras=2)
    if args.lora_path:
        lora_loader.register_lora(Path(args.lora_path), scale=args.lora_scale, name="custom_skin_lora")
    elif not checkpoint_mgr.is_model_available("skin_body_lora"):
        print("\n[INFO] No LoRA loaded. Running base Wan only.")

    # Initialize Master Undress Video Pipeline
    pipeline = UndressVideoPipeline(
        checkpoint_mgr=checkpoint_mgr,
        lora_loader=lora_loader,
        allow_fallback=args.allow_fallback
    )

    print("\nExecuting Master End-to-End Undress Video Pipeline...")
    input_path = Path(args.input)
    output_path = Path(args.output)

    results = pipeline.process_video(
        input_video_path=input_path,
        output_video_path=output_path,
        keyframe_interval=args.keyframe_interval,
        candidate_count=args.candidates,
        debug_masks=args.debug_masks
    )

    print("\n[Video Pipeline Execution Complete]")
    print(f" Output Video    : {results['output_path']}")
    print(f" Resolution      : {results['width']}x{results['height']}")
    print(f" Frame Rate      : {results['fps']:.2f} FPS")
    print(f" Total Frames    : {results['total_frames']}")
    print(f" Keyframes Inpaint: {results['keyframe_count']}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
