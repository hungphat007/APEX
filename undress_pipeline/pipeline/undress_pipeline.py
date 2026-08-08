"""
Master End-to-End Undress Video Pipeline.
Orchestrates video loading, SCHP mask composition, keyframe selection, Wan 2.1 inpainting,
ArcFace identity candidate ranking, temporal optical flow propagation, and video writing.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

from undress_pipeline.runtime.checkpoint_manager import CheckpointManager
from undress_pipeline.runtime.lora_loader import LoRALoader
from undress_pipeline.runtime.wan_runtime import WanRuntime
from undress_pipeline.mask.mask_composer import MaskComposer
from undress_pipeline.identity.arcface_ranker import ArcFaceRanker
from undress_pipeline.temporal.propagator import TemporalPropagator

logger = logging.getLogger(__name__)

class UndressVideoPipeline:
    """Master Undress-Only Video Pipeline Controller."""

    def __init__(
        self,
        checkpoint_mgr: CheckpointManager,
        lora_loader: Optional[LoRALoader] = None,
        allow_fallback: bool = False
    ):
        self.checkpoint_mgr = checkpoint_mgr
        self.allow_fallback = allow_fallback
        
        self.mask_composer = MaskComposer(checkpoint_mgr, allow_fallback=allow_fallback)
        self.wan_runtime = WanRuntime(checkpoint_mgr, lora_loader=lora_loader, allow_fallback=allow_fallback)
        self.arcface_ranker = ArcFaceRanker(checkpoint_mgr, allow_fallback=allow_fallback)
        self.temporal_propagator = TemporalPropagator(checkpoint_mgr, allow_fallback=allow_fallback)

    def process_video(
        self,
        input_video_path: Path,
        output_video_path: Path,
        keyframe_interval: int = 5,
        candidate_count: int = 2,
        debug_masks: bool = False
    ) -> Dict[str, Any]:
        """
        Process clothed video end-to-end and save reconstructed video preserving exact resolution & FPS.
        """
        input_path = Path(input_video_path)
        output_path = Path(output_video_path)

        if not input_path.exists():
            if self.allow_fallback:
                logger.warning(f"Input video '{input_path}' not found. Generating synthetic clip for demo testing...")
                return self._process_synthetic_clip(output_path, keyframe_interval, candidate_count, debug_masks)
            else:
                raise FileNotFoundError(f"Input video file not found: {input_path}")

        # 1. Open Video & Read Metadata
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video Loaded: {width}x{height} @ {fps:.2f} FPS | Total Frames: {total_frames}")

        frames_rgb = []
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames_rgb:
            raise ValueError(f"No frames read from video file: {input_path}")

        return self.process_frames(
            frames=frames_rgb,
            fps=fps,
            output_video_path=output_path,
            keyframe_interval=keyframe_interval,
            candidate_count=candidate_count,
            debug_masks=debug_masks
        )

    def process_frames(
        self,
        frames: List[np.ndarray],
        fps: float,
        output_video_path: Path,
        keyframe_interval: int = 5,
        candidate_count: int = 2,
        debug_masks: bool = False
    ) -> Dict[str, Any]:
        """
        Processes list of RGB numpy frames end-to-end and writes output video file.
        """
        num_frames = len(frames)
        h, w = frames[0].shape[:2]

        logger.info(f"Step 1/5: Composing Clothing Masks across {num_frames} frames...")
        mask_results_list = []
        alpha_masks = []
        binary_masks = []

        for idx, frame in enumerate(frames):
            m_res = self.mask_composer.compose_mask(frame)
            mask_results_list.append(m_res)
            alpha_masks.append(m_res["final_alpha_mask"])
            binary_masks.append(m_res["final_binary_mask"])

        # Step 2: Keyframe Selection (sampling 4 to 8 keyframes min)
        keyframe_indices = list(range(0, num_frames, max(1, keyframe_interval)))
        if (num_frames - 1) not in keyframe_indices:
            keyframe_indices.append(num_frames - 1)
        logger.info(f"Step 2/5: Selected {len(keyframe_indices)} Keyframes: {keyframe_indices}")

        # Step 3: Wan 2.1 Inpainting + ArcFace Identity Candidate Ranking on Keyframes
        logger.info(f"Step 3/5: Wan 2.1 Inpainting with {candidate_count} Candidate ArcFace Ranking per keyframe...")
        inpainted_keyframes: Dict[int, np.ndarray] = {}

        for k_idx in keyframe_indices:
            frame_rgb = frames[k_idx]
            target_binary = binary_masks[k_idx]
            target_alpha = alpha_masks[k_idx]

            candidates = []
            for c_i in range(candidate_count):
                seed = 42 + k_idx * 10 + c_i
                cand_inpainted = self.wan_runtime.inpaint_single_frame(
                    image_rgb=frame_rgb,
                    mask_binary=target_binary,
                    alpha_mask=target_alpha,
                    seed=seed
                )
                candidates.append(cand_inpainted)

            # ArcFace Identity Ranking
            best_idx, best_cand, scores = self.arcface_ranker.rank_candidates(
                reference_frame=frame_rgb,
                candidates=candidates
            )
            inpainted_keyframes[k_idx] = best_cand

        # Step 4: Temporal Optical Flow Propagation across non-keyframes
        logger.info("Step 4/5: Executing RIFE/Optical Flow Temporal Propagation...")
        final_reconstructed_frames = self.temporal_propagator.propagate_clip(
            frames=frames,
            keyframe_indices=keyframe_indices,
            inpainted_keyframes=inpainted_keyframes,
            alpha_masks=alpha_masks
        )

        # Step 5: Encode and Save Output Video
        logger.info(f"Step 5/5: Encoding output video ({w}x{h} @ {fps:.2f} FPS) -> {output_video_path}...")
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        for frame in final_reconstructed_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        # Debug Masks Video Export if requested
        if debug_masks:
            debug_video_path = output_video_path.parent / f"{output_video_path.stem}_debug_masks.mp4"
            debug_writer = cv2.VideoWriter(str(debug_video_path), fourcc, fps, (w, h))
            for idx, frame in enumerate(frames):
                vis_frame = self.mask_composer.generate_debug_visualization(frame, mask_results_list[idx])
                debug_writer.write(cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
            debug_writer.release()
            logger.info(f"Exported debug mask video visualization to {debug_video_path}")

        logger.info(f"Video Processing Complete! Saved to {output_video_path}")
        return {
            "output_path": output_video_path,
            "width": w,
            "height": h,
            "fps": fps,
            "total_frames": num_frames,
            "keyframe_count": len(keyframe_indices)
        }

    def _process_synthetic_clip(
        self,
        output_video_path: Path,
        keyframe_interval: int,
        candidate_count: int,
        debug_masks: bool
    ) -> Dict[str, Any]:
        """Generates synthetic test video frames for CPU demo testing."""
        h, w, fps, num_frames = 480, 640, 24.0, 15
        synth_frames = []
        for i in range(num_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :] = [180 + i * 2, 180, 180] # subtle motion gradient
            synth_frames.append(frame)

        return self.process_frames(
            frames=synth_frames,
            fps=fps,
            output_video_path=output_video_path,
            keyframe_interval=keyframe_interval,
            candidate_count=candidate_count,
            debug_masks=debug_masks
        )
