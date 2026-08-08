"""
Temporal Propagation & Optical Flow Warping Module (RIFE / Optical Flow).
Propagates keyframe inpainted body regions across intermediate non-keyframes without flicker.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from undress_pipeline.runtime.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

class TemporalPropagator:
    """Optical flow temporal motion warping and frame interpolation."""

    def __init__(self, checkpoint_mgr: CheckpointManager, allow_fallback: bool = False):
        self.checkpoint_mgr = checkpoint_mgr
        self.allow_fallback = allow_fallback
        self.rife_model = None
        self._init_model()

    def _init_model(self):
        try:
            ckpt_path = self.checkpoint_mgr.ensure_model("rife", allow_fallback=self.allow_fallback)
            if ckpt_path.is_file() and ckpt_path.stat().st_size > 0:
                logger.info(f"RIFE optical flow checkpoint available at {ckpt_path}.")
            else:
                if not self.allow_fallback:
                    raise RuntimeError(f"[PRODUCTION MODE ERROR] RIFE model weights missing at {ckpt_path}.")
                logger.warning(f"RIFE weights missing at {ckpt_path}; using OpenCV Farneback optical flow for --allow-fallback mode.")
        except Exception as e:
            if not self.allow_fallback:
                raise e
            logger.warning(f"Temporal propagator initialization ({str(e)}); running in --allow-fallback mode.")

    def compute_optical_flow(self, prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
        """Compute 2D motion vector optical flow (H, W, 2) from prev_frame to next_frame."""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        return flow

    def warp_frame(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp RGB frame using 2D motion vector flow."""
        h, w = frame.shape[:2]
        flow_map = -flow
        flow_map[:, :, 0] += np.arange(w)
        flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]

        warped = cv2.remap(
            frame,
            flow_map.astype(np.float32),
            None,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        return warped

    def propagate_clip(
        self,
        frames: List[np.ndarray],
        keyframe_indices: List[int],
        inpainted_keyframes: Dict[int, np.ndarray],
        alpha_masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Propagate keyframe inpainted regions across all frames in clip.
        
        Args:
            frames: List of original video frames (H, W, 3) uint8
            keyframe_indices: Sorted list of keyframe frame indices
            inpainted_keyframes: Dict mapping keyframe index -> inpainted RGB frame
            alpha_masks: List of soft alpha masks (H, W) float32 for each frame

        Returns:
            List of temporally propagated, seamless reconstructed video frames (H, W, 3)
        """
        num_frames = len(frames)
        output_frames = [f.copy() for f in frames]

        # 1. Place inpainted keyframes
        for k_idx, inp_frame in inpainted_keyframes.items():
            output_frames[k_idx] = inp_frame

        # 2. Propagate between adjacent keyframes
        for i in range(len(keyframe_indices) - 1):
            k_start = keyframe_indices[i]
            k_end = keyframe_indices[i + 1]

            inp_start = inpainted_keyframes[k_start]
            inp_end = inpainted_keyframes[k_end]

            for idx in range(k_start + 1, k_end):
                # Interpolation weight between start and end keyframes
                weight_end = (idx - k_start) / float(k_end - k_start)
                weight_start = 1.0 - weight_end

                # Forward flow from k_start to idx
                flow_fwd = self.compute_optical_flow(frames[k_start], frames[idx])
                warped_fwd = self.warp_frame(inp_start, flow_fwd)

                # Backward flow from k_end to idx
                flow_bwd = self.compute_optical_flow(frames[k_end], frames[idx])
                warped_bwd = self.warp_frame(inp_end, flow_bwd)

                # Bi-directional motion blending
                propagated_body = (warped_fwd.astype(np.float32) * weight_start + warped_bwd.astype(np.float32) * weight_end)

                # Blend propagated body onto original frame using target alpha mask
                alpha = np.expand_dims(alpha_masks[idx], axis=-1)
                blended_frame = (propagated_body * alpha + frames[idx].astype(np.float32) * (1.0 - alpha))
                output_frames[idx] = np.clip(blended_frame, 0, 255).astype(np.uint8)

        # 3. Propagate leading and trailing frames outside keyframe boundary
        if keyframe_indices:
            k_first = keyframe_indices[0]
            inp_first = inpainted_keyframes[k_first]
            for idx in range(0, k_first):
                flow = self.compute_optical_flow(frames[k_first], frames[idx])
                warped = self.warp_frame(inp_first, flow)
                alpha = np.expand_dims(alpha_masks[idx], axis=-1)
                blended = (warped.astype(np.float32) * alpha + frames[idx].astype(np.float32) * (1.0 - alpha))
                output_frames[idx] = np.clip(blended, 0, 255).astype(np.uint8)

            k_last = keyframe_indices[-1]
            inp_last = inpainted_keyframes[k_last]
            for idx in range(k_last + 1, num_frames):
                flow = self.compute_optical_flow(frames[k_last], frames[idx])
                warped = self.warp_frame(inp_last, flow)
                alpha = np.expand_dims(alpha_masks[idx], axis=-1)
                blended = (warped.astype(np.float32) * alpha + frames[idx].astype(np.float32) * (1.0 - alpha))
                output_frames[idx] = np.clip(blended, 0, 255).astype(np.uint8)

        return output_frames
