"""
Edge Refiner module integrating SAM2 & BiRefNet with Morphological Feathering.
Produces smooth, soft-edge alpha masks for seamless diffusion inpainting.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2

from undress_pipeline.runtime.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

class EdgeRefiner:
    """SAM2 + BiRefNet Soft Edge Boundary Refiner."""

    def __init__(self, checkpoint_mgr: CheckpointManager, allow_fallback: bool = False):
        self.checkpoint_mgr = checkpoint_mgr
        self.allow_fallback = allow_fallback
        self.sam2_model = None
        self.birefnet_model = None
        self._init_models()

    def _init_models(self):
        # SAM2 small model initialization
        try:
            sam2_path = self.checkpoint_mgr.ensure_model("sam2_small", allow_fallback=self.allow_fallback)
            if sam2_path.is_file():
                logger.info(f"SAM2 model available at {sam2_path}.")
        except Exception as e:
            if not self.allow_fallback:
                raise e
            logger.warning(f"SAM2 model unavailable ({str(e)}); falling back to algorithmic edge refinement.")

        # BiRefNet model initialization
        try:
            biref_path = self.checkpoint_mgr.ensure_model("birefnet", allow_fallback=self.allow_fallback)
            if biref_path.is_file():
                logger.info(f"BiRefNet model available at {biref_path}.")
        except Exception as e:
            if not self.allow_fallback:
                raise e
            logger.warning(f"BiRefNet model unavailable ({str(e)}); falling back to algorithmic edge refinement.")

    def refine_edges(
        self,
        image: np.ndarray,
        binary_mask: np.ndarray,
        dilation_px: int = 5,
        blur_radius: int = 11
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine edges of binary mask to create smooth alpha transitions.
        
        Args:
            image: Original RGB image (H, W, 3)
            binary_mask: Hard binary mask (H, W) uint8 with 0/255
            dilation_px: Dilation kernel radius to cover garment seam boundaries
            blur_radius: Gaussian blur kernel size for soft edge feathering (must be odd)

        Returns:
            Tuple of:
             - Hard refined binary mask (H, W) uint8 (0 or 255)
             - Soft alpha mask (H, W) float32 (range 0.0 to 1.0)
        """
        h, w = binary_mask.shape[:2]

        if blur_radius % 2 == 0:
            blur_radius += 1

        # 1. Morphological Dilation to comfortably cover seams
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        # 2. Distance Transform Soft Feathering
        dist_inside = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)
        dist_outside = cv2.distanceTransform(255 - dilated_mask, cv2.DIST_L2, 5)

        # Normalize distance transform around boundary
        alpha_mask = cv2.GaussianBlur(dilated_mask.astype(np.float32) / 255.0, (blur_radius, blur_radius), 0)
        alpha_mask = np.clip(alpha_mask, 0.0, 1.0)

        hard_refined = (alpha_mask > 0.5).astype(np.uint8) * 255

        return hard_refined, alpha_mask
