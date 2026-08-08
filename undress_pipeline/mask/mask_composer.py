"""
Mask Composer module combining person detection, clothing segmentation, soft edge refinement,
and region locking into the final inpainting mask with visual debug mask export support.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2

from undress_pipeline.runtime.checkpoint_manager import CheckpointManager
from undress_pipeline.mask.person_detector import PersonDetector
from undress_pipeline.mask.clothing_segmentor import ClothingSegmentor
from undress_pipeline.mask.edge_refiner import EdgeRefiner
from undress_pipeline.mask.region_lock import RegionLock

logger = logging.getLogger(__name__)

class MaskComposer:
    """Master Mask Composition pipeline enforcing target=clothing-only rule."""

    def __init__(self, checkpoint_mgr: CheckpointManager, allow_fallback: bool = False):
        self.checkpoint_mgr = checkpoint_mgr
        self.allow_fallback = allow_fallback
        
        self.person_detector = PersonDetector(checkpoint_mgr, allow_fallback=allow_fallback)
        self.clothing_segmentor = ClothingSegmentor(checkpoint_mgr, allow_fallback=allow_fallback)
        self.edge_refiner = EdgeRefiner(checkpoint_mgr, allow_fallback=allow_fallback)
        self.region_lock = RegionLock(safety_margin_px=5)

    def compose_mask(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Processes an RGB image frame end-to-end to generate final clothing inpainting mask.
        
        Returns dict containing:
         - 'final_binary_mask': (H, W) uint8 (255 for target clothing, 0 elsewhere)
         - 'final_alpha_mask': (H, W) float32 (0.0 to 1.0 soft mask)
         - 'parse_map': (H, W) SCHP category map
         - 'protection_mask': (H, W) uint8 locked region mask
         - 'raw_clothing_mask': (H, W) uint8 SCHP clothing mask
         - 'boxes': person bounding boxes list
        """
        h, w = image.shape[:2]

        # 1. Person Detection
        boxes = self.person_detector.detect(image)

        # 2. SCHP ATR-18 Clothing Parsing
        parse_map = self.clothing_segmentor.parse_image(image)
        raw_clothing_mask = self.clothing_segmentor.get_clothing_mask(parse_map)

        # 3. Region Lock Protection Mask (Face, Hair, Hands, Skin, Background)
        protection_mask = self.region_lock.generate_protection_mask(parse_map)

        # 4. Enforce Lock Rule: Target = Clothing Mask AND NOT Protection Mask
        constrained_clothing_mask = cv2.bitwise_and(raw_clothing_mask, cv2.bitwise_not(protection_mask))

        # 5. Soft Edge Refinement
        final_binary_raw, final_alpha_raw = self.edge_refiner.refine_edges(
            image=image,
            binary_mask=constrained_clothing_mask,
            dilation_px=3,
            blur_radius=9
        )

        # 6. Re-enforce Strict Region Lock: Zero target clothing pixels allowed on protected regions
        protection_not = cv2.bitwise_not(protection_mask)
        final_binary = cv2.bitwise_and(final_binary_raw, protection_not)
        final_alpha = final_alpha_raw * (protection_not.astype(np.float32) / 255.0)

        return {
            "final_binary_mask": final_binary,
            "final_alpha_mask": final_alpha,
            "parse_map": parse_map,
            "protection_mask": protection_mask,
            "raw_clothing_mask": raw_clothing_mask,
            "boxes": boxes
        }

    def generate_debug_visualization(self, image: np.ndarray, mask_results: Dict[str, Any]) -> np.ndarray:
        """
        Generates visual debug overlay frame:
         - Red overlay (255, 0, 0): Protected Regions (Face, Hair, Skin, Background)
         - Green overlay (0, 255, 0): Target Clothing Inpaint Region
         - Cyan overlay (0, 255, 255): Soft Edge Transition
        """
        vis_image = image.copy()

        protection = mask_results["protection_mask"] > 0
        target = mask_results["final_binary_mask"] > 0
        alpha = mask_results["final_alpha_mask"]

        # Red overlay for protected regions
        red_layer = np.zeros_like(vis_image)
        red_layer[:, :] = [255, 0, 0] # BGR Red: (0, 0, 255) if OpenCV, RGB (255, 0, 0)
        
        # Green overlay for target clothing
        green_layer = np.zeros_like(vis_image)
        green_layer[:, :] = [0, 255, 0]

        # Blend overlays onto original image
        vis_image[protection] = cv2.addWeighted(vis_image[protection], 0.6, red_layer[protection], 0.4, 0)
        vis_image[target] = cv2.addWeighted(vis_image[target], 0.5, green_layer[target], 0.5, 0)

        # Draw bounding boxes if present
        for (x1, y1, x2, y2) in mask_results.get("boxes", []):
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

        return vis_image

    def export_debug_mask(self, image: np.ndarray, mask_results: Dict[str, Any], output_path: Path) -> None:
        """Export visual debug mask image to file."""
        debug_vis = self.generate_debug_visualization(image, mask_results)
        cv2.imwrite(str(output_path), cv2.cvtColor(debug_vis, cv2.COLOR_RGB2BGR))
        logger.info(f"Exported debug mask visualization to {output_path}")
