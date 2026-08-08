"""
Region Lock module for protecting face, hair, hands, exposed skin, limbs, and background.
Guarantees 100% preservation of non-clothing pixels.
"""

import logging
from typing import Set, Optional
import numpy as np
import cv2

from undress_pipeline.mask.clothing_segmentor import PROTECTED_CATEGORIES

logger = logging.getLogger(__name__)

class RegionLock:
    """Extracts and dilates safety region-lock masks for protected anatomical regions."""

    def __init__(self, protected_categories: Optional[Set[int]] = None, safety_margin_px: int = 5):
        self.protected_categories = protected_categories if protected_categories is not None else PROTECTED_CATEGORIES
        self.safety_margin_px = safety_margin_px

    def generate_protection_mask(self, parse_map: np.ndarray) -> np.ndarray:
        """
        Extract binary protection mask from SCHP category parse map.
        
        Pixels valued 255 in returned mask MUST be protected (locked).
        Pixels valued 0 in returned mask are eligible for clothing inpainting.
        """
        # Select pixels matching protected category set
        raw_lock_mask = np.isin(parse_map, list(self.protected_categories)).astype(np.uint8) * 255

        if self.safety_margin_px > 0:
            # Dilate protected mask slightly to create safety buffer around skin, face, background
            kernel_size = self.safety_margin_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated_lock_mask = cv2.dilate(raw_lock_mask, kernel, iterations=1)
            return dilated_lock_mask

        return raw_lock_mask
