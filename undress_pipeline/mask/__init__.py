"""
Mask subpackage for human detection, SCHP clothing parsing, soft edge refinement, and region locking.
"""

from .person_detector import PersonDetector
from .clothing_segmentor import ClothingSegmentor, CLOTHING_CATEGORIES, PROTECTED_CATEGORIES
from .edge_refiner import EdgeRefiner
from .region_lock import RegionLock
from .mask_composer import MaskComposer

__all__ = [
    "PersonDetector",
    "ClothingSegmentor",
    "CLOTHING_CATEGORIES",
    "PROTECTED_CATEGORIES",
    "EdgeRefiner",
    "RegionLock",
    "MaskComposer"
]
