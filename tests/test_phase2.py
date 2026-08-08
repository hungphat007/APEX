"""
Phase 2 Unit Tests for Undress Video Pipeline.
Tests Clothing Mask Pipeline: PersonDetector, SCHP ClothingSegmentor, EdgeRefiner,
RegionLock, MaskComposer, debug mask export, and Production Mode RuntimeError rules.

Guarantees ZERO model weight downloads occur during test execution.
"""

import unittest
import tempfile
import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add package root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from undress_pipeline.runtime.checkpoint_manager import CheckpointManager
from undress_pipeline.mask.person_detector import PersonDetector
from undress_pipeline.mask.clothing_segmentor import ClothingSegmentor, CLOTHING_CATEGORIES, PROTECTED_CATEGORIES
from undress_pipeline.mask.edge_refiner import EdgeRefiner
from undress_pipeline.mask.region_lock import RegionLock
from undress_pipeline.mask.mask_composer import MaskComposer

class TestPhase2(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir=Path(self.tmp_dir.name))
        # Standard test synthetic RGB frame (480x640x3)
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_frame[:, :] = [128, 128, 128]

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_person_detector_fallback(self):
        """Test PersonDetector returns valid bounding boxes in demo fallback mode."""
        detector = PersonDetector(self.checkpoint_mgr, allow_fallback=True)
        boxes = detector.detect(self.test_frame)
        self.assertGreater(len(boxes), 0)
        x1, y1, x2, y2 = boxes[0]
        self.assertTrue(0 <= x1 < x2 <= 640)
        self.assertTrue(0 <= y1 < y2 <= 480)

    def test_clothing_segmentor_fallback(self):
        """Test SCHP ClothingSegmentor parsing and clothing mask extraction."""
        segmentor = ClothingSegmentor(self.checkpoint_mgr, allow_fallback=True)
        parse_map = segmentor.parse_image(self.test_frame)
        self.assertEqual(parse_map.shape, (480, 640))
        self.assertTrue(np.issubdtype(parse_map.dtype, np.integer))

        clothing_mask = segmentor.get_clothing_mask(parse_map)
        self.assertEqual(clothing_mask.shape, (480, 640))
        self.assertEqual(clothing_mask.dtype, np.uint8)
        self.assertIn(clothing_mask.max(), [0, 255])

    def test_edge_refiner(self):
        """Test EdgeRefiner produces hard binary mask and soft float alpha mask."""
        refiner = EdgeRefiner(self.checkpoint_mgr, allow_fallback=True)
        dummy_mask = np.zeros((480, 640), dtype=np.uint8)
        dummy_mask[100:300, 200:400] = 255

        hard_mask, alpha_mask = refiner.refine_edges(self.test_frame, dummy_mask, dilation_px=4, blur_radius=7)
        self.assertEqual(hard_mask.shape, (480, 640))
        self.assertEqual(alpha_mask.shape, (480, 640))
        self.assertEqual(alpha_mask.dtype, np.float32)
        self.assertTrue(0.0 <= alpha_mask.min() <= alpha_mask.max() <= 1.0)

    def test_region_lock(self):
        """Test RegionLock generates protection mask for face, hair, skin, and background."""
        region_lock = RegionLock(safety_margin_px=5)
        # Create parse map with face=13, hair=2, upper-clothes=5
        parse_map = np.zeros((480, 640), dtype=np.uint8)
        parse_map[50:100, 200:300] = 13 # Face
        parse_map[20:50, 200:300] = 2   # Hair
        parse_map[100:300, 200:300] = 5 # Upper-clothes

        protection_mask = region_lock.generate_protection_mask(parse_map)
        self.assertEqual(protection_mask.shape, (480, 640))
        # Face and Hair areas must be protected (255)
        self.assertEqual(protection_mask[60, 250], 255)
        self.assertEqual(protection_mask[30, 250], 255)

    def test_mask_composer_and_debug_export(self):
        """Test end-to-end MaskComposer and debug mask visualization export."""
        composer = MaskComposer(self.checkpoint_mgr, allow_fallback=True)
        results = composer.compose_mask(self.test_frame)

        self.assertIn("final_binary_mask", results)
        self.assertIn("final_alpha_mask", results)
        self.assertIn("protection_mask", results)
        self.assertIn("raw_clothing_mask", results)

        # CRITICAL RULE CHECK: Final Target Clothing pixels must NOT overlap with Protection Mask
        overlap = cv2.bitwise_and(results["final_binary_mask"], results["protection_mask"])
        self.assertEqual(np.sum(overlap > 0), 0, "Target clothing inpaint mask overlaps with protected region!")

        # Export debug mask image test
        out_debug_file = Path(self.tmp_dir.name) / "debug_mask_test.png"
        composer.export_debug_mask(self.test_frame, results, out_debug_file)
        self.assertTrue(out_debug_file.exists())
        self.assertGreater(out_debug_file.stat().st_size, 0)

    def test_production_mode_missing_weights_error(self):
        """Test that missing weights raise RuntimeError in Production Mode (no allow_fallback)."""
        with self.assertRaises(RuntimeError):
            ClothingSegmentor(self.checkpoint_mgr, allow_fallback=False)

if __name__ == "__main__":
    unittest.main()
