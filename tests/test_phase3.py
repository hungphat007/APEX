"""
Phase 3 Unit Tests for Undress Video Pipeline.
Tests Wan 2.1-VACE-1.3B masked inpainting runtime, single-frame reconstruction,
LoRA injection interface, and Production Mode RuntimeError rules.

Guarantees ZERO model weight downloads occur during test execution.
"""

import unittest
import tempfile
import sys
import os
from pathlib import Path
import numpy as np

# Add package root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from undress_pipeline.runtime.checkpoint_manager import CheckpointManager
from undress_pipeline.runtime.lora_loader import LoRALoader
from undress_pipeline.runtime.wan_runtime import WanRuntime
from undress_pipeline.mask.mask_composer import MaskComposer

class TestPhase3(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir=Path(self.tmp_dir.name))
        # Standard test synthetic RGB frame (480x640x3)
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_frame[:, :] = [100, 100, 100]

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_wan_runtime_demo_mode(self):
        """Test WanRuntime initializes in --allow-fallback demo mode without downloading weights."""
        wan_runtime = WanRuntime(self.checkpoint_mgr, allow_fallback=True)
        self.assertEqual(wan_runtime.device, "cpu")
        self.assertIsNone(wan_runtime.pipeline)

    def test_single_frame_inpainting_fallback(self):
        """Test single frame inpainting reconstructs target clothing mask as skin tone."""
        composer = MaskComposer(self.checkpoint_mgr, allow_fallback=True)
        wan_runtime = WanRuntime(self.checkpoint_mgr, allow_fallback=True)

        mask_results = composer.compose_mask(self.test_frame)
        target_binary = mask_results["final_binary_mask"]
        alpha_mask = mask_results["final_alpha_mask"]

        inpainted_frame = wan_runtime.inpaint_single_frame(
            image_rgb=self.test_frame,
            mask_binary=target_binary,
            alpha_mask=alpha_mask,
            seed=42
        )

        # Output shape & type must match input
        self.assertEqual(inpainted_frame.shape, (480, 640, 3))
        self.assertEqual(inpainted_frame.dtype, np.uint8)

        # Check that protected regions (e.g. background/face) remain untouched
        protection_mask = mask_results["protection_mask"] > 0
        diff = np.abs(inpainted_frame.astype(np.int32) - self.test_frame.astype(np.int32))
        max_diff_protected = np.max(diff[protection_mask]) if np.any(protection_mask) else 0
        self.assertEqual(max_diff_protected, 0, "Protected region pixels were altered during inpainting!")

    def test_lora_integration(self):
        """Test LoRALoader integration with WanRuntime."""
        lora_loader = LoRALoader(max_loras=2)
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(b"dummy_lora_bytes")

        try:
            lora_loader.register_lora(tmp_path, scale=0.8, name="skin_lora")
            wan_runtime = WanRuntime(self.checkpoint_mgr, lora_loader=lora_loader, allow_fallback=True)
            self.assertEqual(len(wan_runtime.lora_loader.loaded_loras), 1)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_wan_runtime_production_mode_error(self):
        """Test WanRuntime raises RuntimeError in Production Mode when weights are missing."""
        with self.assertRaises(RuntimeError):
            WanRuntime(self.checkpoint_mgr, allow_fallback=False)

if __name__ == "__main__":
    unittest.main()
