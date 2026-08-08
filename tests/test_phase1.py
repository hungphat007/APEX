"""
Phase 1 Unit Tests for Undress Video Pipeline.
Tests environment detection, CheckpointManager rules, and LoRALoader.
Guarantees NO model weights are downloaded during local test execution.
"""

import unittest
from pathlib import Path
import tempfile
import sys
import os

# Add package root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from undress_pipeline.runtime.environment import is_colab, get_device_info
from undress_pipeline.runtime.checkpoint_manager import CheckpointManager, MODEL_REGISTRY
from undress_pipeline.runtime.lora_loader import LoRALoader, LoRAConfig

class TestPhase1(unittest.TestCase):

    def test_environment_detection(self):
        """Test environment detection returns valid structure."""
        info = get_device_info()
        self.assertIn("is_colab", info)
        self.assertIn("device", info)
        self.assertIn("precision", info)
        self.assertIn(info["device"], ["cuda", "mps", "cpu"])
        self.assertIn(info["precision"], ["fp16", "fp32"])

    def test_checkpoint_registry(self):
        """Verify model registry contains required keys."""
        required_keys = ["yolo11n", "schp_atr", "sam2_small", "birefnet", "wan_2.1_vace_1.3b", "arcface", "rife", "skin_body_lora"]
        for key in required_keys:
            self.assertIn(key, MODEL_REGISTRY)

    def test_checkpoint_manager_production_mode_error(self):
        """
        Verify CheckpointManager raises RuntimeError on missing weights in local production mode.
        Guarantees NO download occurs.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=Path(tmpdir))
            
            # Model should be missing in empty tmpdir
            self.assertFalse(mgr.is_model_available("yolo11n"))
            
            # Calling ensure_model without force_download or allow_fallback must raise RuntimeError on local
            if not is_colab():
                with self.assertRaises(RuntimeError) as ctx:
                    mgr.ensure_model("yolo11n", force_download=False, allow_fallback=False)
                self.assertIn("PRODUCTION MODE ERROR", str(ctx.exception))
                self.assertIn("yolo11n", str(ctx.exception))

    def test_checkpoint_manager_allow_fallback(self):
        """Verify allow_fallback returns path without downloading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=Path(tmpdir))
            path = mgr.ensure_model("yolo11n", force_download=False, allow_fallback=True)
            self.assertEqual(path, mgr.get_checkpoint_path("yolo11n"))
            # File should still NOT exist because no download took place
            self.assertFalse(path.exists())

    def test_lora_loader(self):
        """Test LoRA loader registration and limit checks."""
        loader = LoRALoader(max_loras=2)
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(b"dummy_bytes")

        try:
            self.assertTrue(loader.register_lora(tmp_path, scale=0.8, name="test_lora_1"))
            self.assertTrue(loader.register_lora(tmp_path, scale=0.5, name="test_lora_2"))
            
            # Exceeding max_loras should raise ValueError
            with self.assertRaises(ValueError):
                loader.register_lora(tmp_path, scale=1.0, name="test_lora_3")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

if __name__ == "__main__":
    unittest.main()
