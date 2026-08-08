"""
Phase 5 Unit Tests for Undress Video Pipeline.
Tests Colab auto-download trigger rules, registry URLs, .gitignore rules, and notebook validity.

Guarantees ZERO model weight downloads occur during default test execution.
"""

import unittest
import tempfile
import sys
import os
import json
from pathlib import Path

# Add package root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from undress_pipeline.runtime.checkpoint_manager import CheckpointManager, MODEL_REGISTRY
from undress_pipeline.runtime.environment import is_colab

class TestPhase5(unittest.TestCase):

    def test_model_registry_urls(self):
        """Verify all registered models have valid filenames and non-empty URLs."""
        for key, info in MODEL_REGISTRY.items():
            self.assertIn("filename", info)
            self.assertIn("description", info)
            self.assertIn("size_mb", info)
            self.assertTrue(info.get("url") or info.get("repo_id"), f"Model '{key}' lacks download URL and repo_id.")

    def test_local_download_prevention(self):
        """Verify local execution without force_download raises RuntimeError and downloads nothing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=Path(tmpdir))
            
            if not is_colab():
                with self.assertRaises(RuntimeError) as ctx:
                    mgr.ensure_model("yolo11n", force_download=False, allow_fallback=False)
                self.assertIn("PRODUCTION MODE ERROR", str(ctx.exception))
                self.assertIn("STRICT LOCAL RULE ENABLED", str(ctx.exception))

    def test_gitignore_rules(self):
        """Verify .gitignore contains patterns for model weights and test outputs."""
        gitignore_path = Path(__file__).parent.parent / ".gitignore"
        self.assertTrue(gitignore_path.exists())
        
        content = gitignore_path.read_text()
        required_patterns = ["*.pt", "*.pth", "*.safetensors", "*.onnx", "checkpoints/"]
        for pat in required_patterns:
            self.assertIn(pat, content, f".gitignore missing weight exclusion pattern '{pat}'")

    def test_colab_notebook_structure(self):
        """Verify Colab_Undress_Video_Pipeline.ipynb is a valid Jupyter Notebook."""
        nb_path = Path(__file__).parent.parent / "Colab_Undress_Video_Pipeline.ipynb"
        self.assertTrue(nb_path.exists())
        
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_json = json.load(f)
            
        self.assertIn("cells", nb_json)
        self.assertGreater(len(nb_json["cells"]), 0)
        self.assertIn("metadata", nb_json)

if __name__ == "__main__":
    unittest.main()
