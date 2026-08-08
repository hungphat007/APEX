"""
Phase 4 Unit Tests for Undress Video Pipeline.
Tests ArcFace identity ranking, Temporal optical flow propagation, master UndressVideoPipeline,
and Production Mode RuntimeError rules.

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
from undress_pipeline.identity.arcface_ranker import ArcFaceRanker
from undress_pipeline.temporal.propagator import TemporalPropagator
from undress_pipeline.pipeline.undress_pipeline import UndressVideoPipeline

class TestPhase4(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir=Path(self.tmp_dir.name))
        
        # Create 10 synthetic test frames (240x320 RGB)
        self.test_frames = []
        for i in range(10):
            f = np.zeros((240, 320, 3), dtype=np.uint8)
            f[:, :] = [150 + i, 150, 150]
            self.test_frames.append(f)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_arcface_ranker_fallback(self):
        """Test ArcFace identity embedding extraction and candidate ranking."""
        ranker = ArcFaceRanker(self.checkpoint_mgr, allow_fallback=True)
        ref_frame = self.test_frames[0]
        
        cand1 = ref_frame.copy()
        cand2 = ref_frame.copy()
        cand2[50:100, 50:100] = [200, 200, 200]

        best_idx, best_img, scores = ranker.rank_candidates(ref_frame, [cand1, cand2])
        self.assertIn(best_idx, [0, 1])
        self.assertEqual(len(scores), 2)
        self.assertEqual(best_img.shape, ref_frame.shape)

    def test_temporal_propagator(self):
        """Test optical flow warping and temporal frame interpolation across non-keyframes."""
        propagator = TemporalPropagator(self.checkpoint_mgr, allow_fallback=True)
        
        keyframe_indices = [0, 9]
        inpainted_keyframes = {
            0: self.test_frames[0].copy(),
            9: self.test_frames[9].copy()
        }
        alpha_masks = [np.zeros((240, 320), dtype=np.float32) for _ in range(10)]
        for m in alpha_masks:
            m[80:160, 100:200] = 1.0

        propagated_frames = propagator.propagate_clip(
            frames=self.test_frames,
            keyframe_indices=keyframe_indices,
            inpainted_keyframes=inpainted_keyframes,
            alpha_masks=alpha_masks
        )

        self.assertEqual(len(propagated_frames), 10)
        for f in propagated_frames:
            self.assertEqual(f.shape, (240, 320, 3))
            self.assertEqual(f.dtype, np.uint8)

    def test_undress_video_pipeline_end_to_end(self):
        """Test master UndressVideoPipeline end-to-end video synthesis."""
        pipeline = UndressVideoPipeline(self.checkpoint_mgr, allow_fallback=True)
        out_video = Path(self.tmp_dir.name) / "output_test.mp4"

        results = pipeline.process_frames(
            frames=self.test_frames,
            fps=24.0,
            output_video_path=out_video,
            keyframe_interval=4,
            candidate_count=2,
            debug_masks=True
        )

        self.assertTrue(out_video.exists())
        self.assertGreater(out_video.stat().st_size, 0)
        self.assertEqual(results["width"], 320)
        self.assertEqual(results["height"], 240)
        self.assertEqual(results["total_frames"], 10)

        # Check debug mask video file
        debug_video = Path(self.tmp_dir.name) / "output_test_debug_masks.mp4"
        self.assertTrue(debug_video.exists())

    def test_pipeline_production_mode_error(self):
        """Test master pipeline raises RuntimeError in Production Mode when weights missing."""
        with self.assertRaises(RuntimeError):
            UndressVideoPipeline(self.checkpoint_mgr, allow_fallback=False)

if __name__ == "__main__":
    unittest.main()
