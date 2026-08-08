"""
ArcFace Identity Ranking module using InsightFace embeddings.
Ranks candidate keyframe reconstructions against original identity embeddings.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2

from undress_pipeline.runtime.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

class ArcFaceRanker:
    """InsightFace ArcFace Identity Ranker."""

    def __init__(self, checkpoint_mgr: CheckpointManager, allow_fallback: bool = False):
        self.checkpoint_mgr = checkpoint_mgr
        self.allow_fallback = allow_fallback
        self.session = None
        self._init_model()

    def _init_model(self):
        try:
            ckpt_path = self.checkpoint_mgr.ensure_model("arcface", allow_fallback=self.allow_fallback)
            if ckpt_path.is_file() and ckpt_path.stat().st_size > 0:
                import onnxruntime as ort
                self.session = ort.InferenceSession(str(ckpt_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                logger.info(f"Loaded ArcFace ONNX model from {ckpt_path}.")
            else:
                if not self.allow_fallback:
                    raise RuntimeError(f"[PRODUCTION MODE ERROR] ArcFace model weights missing at {ckpt_path}.")
                logger.warning(f"ArcFace model missing at {ckpt_path}; using structural identity similarity for --allow-fallback mode.")
        except Exception as e:
            if not self.allow_fallback:
                raise e
            logger.warning(f"ArcFace initialization issue ({str(e)}); running in --allow-fallback mode.")

    def extract_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Extract 512-D identity embedding vector from cropped face image.
        """
        if self.session is not None:
            try:
                img_resized = cv2.resize(face_crop, (112, 112))
                img_normalized = (img_resized.astype(np.float32) - 127.5) / 127.5
                input_blob = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]

                input_name = self.session.get_inputs()[0].name
                embedding = self.session.run(None, {input_name: input_blob})[0][0]
                norm = np.linalg.norm(embedding)
                return embedding / (norm + 1e-10)
            except Exception as e:
                logger.error(f"ArcFace embedding extraction failed: {str(e)}")
                if not self.allow_fallback:
                    raise e

        # Synthetic fallback embedding vector (512-D float32 normalized vector based on image color statistics)
        mean_val = np.mean(face_crop, axis=(0, 1)) / 255.0
        vec = np.zeros(512, dtype=np.float32)
        vec[:3] = mean_val
        vec[3:10] = np.std(face_crop) / 255.0
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)

    def rank_candidates(
        self,
        reference_frame: np.ndarray,
        candidates: List[np.ndarray],
        face_box: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[int, np.ndarray, List[float]]:
        """
        Rank N candidate reconstructions against original reference frame identity.
        
        Returns:
            Tuple of:
             - Best candidate index (0 to N-1)
             - Best candidate RGB image
             - List of similarity scores for all candidates
        """
        if not candidates:
            raise ValueError("Candidates list cannot be empty.")

        if len(candidates) == 1:
            return 0, candidates[0], [1.0]

        h, w = reference_frame.shape[:2]
        if face_box is None:
            # Crop upper center region for face embedding
            fx1, fy1, fx2, fy2 = int(w * 0.3), int(h * 0.1), int(w * 0.7), int(h * 0.4)
        else:
            fx1, fy1, fx2, fy2 = face_box

        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(w, fx2), min(h, fy2)

        ref_face = reference_frame[fy1:fy2, fx1:fx2]
        ref_embed = self.extract_embedding(ref_face)

        scores = []
        for cand in candidates:
            cand_face = cand[fy1:fy2, fx1:fx2]
            cand_embed = self.extract_embedding(cand_face)
            # Cosine similarity score
            sim = float(np.dot(ref_embed, cand_embed))
            scores.append(sim)

        best_idx = int(np.argmax(scores))
        logger.info(f"ArcFace Ranked {len(candidates)} candidates. Scores: {[round(s, 4) for s in scores]}. Best index: {best_idx}")
        return best_idx, candidates[best_idx], scores
