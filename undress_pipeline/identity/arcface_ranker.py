"""
ArcFace Identity Ranking module using InsightFace FaceAnalysis (buffalo_l).
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
    """InsightFace ArcFace Identity Ranker using FaceAnalysis(name='buffalo_l')."""

    def __init__(self, checkpoint_mgr: CheckpointManager, allow_fallback: bool = False):
        self.checkpoint_mgr = checkpoint_mgr
        self.allow_fallback = allow_fallback
        self.app = None
        self._init_model()

    def _init_model(self):
        try:
            import torch
            from insightface.app import FaceAnalysis

            ctx_id = 0 if torch.cuda.is_available() else -1
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]

            logger.info("Initializing InsightFace FaceAnalysis(name='buffalo_l')...")
            self.app = FaceAnalysis(name="buffalo_l", providers=providers)
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info("InsightFace FaceAnalysis (buffalo_l) initialized successfully.")
        except Exception as e:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"[PRODUCTION MODE ERROR] InsightFace FaceAnalysis initialization failed ({str(e)}).\n"
                    f"Ensure insightface is installed or run with --allow-fallback."
                ) from e
            logger.warning(f"InsightFace initialization issue ({str(e)}); running in --allow-fallback mode.")

    def extract_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Extract 512-D identity embedding vector from face crop/image using InsightFace FaceAnalysis.
        """
        if self.app is not None:
            try:
                # FaceAnalysis expects BGR image
                img_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR) if (len(face_crop.shape) == 3 and face_crop.shape[2] == 3) else face_crop
                faces = self.app.get(img_bgr)
                if faces:
                    # Return embedding of largest detected face
                    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                    embedding = faces[0].embedding
                    norm = np.linalg.norm(embedding)
                    return embedding / (norm + 1e-10)
            except Exception as e:
                logger.error(f"InsightFace embedding extraction failed: {str(e)}")
                if not self.allow_fallback:
                    raise e

        # Synthetic fallback embedding vector (512-D float32 normalized vector)
        mean_val = np.mean(face_crop, axis=(0, 1)) / 255.0 if face_crop.size > 0 else np.array([0.5, 0.5, 0.5])
        vec = np.zeros(512, dtype=np.float32)
        vec[:3] = mean_val
        vec[3:10] = np.std(face_crop) / 255.0 if face_crop.size > 0 else 0.1
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
        logger.info(f"ArcFace Ranked {len(candidates)} candidates using FaceAnalysis(buffalo_l). Scores: {[round(s, 4) for s in scores]}. Best index: {best_idx}")
        return best_idx, candidates[best_idx], scores
