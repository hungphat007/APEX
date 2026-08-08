"""
Person Detector module using YOLO11n.
Detects person bounding boxes (COCO class 0) in video frames.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from undress_pipeline.runtime.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

class PersonDetector:
    """YOLO11n-based Person Detector."""

    def __init__(self, checkpoint_mgr: CheckpointManager, allow_fallback: bool = False):
        self.checkpoint_mgr = checkpoint_mgr
        self.allow_fallback = allow_fallback
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            ckpt_path = self.checkpoint_mgr.ensure_model("yolo11n", allow_fallback=self.allow_fallback)
            if ckpt_path.is_file():
                from ultralytics import YOLO
                self.model = YOLO(str(ckpt_path))
                logger.info(f"Loaded YOLO11n model from {ckpt_path}")
            else:
                if not self.allow_fallback:
                    raise RuntimeError("YOLO11n checkpoint file missing.")
                logger.warning("YOLO11n weights missing; running in --allow-fallback synthetic mode.")
        except Exception as e:
            if not self.allow_fallback:
                raise e
            logger.warning(f"Failed to load YOLO11n model ({str(e)}); using synthetic fallback detector.")

    def detect(self, image: np.ndarray, conf_threshold: float = 0.4) -> List[Tuple[int, int, int, int]]:
        """
        Detect person bounding boxes in an RGB/BGR image.
        Returns list of (x1, y1, x2, y2) bounding boxes.
        """
        h, w = image.shape[:2]

        if self.model is not None:
            results = self.model(image, classes=[0], conf=conf_threshold, verbose=False)
            boxes = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = map(int, box[:4])
                        boxes.append((x1, y1, x2, y2))
            if boxes:
                return boxes

        # Synthetic fallback box if no model loaded or no person detected in demo mode
        if self.allow_fallback:
            # Generate centered person bounding box (e.g. middle 70% of image)
            x1, y1 = int(w * 0.2), int(h * 0.1)
            x2, y2 = int(w * 0.8), int(h * 0.95)
            return [(x1, y1, x2, y2)]

        return []
