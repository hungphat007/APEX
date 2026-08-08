"""
LoRA Loader interface for Wan 2.1-VACE pipeline.
Supports loading up to 2 LoRAs (Undress/Skin-Body LoRA & Detail LoRA) with dynamic scale factors.
Optional LoRAs do NOT block execution if missing.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class LoRAConfig:
    def __init__(self, path: Path, scale: float = 1.0, name: str = "custom_lora"):
        self.path = Path(path)
        self.scale = scale
        self.name = name

    def is_valid(self) -> bool:
        return self.path.is_file() and self.path.stat().st_size > 0

class LoRALoader:
    def __init__(self, max_loras: int = 2):
        self.max_loras = max_loras
        self.loaded_loras: List[LoRAConfig] = []

    def register_lora(self, path: Path, scale: float = 1.0, name: Optional[str] = None) -> bool:
        """Register a LoRA checkpoint for injection into diffusion runtime."""
        if len(self.loaded_loras) >= self.max_loras:
            raise ValueError(f"Maximum allowed active LoRAs is {self.max_loras}. Cannot register more.")
        
        lora_name = name or Path(path).stem
        config = LoRAConfig(path=Path(path), scale=scale, name=lora_name)
        self.loaded_loras.append(config)
        logger.info(f"Registered LoRA '{config.name}' (scale={scale}) at {path}")
        return True

    def apply_loras(self, pipeline: Any, allow_fallback: bool = False) -> Any:
        """
        Inject registered LoRAs into the diffusers pipeline model.
        Skips missing or optional LoRAs gracefully without crashing.
        """
        if not self.loaded_loras:
            logger.info("No active LoRAs registered. Running base model only.")
            return pipeline

        valid_configs = [c for c in self.loaded_loras if c.is_valid()]
        if not valid_configs:
            logger.warning("No valid LoRA checkpoint files found. Running base Wan model only.")
            return pipeline

        for config in valid_configs:
            logger.info(f"Applying LoRA '{config.name}' with scale {config.scale}...")
            if hasattr(pipeline, "load_lora_weights"):
                try:
                    pipeline.load_lora_weights(str(config.path), adapter_name=config.name)
                    if hasattr(pipeline, "set_adapters"):
                        pipeline.set_adapters([config.name], adapter_scales=[config.scale])
                except Exception as e:
                    logger.warning(f"Failed to inject LoRA {config.name} ({str(e)}). Continuing with base model only.")

        return pipeline
