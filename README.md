# Undress Video Pipeline

A dedicated, production-grade **Undress-Only Video Reconstruction Pipeline** built for **Google Colab T4 (16GB VRAM)** and **RTX 3050 (8GB VRAM)**.

Reconstructs target clothing regions as natural body skin while preserving **100%** of face identity, hair, hands, arms/legs skin, feet, background, pose, motion, and lighting.

---

## Model Registry & Auto-Download Policy

| Model Key | Description | File | Size |
| :--- | :--- | :--- | :--- |
| `yolo11n` | Person Bounding Box Detector | `yolo11n.pt` | ~5.6 MB |
| `schp_atr` | Self-Correction Human Parsing (ATR-18) | `schp-atr.pth` | ~204 MB |
| `sam2_small` | SAM2 Small Edge Refinement | `sam2_hiera_small.pt` | ~184 MB |
| `birefnet` | BiRefNet High-Resolution Edge Refinement | `BiRefNet.safetensors` | ~978 MB |
| `wan_2.1_vace_1.3b` | Wan 2.1-VACE-1.3B Masked Inpainting | `wan2.1_vace_1.3b_inpainting.safetensors` | ~2.8 GB |
| `arcface` | InsightFace ArcFace Identity Model (buffalo_l) | `buffalo_l` (Auto) | ~250 MB |
| `rife` | RIFE Optical Flow Motion Propagator | `flownet_rife.pth` | ~70 MB |
| `skin_body_lora` | Skin / Body Detail LoRA | `wan_skin_body_v1.safetensors` | ~150 MB |

> [!IMPORTANT]
> **STRICT LOCAL RULE**: Zero model weights are downloaded automatically during local machine development. In Production Mode, missing weights throw a clean `RuntimeError`. Pass `--download-weights` to explicitly trigger downloading weights locally, or `--allow-fallback` for local CPU demo testing.
> On **Google Colab**, missing models auto-download automatically upon running.

---

## One-Command Google Colab Setup

Run the following in Google Colab (T4 GPU Runtime):

```bash
!git clone https://github.com/your-org/undress_pipeline.git
%cd undress_pipeline
!pip install -r requirements.txt
!python run.py --input video.mp4 --output result.mp4 --keyframe-interval 5 --candidates 2 --debug-masks
```

---

## Local Development & Testing

### 1. Run Unit Tests (CPU / Local)
```bash
python -m unittest discover tests
```

### 2. Run Local Demo Mode (`--allow-fallback`)
```bash
python run.py --input sample.mp4 --output result.mp4 --allow-fallback --debug-masks
```

### 3. Explicit Local Model Download
```bash
python run.py --input sample.mp4 --output result.mp4 --download-weights
```

---

## CLI Options

```
options:
  -i, --input INPUT          Path to input clothed video file (.mp4)
  -o, --output OUTPUT        Path to save reconstructed output video (.mp4)
  --keyframe-interval N     Keyframe interval for Wan 2.1 inpainting (default: 5)
  --candidates N            ArcFace candidate variations to rank per keyframe (default: 2)
  --checkpoint-dir PATH     Custom directory to store/load model weights
  --download-weights        Allow downloading missing model weights on local machine
  --allow-fallback          Enable synthetic demo mode on CPU if weights are missing
  --debug-masks             Export visual mask debug video (Red=Protected, Green=Target)
  --lora-path PATH          Custom skin/body LoRA checkpoint path
  --lora-scale FACTOR       LoRA influence scale (0.0 to 1.0, default: 0.8)
  -v, --verbose             Enable detailed debug logging
```

---

## Project Structure

```
undress_pipeline/
├── mask/
│   ├── person_detector.py      # YOLO11n person bounding box detector
│   ├── clothing_segmentor.py   # SCHP ATR-18 clothing parser
│   ├── edge_refiner.py         # SAM2 + BiRefNet soft edge refiner
│   ├── region_lock.py          # Protection mask (face, hair, hands, skin, bg)
│   └── mask_composer.py        # Master mask composer & debug visualization
├── runtime/
│   ├── environment.py          # Colab / CUDA / CPU environment detector
│   ├── checkpoint_manager.py   # Model weight registry & auto-download manager
│   ├── wan_runtime.py          # Wan 2.1-VACE-1.3B masked inpainting runtime
│   └── lora_loader.py          # Skin/body LoRA injection manager
├── identity/
│   └── arcface_ranker.py       # InsightFace ArcFace identity candidate ranker
├── temporal/
│   └── propagator.py           # RIFE / Optical Flow temporal propagation
├── pipeline/
│   └── undress_pipeline.py     # Master video processing controller
├── tests/                      # Unit test suite across all phases
├── Colab_Undress_Video_Pipeline.ipynb # One-click Google Colab notebook
├── run.py                      # Main CLI entrypoint
├── requirements.txt            # Package dependencies
├── .gitignore                  # Enforces zero model weights in Git
└── README.md
```
