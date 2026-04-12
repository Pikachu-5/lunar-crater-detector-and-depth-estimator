"""
YOLO11s Crater Detection Training Script — Intel Arc GPU (XPU)
==============================================================

This script trains YOLO11s on the Impact Moon Craters (LU3M6TGT) dataset
using an Intel Arc GPU via native PyTorch XPU support.

Usage:
    .venv_train\Scripts\python train_crater_yolo.py

The trained model will be saved to:
    runs/detect/crater_detector/weights/best.pt

After training, copy best.pt to the project root to replace yolov8n.pt,
or update detector.py to point to the new weights.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# 1. Detect Intel XPU availability
# ---------------------------------------------------------------------------
def detect_device() -> torch.device:
    """Detect the best available training device.

    Returns a torch.device object (not a string) so that Ultralytics'
    select_device() accepts it directly without CUDA validation.

    Priority: Intel XPU (Arc) → CUDA → CPU
    """
    # 1. Try native XPU (PyTorch 2.11+ bundles XPU support)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        dev_name = torch.xpu.get_device_name(0)
        try:
            vram_mb = torch.xpu.get_device_properties(0).total_memory // (1024 * 1024)
            print(f"[INFO] Intel XPU detected: {dev_name} ({vram_mb} MB)")
        except Exception:
            print(f"[INFO] Intel XPU detected: {dev_name}")
        return torch.device("xpu")

    # 2. Try CUDA
    if torch.cuda.is_available():
        print(f"[INFO] CUDA detected: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda:0")

    print("[INFO] No GPU detected. Training will use CPU (will be very slow).")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 2. Training configuration — tuned for max accuracy on 416x416 crater images
# ---------------------------------------------------------------------------
DATA_YAML    = str(Path(__file__).parent / "LU3M6TGT_yolo_format" / "data.yaml")
BASE_MODEL   = "yolo11s.pt"           # Start from COCO-pretrained YOLO11s (9.4M params, 47% mAP)
PROJECT_NAME = "runs/detect"
RUN_NAME     = "crater_detector"

# Training hyperparameters — maximize accuracy within 16 GB RAM
TRAIN_CONFIG = dict(
    data       = DATA_YAML,
    epochs     = 150,                  # More epochs for convergence
    imgsz      = 416,                  # Match dataset native resolution
    batch      = 16,                   # Safe for 16 GB RAM + Arc GPU
    patience   = 30,                   # Early stopping if no improvement for 30 epochs
    optimizer  = "AdamW",              # Better convergence than SGD for smaller datasets
    lr0        = 0.001,                # Initial learning rate
    lrf        = 0.01,                 # Final LR as fraction of lr0
    warmup_epochs = 5,                 # Warm-up for stable training start
    weight_decay  = 0.0005,
    cos_lr     = True,                 # Cosine LR schedule for smoother decay

    # Augmentation — aggressive for max accuracy
    hsv_h      = 0.015,                # Hue shift (minimal for grayscale-like images)
    hsv_s      = 0.2,                  # Saturation shift
    hsv_v      = 0.4,                  # Value/brightness shift (important for shadow variation)
    degrees    = 15.0,                 # Rotation augmentation
    translate  = 0.15,                 # Translation augmentation
    scale      = 0.5,                  # Scale augmentation range
    flipud     = 0.5,                  # Vertical flip (craters are rotation-invariant)
    fliplr     = 0.5,                  # Horizontal flip
    mosaic     = 1.0,                  # Mosaic augmentation probability
    mixup      = 0.1,                  # Mixup augmentation

    # Output
    project    = PROJECT_NAME,
    name       = RUN_NAME,
    exist_ok   = True,                 # Overwrite previous run
    save       = True,
    save_period = 25,                  # Checkpoint every 25 epochs
    plots      = True,                 # Generate training plots

    # Workers — keep low for 16 GB RAM
    workers    = 4,

    # Validation
    val        = True,
)


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run YOLO11s training pipeline."""

    # Validate dataset exists
    if not os.path.isfile(DATA_YAML):
        print(f"[ERROR] Dataset config not found: {DATA_YAML}")
        sys.exit(1)

    device = detect_device()
    is_xpu = device.type == "xpu"

    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"  CRATER DETECTION TRAINING")
    print(f"  Model:   {BASE_MODEL}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  Device:  {device}")
    print(f"  Epochs:  {TRAIN_CONFIG['epochs']}")
    print(f"  ImgSize: {TRAIN_CONFIG['imgsz']}")
    print(f"  Batch:   {TRAIN_CONFIG['batch']}")
    print(f"{'='*60}\n")

    model = YOLO(BASE_MODEL)

    # For XPU: disable AMP to avoid CUDA-specific AMP code paths
    if is_xpu:
        TRAIN_CONFIG["amp"] = False

    results = model.train(device=device, **TRAIN_CONFIG)

    # Print final metrics
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best model: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print(f"{'='*60}\n")

    # Run validation on best weights
    print("[INFO] Running final validation on best weights...")
    best_model = YOLO(f"{PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    metrics = best_model.val(data=DATA_YAML, device=device)

    print(f"\n  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:     {metrics.box.mp:.4f}")
    print(f"  Recall:        {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
