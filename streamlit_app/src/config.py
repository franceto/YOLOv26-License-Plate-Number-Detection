import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(r"D:\Documents\KimTin\BienSo Detection")
MODEL_PATH = ROOT / "weights" / "yolo26_biensao_best.pt"

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]