from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "weights" / "yolo26_bienso_best.pt"

APP_TITLE = "YOLOv26 License Plate Recognition"
DEVICE = "0"
IMGSZ = 640
CONF = 0.25
TRACKER = "bytetrack.yaml"
HALF = False

VIDEO_MAX_W = 1800
VIDEO_MAX_H = 1200
CROP_MIN_H = 96

TOP_K_CROPS = 5
MIN_TRACK_HITS = 2
REALTIME_CROP_UPDATE_SEC = 0.45

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ARCHIVE_EXTS = {".zip", ".rar"}
AUTO_OCR_QUALITY = 0.48
AUTO_OCR_INTERVAL_SEC = 0.8
AUTO_OCR_QUALITY = 0.45
AUTO_OCR_INTERVAL_SEC = 0.5

RESULT_PANEL_W = 260
