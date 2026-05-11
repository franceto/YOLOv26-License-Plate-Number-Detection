import sys, base64, uuid
from pathlib import Path
from functools import lru_cache
from typing import Dict, List

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from streamlit_app.src.ocr_engine import recognize_plate

WEIGHT_DIR = ROOT / "weights"
MODEL_PATH = next(WEIGHT_DIR.glob("*best.pt"), None)

app = FastAPI(title="YOLOv26 License Plate FastAPI")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

TOPS: Dict[str, List[dict]] = {}

class FrameReq(BaseModel):
    image: str
    conf: float = 0.35
    imgsz: int = 512
    session_id: str = "default"
    max_top: int = 3

class OcrReq(BaseModel):
    session_id: str = "default"

@lru_cache(maxsize=1)
def load_model():
    from ultralytics import YOLO
    if MODEL_PATH is None:
        raise FileNotFoundError(f"Không tìm thấy file *best.pt trong {WEIGHT_DIR}")
    return YOLO(str(MODEL_PATH))

@lru_cache(maxsize=1)
def load_ocr():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()

def decode_image(data_url):
    data = data_url.split(",", 1)[1] if "," in data_url else data_url
    arr = np.frombuffer(base64.b64decode(data), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def encode_jpg(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf).decode() if ok else ""

def safe_crop(img, x1, y1, x2, y2, pad_ratio=0.08):
    h, w = img.shape[:2]
    pad = max(4, int(pad_ratio * max(x2 - x1, y2 - y1)))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)

def crop_quality(item):
    crop = item["crop"]
    if crop is None or crop.size == 0:
        return 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    bright = gray.mean()
    h, w = crop.shape[:2]
    area = h * w
    bright_score = 100 - min(abs(bright - 125), 100)
    return item["score"] * 100000 + min(sharp, 2500) * 12 + min(area, 60000) * 0.01 + bright_score * 20
def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter / (area_a + area_b - inter + 1e-6)

def box_center_dist(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2

    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
def update_top(session_id, items, max_top):
    old = TOPS.get(session_id, [])

    for item in items:
        matched = -1

        for i, old_item in enumerate(old):
            iou = box_iou(item["box"], old_item["box"])
            dist = box_center_dist(item["box"], old_item["box"])

            if iou > 0.35 or dist < 80:
                matched = i
                break

        if matched >= 0:
            if crop_quality(item) > crop_quality(old[matched]):
                old[matched] = item
        else:
            old.append(item)

    old = sorted(old, key=crop_quality, reverse=True)
    TOPS[session_id] = old[:max_top]
    old = TOPS.get(session_id, [])
    all_items = old + items
    all_items = sorted(all_items, key=crop_quality, reverse=True)
    TOPS[session_id] = all_items[:max_top]
def valid_plate_box(img, x1, y1, x2, y2, score):
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1

    if bw <= 0 or bh <= 0:
        return False

    area = bw * bh
    ratio = bw / max(1, bh)
    img_area = h * w

    if score < 0.35:
        return False
    if area < img_area * 0.00015:
        return False
    if area > img_area * 0.08:
        return False
    if ratio < 0.35 or ratio > 6.5:
        return False

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if gray.mean() < 12:
        return False
    if gray.std() < 8:
        return False

    return True
@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

@app.post("/reset")
def reset(req: OcrReq):
    TOPS.pop(req.session_id, None)
    return {"ok": True}

@app.post("/detect_frame")
def detect_frame(req: FrameReq):
    img = decode_image(req.image)
    if img is None:
        return {"boxes": [], "frame_w": 0, "frame_h": 0, "error": "decode failed"}

    model = load_model()
    res = model(img, conf=req.conf, iou=0.45, imgsz=req.imgsz, verbose=False)[0]

    boxes = []
    crops = []
    H, W = img.shape[:2]

    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        score = float(b.conf[0])

        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)

        if not valid_plate_box(img, x1, y1, x2, y2, score):
            continue

        raw = img[y1:y2, x1:x2]
        crop, box = safe_crop(img, x1, y1, x2, y2, 0.08)

        bx1, by1, bx2, by2 = box

        boxes.append({
            "x1": bx1,
            "y1": by1,
            "x2": bx2,
            "y2": by2,
            "score": round(score, 4),
            "label": "Bien-so"
        })

        crops.append({
            "crop": crop,
            "raw": raw,
            "score": score,
            "box": [bx1, by1, bx2, by2]
        })

    update_top(req.session_id, crops, req.max_top)

    return {
        "boxes": boxes,
        "frame_w": W,
        "frame_h": H,
        "top_count": len(TOPS.get(req.session_id, []))
    }

@app.post("/ocr_top")
def ocr_top(req: OcrReq):
    ocr = load_ocr()
    items = TOPS.get(req.session_id, [])
    rows = []

    for i, item in enumerate(items):
        text, cands = recognize_plate(item["raw"], item["crop"], ocr)
        rows.append({
            "rank": i + 1,
            "score": round(float(item["score"]), 4),
            "text": text,
            "cands": cands,
            "image": encode_jpg(item["crop"])
        })

    return {"rows": rows}

@app.get("/new_session")
def new_session():
    return {"session_id": str(uuid.uuid4())}