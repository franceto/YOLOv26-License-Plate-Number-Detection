from pathlib import Path
import base64
import re
import threading
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:
    RapidOCR = None

try:
    from streamlit_app.src.ocr_engine import recognize_plate, recognize_plate_fast
except Exception:
    recognize_plate = None
    recognize_plate_fast = None


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "fastapi_app" / "static"
MODEL_PATH = ROOT / "weights" / "yolo26_bienso_best.pt"

app = FastAPI(title="YOLOv26 Realtime License Plate Demo")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MODEL = None
OCR_ENGINE = None
MODEL_LOCK = threading.Lock()
INFER_LOCK = threading.Lock()
SESSIONS = {}

TOP_K = 5


class FrameRequest(BaseModel):
    session_id: str
    image: str
    conf: float = 0.18
    imgsz: int = 640


class SessionRequest(BaseModel):
    session_id: str


def get_model():
    global MODEL

    with MODEL_LOCK:
        if MODEL is None:
            if not MODEL_PATH.exists():
                raise RuntimeError(f"Không tìm thấy model: {MODEL_PATH}")
            MODEL = YOLO(str(MODEL_PATH))

    return MODEL


def get_ocr():
    global OCR_ENGINE

    if OCR_ENGINE is None:
        if RapidOCR is None:
            raise RuntimeError("Chưa cài rapidocr-onnxruntime")
        OCR_ENGINE = RapidOCR()

    return OCR_ENGINE


def decode_base64_image(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Không decode được frame")

    return img


def image_to_base64(img):
    ok, buf = cv2.imencode(".jpg", img)

    if not ok:
        return ""

    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def clean_text(text):
    if not text:
        return ""

    text = str(text).upper()
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t_]+", "", text)
    text = text.replace("|", "1")
    text = re.sub(r"[^A-Z0-9.\-\n]", "", text)

    return text.strip()


def plate_key(text):
    text = clean_text(text)
    text = text.replace("\n", "")
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def normalize_ocr_result(result):
    if result is None:
        return ""

    if isinstance(result, str):
        return clean_text(result)

    if isinstance(result, dict):
        for key in ["text", "plate", "best_text", "final_text", "result"]:
            if result.get(key):
                return clean_text(result[key])
        return ""

    if isinstance(result, tuple):
        return normalize_ocr_result(result[0])

    if isinstance(result, list):
        texts = []

        for item in result:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                for key in ["text", "plate", "best_text"]:
                    if item.get(key):
                        texts.append(str(item[key]))
                        break
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                if isinstance(item[1], str):
                    texts.append(item[1])

        return clean_text("".join(texts))

    return clean_text(str(result))


def run_ocr(img):
    for fn in [recognize_plate, recognize_plate_fast]:
        if fn is None:
            continue

        try:
            return normalize_ocr_result(fn(img, get_ocr()))
        except TypeError:
            try:
                return normalize_ocr_result(fn(img))
            except Exception:
                pass
        except Exception:
            pass

    try:
        return normalize_ocr_result(get_ocr()(img))
    except Exception:
        return ""


def valid_plate_box(box, w, h):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    if bw < 10 or bh < 8:
        return False

    ratio = bw / max(bh, 1)
    area = bw * bh

    if ratio < 0.6 or ratio > 8.0:
        return False

    if area < 120:
        return False

    if area / max(w * h, 1) < 0.000025:
        return False

    return True


def clamp_box(box, w, h):
    x1, y1, x2, y2 = map(float, box)

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    return [int(x1), int(y1), int(x2), int(y2)]


def crop_with_pad(frame, box, pad_ratio=0.12):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w - 1, x2 + px)
    y2 = min(h - 1, y2 + py)

    return frame[y1:y2, x1:x2].copy()


def crop_quality(crop, conf, box, frame_w, frame_h):
    if crop is None or crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())

    x1, y1, x2, y2 = box
    area_ratio = ((x2 - x1) * (y2 - y1)) / max(frame_w * frame_h, 1)

    sharp_score = min(sharpness / 420.0, 1.0)
    bright_score = max(0.0, 1.0 - abs(brightness - 135.0) / 135.0)
    area_score = min(area_ratio * 130.0, 1.0)

    return 0.52 * float(conf) + 0.28 * sharp_score + 0.15 * bright_score + 0.05 * area_score


def push_candidate(session_id, track_id, crop, quality, conf):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {}

    tracks = SESSIONS[session_id]

    if track_id not in tracks:
        tracks[track_id] = {
            "track_id": track_id,
            "hits": 0,
            "best_conf": 0.0,
            "candidates": [],
        }

    t = tracks[track_id]
    t["hits"] += 1
    t["best_conf"] = max(t["best_conf"], float(conf))

    t["candidates"].append(
        {
            "crop": crop,
            "quality": quality,
            "conf": float(conf),
        }
    )

    t["candidates"] = sorted(
        t["candidates"],
        key=lambda x: x["quality"],
        reverse=True,
    )[:TOP_K]


def choose_best_text(candidates):
    votes = {}

    for item in candidates:
        crop = item["crop"]
        quality = item["quality"]

        text = run_ocr(crop)
        key = plate_key(text)

        if len(key) < 4:
            continue

        score = quality + min(len(key), 10) * 0.03

        if "\n" in text:
            score += 0.08

        if key not in votes:
            votes[key] = {
                "text": text,
                "score": 0.0,
                "count": 0,
                "crop": crop,
                "quality": quality,
            }

        votes[key]["score"] += score
        votes[key]["count"] += 1

        if quality > votes[key]["quality"]:
            votes[key]["crop"] = crop
            votes[key]["quality"] = quality
            votes[key]["text"] = text

    if not votes:
        best = candidates[0]
        return "", best["crop"], best["quality"]

    best = max(votes.values(), key=lambda x: (x["count"], x["score"], x["quality"]))
    return best["text"], best["crop"], best["quality"]


@app.get("/")
def home():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/new_session")
def new_session():
    session_id = uuid.uuid4().hex[:12]
    SESSIONS[session_id] = {}

    try:
        model = get_model()
        model.predictor = None
    except Exception:
        pass

    return {"session_id": session_id}


@app.post("/api/reset")
def reset(req: SessionRequest):
    SESSIONS[req.session_id] = {}

    try:
        model = get_model()
        model.predictor = None
    except Exception:
        pass

    return {"ok": True}


@app.post("/api/detect_frame")
def detect_frame(req: FrameRequest):
    frame = decode_base64_image(req.image)
    h, w = frame.shape[:2]

    model = get_model()

    with INFER_LOCK:
        try:
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=req.conf,
                imgsz=req.imgsz,
                verbose=False,
            )
        except Exception:
            results = model.predict(
                frame,
                conf=req.conf,
                imgsz=req.imgsz,
                verbose=False,
            )

    boxes_out = []

    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))

        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        else:
            ids = np.arange(1, len(xyxy) + 1)

        for box_raw, conf, track_id in zip(xyxy, confs, ids):
            box = clamp_box(box_raw, w, h)

            if not valid_plate_box(box, w, h):
                continue

            crop = crop_with_pad(frame, box)
            quality = crop_quality(crop, conf, box, w, h)

            push_candidate(
                req.session_id,
                int(track_id),
                crop,
                quality,
                conf,
            )

            boxes_out.append(
                {
                    "track_id": int(track_id),
                    "conf": round(float(conf), 4),
                    "box": box,
                }
            )

    return {
        "frame_w": w,
        "frame_h": h,
        "boxes": boxes_out,
        "total_tracks": len(SESSIONS.get(req.session_id, {})),
        "time": time.time(),
    }


@app.post("/api/ocr_best")
def ocr_best(req: SessionRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Không tìm thấy session")

    records = []
    seen = set()

    tracks = SESSIONS[req.session_id]

    for track_id, t in tracks.items():
        candidates = t["candidates"]

        if not candidates:
            continue

        text, crop, quality = choose_best_text(candidates)
        key = plate_key(text)

        if len(key) < 4:
            continue

        if key in seen:
            continue

        seen.add(key)

        records.append(
            {
                "track_id": int(track_id),
                "text": text,
                "image": image_to_base64(crop),
                "hits": int(t["hits"]),
                "conf": round(float(t["best_conf"]), 4),
                "quality": round(float(quality), 4),
            }
        )

    records = sorted(records, key=lambda x: x["quality"], reverse=True)

    return {
        "total": len(records),
        "results": records,
    }