import cv2
import numpy as np

from lp_app.config import TOP_K_CROPS, CROP_MIN_H


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def fit_frame(frame, max_w, max_h):
    h, w = frame.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)

    if scale >= 1.0:
        return frame

    return cv2.resize(
        frame,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def clamp_box(box, w, h):
    x1, y1, x2, y2 = map(float, box)

    return [
        int(max(0, min(w - 1, x1))),
        int(max(0, min(h - 1, y1))),
        int(max(0, min(w - 1, x2))),
        int(max(0, min(h - 1, y2))),
    ]


def valid_plate_box(box, w, h):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    if bw < 10 or bh < 8:
        return False

    ratio = bw / max(bh, 1)
    area = bw * bh

    if ratio < 0.55 or ratio > 8.5:
        return False

    if area < 100:
        return False

    if area / max(w * h, 1) < 0.00002:
        return False

    return True


def crop_with_pad(frame, box, pad=0.22):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * pad)
    py = int(bh * pad)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w - 1, x2 + px)
    y2 = min(h - 1, y2 + py)

    return frame[y1:y2, x1:x2].copy()


def enhance_crop(crop, min_h=120, max_scale=5.0):
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]

    if h <= 0 or w <= 0:
        return crop

    scale = min(max_scale, max(1.0, min_h / max(h, 1)))

    if scale > 1.0:
        crop = cv2.resize(
            crop,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.2,
        tileGridSize=(8, 8),
    )
    l = clahe.apply(l)

    crop = cv2.merge([l, a, b])
    crop = cv2.cvtColor(crop, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(crop, (0, 0), 1.0)
    crop = cv2.addWeighted(crop, 1.45, blur, -0.45, 0)

    return crop


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

    return (
        0.52 * float(conf)
        + 0.28 * sharp_score
        + 0.15 * bright_score
        + 0.05 * area_score
    )


def parse_result(result, frame_idx=0):
    if not result or len(result) == 0 or result[0].boxes is None:
        return []

    boxes = result[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))

    if boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int)
    else:
        ids = np.arange(1, len(xyxy) + 1) + frame_idx * 10000

    items = []

    for box, conf, track_id in zip(xyxy, confs, ids):
        items.append(
            {
                "box": box,
                "conf": float(conf),
                "track_id": int(track_id),
            }
        )

    return items


def detect_predict(model, frame, conf, imgsz, device, half):
    results = model.predict(
        frame,
        conf=conf,
        imgsz=imgsz,
        device=device,
        half=half,
        verbose=False,
    )

    h, w = frame.shape[:2]
    out = []

    for item in parse_result(results):
        box = clamp_box(item["box"], w, h)

        if valid_plate_box(box, w, h):
            out.append(
                {
                    "box": box,
                    "conf": item["conf"],
                    "track_id": item["track_id"],
                }
            )

    return out


def detect_track(model, frame, conf, imgsz, device, half, tracker, frame_idx):
    try:
        results = model.track(
            frame,
            persist=True,
            tracker=tracker,
            conf=conf,
            imgsz=imgsz,
            device=device,
            half=half,
            verbose=False,
        )
    except Exception:
        results = model.predict(
            frame,
            conf=conf,
            imgsz=imgsz,
            device=device,
            half=half,
            verbose=False,
        )

    h, w = frame.shape[:2]
    out = []

    for item in parse_result(results, frame_idx):
        box = clamp_box(item["box"], w, h)

        if valid_plate_box(box, w, h):
            out.append(
                {
                    "box": box,
                    "conf": item["conf"],
                    "track_id": item["track_id"],
                }
            )

    return out


def draw_boxes(frame, boxes, total=0, fps=0):
    out = frame.copy()

    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        conf = item["conf"]
        track_id = item["track_id"]

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 80), 2)

        label = f"ID {track_id} | {conf:.2f}"
        y0 = max(0, y1 - 28)

        cv2.rectangle(out, (x1, y0), (x1 + 150, y1), (0, 255, 80), -1)

        cv2.putText(
            out,
            label,
            (x1 + 5, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    info = f"tracks: {total} | fps: {fps:.1f}"

    cv2.rectangle(out, (0, 0), (340, 34), (0, 0, 0), -1)

    cv2.putText(
        out,
        info,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )

    return out


class TrackStore:
    def __init__(self):
        self.tracks = {}

    def clear(self):
        self.tracks.clear()

    def add(self, track_id, crop, quality, conf, box, frame_idx):
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                "track_id": int(track_id),
                "hits": 0,
                "first_frame": frame_idx,
                "last_frame": frame_idx,
                "best_conf": 0.0,
                "candidates": [],
            }

        t = self.tracks[track_id]
        t["hits"] += 1
        t["last_frame"] = frame_idx
        t["best_conf"] = max(t["best_conf"], float(conf))

        t["candidates"].append(
            {
                "crop": crop,
                "quality": float(quality),
                "conf": float(conf),
                "box": box,
                "frame": frame_idx,
            }
        )

        t["candidates"] = sorted(
            t["candidates"],
            key=lambda x: x["quality"],
            reverse=True,
        )[:TOP_K_CROPS]

    def snapshot(self, limit=None):
        items = []

        for tid, t in sorted(self.tracks.items(), key=lambda x: x[1]["first_frame"]):
            if not t["candidates"]:
                continue

            best = max(t["candidates"], key=lambda x: x["quality"])

            items.append(
                {
                    "track_id": tid,
                    "crop": best["crop"].copy(),
                    "quality": best["quality"],
                }
            )

        items = sorted(items, key=lambda x: x["quality"], reverse=True)

        if limit:
            items = items[:limit]

        return items

    def __len__(self):
        return len(self.tracks)