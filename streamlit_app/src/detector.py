import cv2
import numpy as np
from .loaders import load_model, load_ocr
from .image_utils import safe_crop
from .ocr_engine import recognize_plate

def infer_boxes(model, img, conf, offset=(0, 0), imgsz=960):
    res = model(img, conf=conf, iou=0.45, imgsz=imgsz, verbose=False)[0]
    boxes = []
    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        s = float(b.conf[0])
        ox, oy = offset
        boxes.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy, s])
    return boxes

def tile_boxes(model, img, conf, tile=960, overlap=0.3):
    h, w = img.shape[:2]
    if max(h, w) <= tile:
        return []
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(1, w - tile + 1), step))
    ys = list(range(0, max(1, h - tile + 1), step))
    if xs[-1] != max(0, w - tile):
        xs.append(max(0, w - tile))
    if ys[-1] != max(0, h - tile):
        ys.append(max(0, h - tile))
    boxes = []
    for y in ys:
        for x in xs:
            crop = img[y:min(y+tile, h), x:min(x+tile, w)]
            boxes += infer_boxes(model, crop, conf, (x, y), imgsz=tile)
    return boxes

def nms_boxes(boxes, iou=0.45):
    if not boxes:
        return []
    xywh = [[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2, s in boxes]
    scores = [s for *_, s in boxes]
    idx = cv2.dnn.NMSBoxes(xywh, scores, 0.001, iou)
    if len(idx) == 0:
        return []
    idx = np.array(idx).reshape(-1)
    return [boxes[i] for i in idx]

def detect_image(img, conf, do_ocr=True, use_tiles=True, imgsz=960):
    model = load_model()
    ocr = load_ocr() if do_ocr else None

    boxes = infer_boxes(model, img, conf, imgsz=imgsz)
    if use_tiles:
        boxes += tile_boxes(model, img, max(conf * 0.8, 0.15), tile=imgsz, overlap=0.3)

    boxes = sorted(nms_boxes(boxes, 0.45), key=lambda x: x[4], reverse=True)

    out = img.copy()
    crops = []
    H, W = img.shape[:2]

    for x1, y1, x2, y2, score in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        raw = img[y1:y2, x1:x2]
        pad, box = safe_crop(img, x1, y1, x2, y2, 0.08)

        text, cands = recognize_plate(raw, pad, ocr) if ocr else ("", [])
        crops.append({"crop": pad, "raw": raw, "text": text, "score": score, "cands": cands})

        px1, py1, px2, py2 = box
        cv2.rectangle(out, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.putText(out, f"Bien-so {score:.2f}", (px1, max(25, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return out, crops
def detect_frame_fast(img, conf, imgsz=640):
    model = load_model()
    boxes = infer_boxes(model, img, conf, imgsz=imgsz)
    boxes = sorted(nms_boxes(boxes, 0.45), key=lambda x: x[4], reverse=True)

    out = img.copy()
    crops = []
    H, W = img.shape[:2]

    for x1, y1, x2, y2, score in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        raw = img[y1:y2, x1:x2]
        pad, box = safe_crop(img, x1, y1, x2, y2, 0.08)

        crops.append({
            "crop": pad,
            "raw": raw,
            "text": "",
            "score": score,
            "cands": [],
            "box": box
        })

        px1, py1, px2, py2 = box
        cv2.rectangle(out, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.putText(out, f"Bien-so {score:.2f}", (px1, max(25, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return out, crops
def detect_frame_fast(img, conf, imgsz=512):
    model = load_model()
    boxes = infer_boxes(model, img, conf, imgsz=imgsz)
    boxes = sorted(nms_boxes(boxes, 0.45), key=lambda x: x[4], reverse=True)

    out = img.copy()
    crops = []
    H, W = img.shape[:2]

    for x1, y1, x2, y2, score in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        raw = img[y1:y2, x1:x2]
        pad, box = safe_crop(img, x1, y1, x2, y2, 0.08)

        crops.append({
            "crop": pad,
            "raw": raw,
            "text": "",
            "score": score,
            "cands": []
        })

        px1, py1, px2, py2 = box
        cv2.rectangle(out, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.putText(out, f"Bien-so {score:.2f}", (px1, max(25, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return out, crops