import cv2
import numpy as np

def fmt_bytes(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"

def fit_canvas(img, max_w=900, max_h=520):
    canvas = np.full((max_h, max_w, 3), 245, dtype=np.uint8)
    if img is None or img.size == 0:
        return canvas
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y, x = (max_h - nh) // 2, (max_w - nw) // 2
    canvas[y:y+nh, x:x+nw] = r
    return canvas

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def safe_crop(img, x1, y1, x2, y2, pad_ratio=0.04):
    h, w = img.shape[:2]
    pad = max(4, int(pad_ratio * max(x2 - x1, y2 - y1)))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)