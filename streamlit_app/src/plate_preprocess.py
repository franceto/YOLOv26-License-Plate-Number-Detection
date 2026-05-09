import cv2
import numpy as np

def resize_h(img, target=280):
    if img is None or img.size == 0:
        return img
    h = img.shape[0]
    s = max(1, target / max(1, h))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

def add_border(img, p=0.12):
    h, w = img.shape[:2]
    px, py = int(w * p), int(h * p)
    return cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_REPLICATE)

def clahe_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(g)

def sharpen(img):
    g = clahe_gray(img)
    b = cv2.GaussianBlur(g, (0, 0), 1)
    s = cv2.addWeighted(g, 1.9, b, -0.9, 0)
    return cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)

def otsu(img):
    g = clahe_gray(img)
    t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

def otsu_inv(img):
    g = clahe_gray(img)
    t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

def adaptive(img):
    g = clahe_gray(img)
    t = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

def gray3(img):
    return cv2.cvtColor(clahe_gray(img), cv2.COLOR_GRAY2BGR)

def rotate_img(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def deskew(img):
    if img is None or img.size == 0:
        return img
    x = resize_h(img, 280)
    g = clahe_gray(x)
    t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, np.ones((5, 25), np.uint8))
    cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return x
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.03 * x.shape[0] * x.shape[1]:
        return x
    rect = cv2.minAreaRect(c)
    angle = rect[-1]
    angle = angle + 90 if angle < -45 else angle
    if abs(angle) > 18:
        return x
    return rotate_img(x, angle)

def make_variants(raw_crop, pad_crop):
    imgs = []
    for name, img in [("raw", raw_crop), ("pad", pad_crop)]:
        if img is None or img.size == 0:
            continue
        x = add_border(resize_h(img, 280), 0.10)
        d = deskew(x)
        for n, v in [
            (name, x),
            (name+"_gray", gray3(x)),
            (name+"_sharp", sharpen(x)),
            (name+"_otsu", otsu(x)),
            (name+"_adaptive", adaptive(x)),
            (name+"_deskew", d),
            (name+"_deskew_sharp", sharpen(d)),
            (name+"_deskew_otsu", otsu(d)),
        ]:
            imgs.append((n, v))
    return imgs

def split_lines(img):
    if img is None or img.size == 0:
        return []
    x = resize_h(img, 360)
    h = x.shape[0]
    gap = int(h * 0.04)
    return [x[:h//2+gap], x[h//2-gap:]]