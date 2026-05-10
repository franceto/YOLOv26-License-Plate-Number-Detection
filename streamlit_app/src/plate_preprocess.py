import cv2
import numpy as np

def resize_h(img, target=300):
    if img is None or img.size == 0:
        return img
    h = img.shape[0]
    s = max(1, target / max(1, h))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

def add_border(img, p=0.12):
    h, w = img.shape[:2]
    px, py = int(w * p), int(h * p)
    return cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_REPLICATE)

def gamma_correct(img, gamma):
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def clahe_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(g)

def gray3(img):
    return cv2.cvtColor(clahe_gray(img), cv2.COLOR_GRAY2BGR)

def low_light(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = g.mean()
    if m < 90:
        x = gamma_correct(img, 0.65)
    elif m > 180:
        x = gamma_correct(img, 1.35)
    else:
        x = img.copy()
    y = clahe_gray(x)
    return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)

def sharpen(img):
    g = clahe_gray(img)
    b = cv2.GaussianBlur(g, (0, 0), 1)
    s = cv2.addWeighted(g, 1.9, b, -0.9, 0)
    return cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)

def deblur(img):
    x = cv2.bilateralFilter(img, 5, 45, 45)
    g = clahe_gray(x)
    b = cv2.GaussianBlur(g, (0, 0), 1.2)
    s = cv2.addWeighted(g, 2.1, b, -1.1, 0)
    return cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)

def otsu(img):
    g = clahe_gray(img)
    t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

def adaptive(img):
    g = clahe_gray(img)
    t = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

def sr2(img):
    x = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return sharpen(x)

def rotate_img(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def deskew(img):
    if img is None or img.size == 0:
        return img
    x = resize_h(img, 300)
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
        x = add_border(resize_h(img, 300), 0.10)
        l = low_light(x)
        d = deskew(l)
        for n, v in [
            (name, x),
            (name + "_gray", gray3(x)),
            (name + "_light", l),
            (name + "_sharp", sharpen(x)),
            (name + "_deblur", deblur(x)),
            (name + "_otsu", otsu(x)),
            (name + "_adaptive", adaptive(x)),
            (name + "_sr2", sr2(x)),
            (name + "_sr2_light", sr2(l)),
            (name + "_deskew", d),
            (name + "_deskew_sharp", sharpen(d)),
            (name + "_deskew_sr2", sr2(d)),
        ]:
            imgs.append((n, v))
    return imgs

def split_lines(img):
    if img is None or img.size == 0:
        return []
    x = resize_h(img, 380)
    h = x.shape[0]
    gap = int(h * 0.05)
    return [x[:h//2+gap], x[h//2-gap:]]