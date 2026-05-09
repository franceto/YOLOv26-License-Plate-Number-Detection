from .plate_preprocess import make_variants, split_lines
from .postprocess import clean_text, pick_plate

def ocr_read(ocr, img):
    try:
        res, _ = ocr(img)
        if not res:
            return []
        return [x[1] for x in res if len(x) >= 2]
    except Exception:
        return []

def recognize_plate(raw_crop, pad_crop, ocr):
    all_cands, top_cands, bottom_cands = [], [], []

    for name, img in make_variants(raw_crop, pad_crop):
        all_cands += ocr_read(ocr, img)

        lines = split_lines(img)
        if len(lines) >= 2:
            t = ocr_read(ocr, lines[0])
            b = ocr_read(ocr, lines[1])
            top_cands += t
            bottom_cands += b
            all_cands += t + b

    all_cands = [clean_text(x) for x in all_cands if clean_text(x)]
    top_cands = [clean_text(x) for x in top_cands if clean_text(x)]
    bottom_cands = [clean_text(x) for x in bottom_cands if clean_text(x)]

    return pick_plate(top_cands, bottom_cands, all_cands)