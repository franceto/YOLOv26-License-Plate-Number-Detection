from .plate_preprocess import make_variants, make_variants_video, split_lines
from .postprocess import clean_text, pick_plate

def ocr_read(ocr, img):
    try:
        res, _ = ocr(img)
        if not res:
            return []
        out = []
        for x in res:
            txt = x[1] if len(x) >= 2 else ""
            cf = float(x[2]) if len(x) >= 3 else 0.5
            n = 3 if cf >= 0.85 else 2 if cf >= 0.65 else 1
            out += [txt] * n
        return out
    except Exception:
        return []

def recognize_plate(raw_crop, pad_crop, ocr):
    all_cands, top_cands, bottom_cands = [], [], []

    for name, img in make_variants(raw_crop, pad_crop):
        full = ocr_read(ocr, img)
        all_cands += full

        lines = split_lines(img)
        if len(lines) >= 2:
            top = ocr_read(ocr, lines[0])
            bot = ocr_read(ocr, lines[1])
            top_cands += top
            bottom_cands += bot
            all_cands += top + bot

    all_cands = [clean_text(x) for x in all_cands if clean_text(x)]
    top_cands = [clean_text(x) for x in top_cands if clean_text(x)]
    bottom_cands = [clean_text(x) for x in bottom_cands if clean_text(x)]

    return pick_plate(top_cands, bottom_cands, all_cands)
def recognize_plate_fast(raw_crop, pad_crop, ocr):
    all_cands, top_cands, bottom_cands = [], [], []
    variants = make_variants_video(raw_crop, pad_crop)

    for i, (name, img) in enumerate(variants):
        full = ocr_read(ocr, img)
        all_cands += full

        if i < 2:
            lines = split_lines(img)
            if len(lines) >= 2:
                top = ocr_read(ocr, lines[0])
                bot = ocr_read(ocr, lines[1])
                top_cands += top
                bottom_cands += bot
                all_cands += top + bot

    all_cands = [clean_text(x) for x in all_cands if clean_text(x)]
    top_cands = [clean_text(x) for x in top_cands if clean_text(x)]
    bottom_cands = [clean_text(x) for x in bottom_cands if clean_text(x)]

    return pick_plate(top_cands, bottom_cands, all_cands)