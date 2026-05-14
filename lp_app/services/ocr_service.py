import difflib
import re

import cv2

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:
    RapidOCR = None

try:
    from streamlit_app.src.ocr_engine import recognize_plate, recognize_plate_fast
except Exception:
    recognize_plate = None
    recognize_plate_fast = None

from lp_app.config import MIN_TRACK_HITS


VALID_SERIES_LETTERS = set("ABCDEFGHKLMNPRSTUVXYZ")

DIGIT_FIX = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "T": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}

SERIES_DIGIT_FIX = {
    **DIGIT_FIX,
    "H": "1",
    "M": "",
    "N": "",
}

SERIES_LETTER_FIX = {
    "0": "D",
    "1": "I",
    "5": "S",
    "6": "G",
    "8": "B",
}


def clean_text(text):
    if not text:
        return ""

    text = str(text).upper()
    text = text.replace("\r", "\n")
    text = text.replace("|", "1")
    text = re.sub(r"[ \t_]+", "", text)
    text = re.sub(r"[^A-Z0-9.\-\n]", "", text)

    lines = []
    for line in text.split("\n"):
        line = line.strip(".-")
        if line:
            lines.append(line)

    return "\n".join(lines).strip()


def only_alnum(text):
    return re.sub(r"[^A-Z0-9]", "", clean_text(text))


def plate_key(text):
    return only_alnum(text)


def to_digit(ch):
    return DIGIT_FIX.get(ch, ch)


def fix_digits(text):
    out = []
    for ch in str(text).upper():
        ch = to_digit(ch)
        if ch.isdigit():
            out.append(ch)
    return "".join(out)


def normalize_series_letter(ch):
    ch = SERIES_LETTER_FIX.get(ch, ch)

    if ch == "I":
        ch = "H"

    if ch == "O" or ch == "Q":
        ch = "D"

    return ch if ch in VALID_SERIES_LETTERS else ""


def normalize_series_digit(ch):
    ch = SERIES_DIGIT_FIX.get(ch, ch)
    return ch if ch.isdigit() else ""


def extract_series(raw):
    """
    Chỉ cho phép series an toàn:
    - 1 chữ cái
    - hoặc 1 chữ cái + 1 số
    Ví dụ:
    F1, V2, Y3, H
    Không giữ dạng dài như FHM, 9VZ, VZZ.
    """
    raw = only_alnum(raw)

    letter = ""
    digit = ""
    start_idx = -1

    for i, ch in enumerate(raw[:5]):
        cand = normalize_series_letter(ch)
        if cand:
            letter = cand
            start_idx = i
            break

    if not letter:
        return ""

    for ch in raw[start_idx + 1 : start_idx + 5]:
        cand = normalize_series_digit(ch)
        if cand:
            digit = cand
            break

    return letter + digit


def format_tail(tail, prefer_four=False):
    digits = fix_digits(tail)

    if not digits:
        return ""

    if prefer_four and len(digits) >= 4:
        digits = digits[-4:]
        return digits

    if len(digits) >= 5:
        digits = digits[-5:]
        return f"{digits[:3]}.{digits[3:]}"

    if len(digits) == 4:
        return digits

    return digits


def build_plate(province, series, tail, two_line=False, prefer_four_tail=False):
    province = fix_digits(province)[:2]
    series = extract_series(series)
    tail = format_tail(tail, prefer_four=prefer_four_tail)

    if len(province) != 2 or not province.isdigit():
        return ""

    if not series or not tail:
        return ""

    if len(series) >= 2:
        head = f"{province}-{series}"
    else:
        head = f"{province}{series}"

    if two_line:
        return f"{head}\n{tail}"

    return f"{head}-{tail}"


def parse_two_line_from_lines(lines):
    """
    Dành cho biển 2 dòng.
    Sửa các lỗi:
    - 59-9VZ / 761.77 -> 59-V2 / 761.77
    - 59-FHM / 299.21 -> 59-F1 / 299.21
    - 52Y / 36347 -> 52-Y3 / 6347
    """
    if len(lines) < 2:
        return []

    top = only_alnum(lines[0])
    bottom = fix_digits("".join(lines[1:]))

    if len(top) < 3 or len(bottom) < 4:
        return []

    province = fix_digits(top[:2])
    rest = top[2:]
    series = extract_series(rest)

    candidates = []

    if len(province) == 2 and series:
        candidates.append(
            build_plate(
                province,
                series,
                bottom,
                two_line=True,
                prefer_four_tail=False,
            )
        )

    if len(province) == 2 and len(series) == 1 and len(bottom) >= 5:
        borrowed_digit = bottom[0]
        new_bottom = bottom[1:]

        candidates.append(
            build_plate(
                province,
                series + borrowed_digit,
                new_bottom,
                two_line=True,
                prefer_four_tail=True,
            )
        )

    return [c for c in candidates if c]


def parse_key_candidates(key, prefer_two_line=False):
    """
    Parse từ chuỗi dính liền:
    - 50Y109597 -> 50-Y1 / 095.97 nếu prefer_two_line=True
    - 51H10796 -> 51H-107.96
    - 52Y36347 -> 52-Y3 / 6347 nếu prefer_two_line=True
    """
    key = only_alnum(key)
    candidates = []

    if len(key) < 6:
        return candidates

    starts = [0]
    if len(key) >= 8:
        starts.append(1)

    tail_order = [4, 5] if prefer_two_line else [5, 4]

    for start in starts:
        sub = key[start:]

        for tail_len in tail_order:
            if len(sub) <= 2 + tail_len:
                continue

            head = sub[:-tail_len]
            tail = sub[-tail_len:]

            if len(head) < 3:
                continue

            province = fix_digits(head[:2])
            rest = head[2:]
            series = extract_series(rest)

            if len(province) != 2 or not series:
                continue

            prefer_four_tail = prefer_two_line and tail_len == 4

            plate = build_plate(
                province,
                series,
                tail,
                two_line=prefer_two_line,
                prefer_four_tail=prefer_four_tail,
            )

            if plate:
                candidates.append(plate)

    return candidates


def format_candidates_from_text(text, prefer_two_line=False):
    text = clean_text(text)
    if not text:
        return []

    out = []
    lines = [line for line in text.split("\n") if line.strip()]

    if len(lines) >= 2:
        out.extend(parse_two_line_from_lines(lines))

    key = only_alnum(text)

    out.extend(parse_key_candidates(key, prefer_two_line=prefer_two_line))
    out.extend(parse_key_candidates(key, prefer_two_line=False))

    if prefer_two_line:
        out.extend(parse_key_candidates(key, prefer_two_line=True))

    seen = set()
    final = []

    for item in out:
        item = clean_text(item)
        key = plate_key(item)

        if len(key) < 6:
            continue

        if key not in seen:
            final.append(item)
            seen.add(key)

    return final


def norm(result):
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
        return norm(result[0])

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


def is_two_line_crop(crop, raw_text=""):
    if "\n" in clean_text(raw_text):
        return True

    if crop is None or crop.size == 0:
        return False

    h, w = crop.shape[:2]
    ratio = w / max(h, 1)

    return ratio < 3.4


def score_plate(text, prefer_two_line=False):
    text = clean_text(text)
    key = plate_key(text)

    if len(key) < 6:
        return -5.0

    score = 0.0

    digits = sum(ch.isdigit() for ch in key)
    letters = sum(ch.isalpha() for ch in key)

    if 6 <= len(key) <= 10:
        score += 1.0

    if digits >= 4:
        score += 1.0

    if len(key) >= 2 and key[:2].isdigit():
        score += 1.2

    if len(key) >= 4 and key[-4:].isdigit():
        score += 0.8

    if letters <= 2:
        score += 0.8
    elif letters == 3:
        score += 0.1
    else:
        score -= 2.0

    if "\n" in text:
        score += 1.0 if prefer_two_line else -0.3

    if text.count(".") > 1:
        score -= 1.5

    if re.search(r"[A-Z]{3,}", key):
        score -= 1.2

    if re.search(r"\d{2}-?[A-Z]\d", text):
        score += 1.0

    if re.search(r"\d{3}\.\d{2}$", text):
        score += 0.8

    if re.search(r"\n\d{4}$", text):
        score += 0.8

    return score


def duplicate(a, b):
    ka = plate_key(a)
    kb = plate_key(b)

    if len(ka) < 4 or len(kb) < 4:
        return False

    if ka == kb:
        return True

    return difflib.SequenceMatcher(None, ka, kb).ratio() >= 0.86


def deduplicate(records):
    kept = []

    for record in sorted(records, key=lambda x: x["quality"], reverse=True):
        if not any(duplicate(record["text"], old["text"]) for old in kept):
            kept.append(record)

    return sorted(kept, key=lambda x: x["first_frame"])


class OCRService:
    def __init__(self):
        self.ocr = None

    def engine(self):
        if self.ocr is None:
            if RapidOCR is None:
                raise RuntimeError("Chưa cài rapidocr-onnxruntime")
            self.ocr = RapidOCR()

        return self.ocr

    def raw(self, crop):
        try:
            return norm(self.engine()(crop))
        except Exception:
            return ""

    def read_project_ocr(self, crop):
        for fn in [recognize_plate, recognize_plate_fast]:
            if fn is None:
                continue

            try:
                return norm(fn(crop, self.engine()))
            except TypeError:
                try:
                    return norm(fn(crop))
                except Exception:
                    pass
            except Exception:
                pass

        return self.raw(crop)

    def rotate_crop(self, crop, angle):
        h, w = crop.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        return cv2.warpAffine(
            crop,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def make_variants(self, crop):
        variants = []

        if crop is None or crop.size == 0:
            return variants

        variants.append(crop)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        gray_clahe = clahe.apply(gray)
        clahe_bgr = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
        variants.append(clahe_bgr)

        blur = cv2.GaussianBlur(gray_clahe, (0, 0), 1.0)
        sharp = cv2.addWeighted(gray_clahe, 1.45, blur, -0.45, 0)
        sharp_bgr = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
        variants.append(sharp_bgr)

        h, w = crop.shape[:2]
        ratio = w / max(h, 1)

        if ratio < 5.5:
            variants.append(self.rotate_crop(sharp_bgr, -3))
            variants.append(self.rotate_crop(sharp_bgr, 3))

        return variants

    def split_line_candidates(self, crop):
        if crop is None or crop.size == 0:
            return []

        h, w = crop.shape[:2]
        ratio = w / max(h, 1)

        if h < 40 or ratio >= 3.9:
            return []

        mid = h // 2
        top = crop[:mid, :]
        bottom = crop[mid:, :]

        top_text = self.raw(top)
        bottom_text = self.raw(bottom)

        texts = []

        if top_text and bottom_text:
            texts.append(clean_text(top_text + "\n" + bottom_text))

        return texts

    def candidates(self, crop):
        texts = []

        for variant in self.make_variants(crop):
            raw_text = self.read_project_ocr(variant)

            if raw_text:
                texts.append(raw_text)

            for split_text in self.split_line_candidates(variant):
                if split_text:
                    texts.append(split_text)

        out = []
        seen = set()

        for raw_text in texts:
            prefer_two = is_two_line_crop(crop, raw_text)

            formatted = format_candidates_from_text(
                raw_text,
                prefer_two_line=prefer_two,
            )

            for candidate in formatted:
                key = plate_key(candidate)

                if len(key) < 6:
                    continue

                if key not in seen:
                    out.append(candidate)
                    seen.add(key)

        return out

    def best_text(self, candidates):
        if not candidates:
            return "", None, 0.0

        votes = {}

        for item in candidates:
            crop = item["crop"]
            quality = float(item.get("quality", 0.0))
            prefer_two = is_two_line_crop(crop)

            for text in self.candidates(crop):
                key = plate_key(text)

                if len(key) < 6:
                    continue

                score = quality
                score += score_plate(text, prefer_two_line=prefer_two)
                score += min(len(key), 10) * 0.03

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
            best = max(candidates, key=lambda x: x.get("quality", 0.0))
            return "", best["crop"], best.get("quality", 0.0)

        best = max(
            votes.values(),
            key=lambda x: (
                x["score"],
                x["count"],
                score_plate(x["text"]),
                x["quality"],
            ),
        )

        return best["text"], best["crop"], best["quality"]

    def records_from_tracks(self, tracks):
        records = []

        for tid, track in sorted(tracks.items(), key=lambda x: x[1]["first_frame"]):
            if track["hits"] < MIN_TRACK_HITS or not track["candidates"]:
                continue

            text, crop, quality = self.best_text(track["candidates"])

            if len(plate_key(text)) < 6:
                continue

            records.append(
                {
                    "track_id": tid,
                    "text": text,
                    "crop": crop,
                    "hits": track["hits"],
                    "first_frame": track["first_frame"],
                    "conf": track["best_conf"],
                    "quality": quality,
                }
            )

        return deduplicate(records)