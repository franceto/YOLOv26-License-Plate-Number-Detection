import streamlit as st
import cv2, tempfile, time, io, hashlib
import numpy as np
from pathlib import Path
from PIL import Image

from src.config import MODEL_PATH, IMAGE_TYPES, VIDEO_TYPES
from src.detector import detect_image, detect_frame_fast
from src.loaders import load_ocr
from src.ocr_engine import recognize_plate
from src.image_utils import fmt_bytes, bgr2rgb, fit_canvas
from src.ui import inject_css, show_meta, show_compare, show_plates

st.set_page_config(page_title="YOLOv26 Biển số OCR", layout="wide")
st.title("YOLOv26 Detect → Crop → RapidOCR")
inject_css()

if not MODEL_PATH.exists():
    st.error(f"Không tìm thấy model: {MODEL_PATH}")
    st.stop()

tab1, tab2 = st.tabs(["Detect ảnh", "Detect video realtime"])

with tab1:
    st.header("1. Detect biển số")

    conf_img = st.slider("Con_f", 0.1, 0.9, 0.35, 0.05, key="conf_img")
    st.selectbox("OCR engine", ["RapidOCR ONNX"], index=0)
    img_file = st.file_uploader("Upload ảnh", type=IMAGE_TYPES, key="img_file")

    if img_file is None:
        for k in ["img_id", "img", "out", "crops", "meta"]:
            st.session_state.pop(k, None)
    else:
        img_bytes = img_file.getvalue()
        img_id = hashlib.md5(img_bytes).hexdigest()

        if st.session_state.get("img_id") != img_id:
            for k in ["out", "crops"]:
                st.session_state.pop(k, None)

            pil = Image.open(io.BytesIO(img_bytes))
            fmt = pil.format or Path(img_file.name).suffix.replace(".", "").upper()
            w, h = pil.size

            st.session_state["img_id"] = img_id
            st.session_state["meta"] = {
                "name": img_file.name,
                "size": fmt_bytes(len(img_bytes)),
                "resolution": f"{w} × {h}",
                "format": fmt
            }

            img = np.array(pil.convert("RGB"))
            st.session_state["img"] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        show_meta(st.session_state["meta"])

        if st.button("Detect", key="detect_img"):
            with st.spinner("Đang detect và OCR..."):
                out, crops = detect_image(st.session_state["img"], conf_img, True)
                st.session_state["out"] = out
                st.session_state["crops"] = crops

        if "out" in st.session_state:
            show_compare(st.session_state["img"], st.session_state["out"])
            show_plates(st.session_state["crops"])

with tab2:
    st.header("Detect video realtime")

    conf_vid = st.slider("Con_f video", 0.1, 0.9, 0.35, 0.05, key="conf_vid_realtime_single")
    video_imgsz = st.select_slider("Kích thước inference", options=[416, 512, 640, 768], value=512, key="video_imgsz_realtime_single")
    display_every = st.slider("Cập nhật khung hình mỗi N frame", 1, 10, 2, key="display_every_realtime_single")
    top_k = st.slider("Số bbox tốt nhất để OCR", 1, 10, 3, key="top_k_realtime_single")

    vid_file = st.file_uploader("Upload video", type=VIDEO_TYPES, key="vid_file_realtime_single")

    _, video_col, _ = st.columns([1, 2.2, 1])
    frame_box = video_col.empty()

    stat = st.empty()
    prog = st.empty()
    result_box = st.container()

    def crop_quality(item):
        crop = item["crop"]
        if crop is None or crop.size == 0:
            return 0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        h, w = crop.shape[:2]
        area = h * w
        return item["score"] * 100000 + min(sharp, 2000) * 10 + min(area, 50000) * 0.01

    def update_top(items, crops, k):
        items += crops
        items = sorted(items, key=crop_quality, reverse=True)
        return items[:k]

    def show_detect_frame(frame):
        view = fit_canvas(frame, 720, 405)
        frame_box.image(bgr2rgb(view), width="stretch")

    def ocr_best_crops(items):
        ocr = load_ocr()
        rows = []
        for item in items:
            text, cands = recognize_plate(item["raw"], item["crop"], ocr)
            item["text"] = text
            item["cands"] = cands
            rows.append(item)
        return rows

    if vid_file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_file.name).suffix)
        tmp.write(vid_file.read())

        cap0 = cv2.VideoCapture(tmp.name)
        ok, first_frame = cap0.read()
        cap0.release()

        if ok:
            show_detect_frame(first_frame)

        run = st.button("Bắt đầu detect realtime", key="start_detect_realtime_single")

        if run:
            cap = cv2.VideoCapture(tmp.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            bar = prog.progress(0)
            top_items, total_bbox, frame_idx = [], 0, 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                out, crops = detect_frame_fast(frame, conf_vid, imgsz=video_imgsz)

                top_items = update_top(top_items, crops, top_k)
                total_bbox += len(crops)

                if frame_idx % display_every == 0:
                    show_detect_frame(out)
                    stat.info(
                        f"Đang detect: {frame_idx + 1}/{total} frame | "
                        f"Tổng bbox: {total_bbox} | "
                        f"Top giữ lại: {len(top_items)}"
                    )

                frame_idx += 1

                if total:
                    bar.progress(min(frame_idx / total, 1.0))

            cap.release()

            stat.success(f"Detect xong: {frame_idx} frame | Tổng bbox: {total_bbox}")

            with st.spinner("Đang OCR top bbox tốt nhất..."):
                rows = ocr_best_crops(top_items)

            with result_box:
                st.subheader(f"OCR top {len(rows)} bbox tốt nhất")

                if not rows:
                    st.warning("Không có bbox để OCR.")
                else:
                    cols = st.columns(len(rows))
                    for i, item in enumerate(rows):
                        with cols[i]:
                            st.image(
                                bgr2rgb(fit_canvas(item["crop"], 320, 180)),
                                caption=f"Top {i+1} | conf={item['score']:.2f}",
                                width="stretch"
                            )
                            st.code(item["text"] if item["text"] else "Không đọc được")
                            st.write(item["cands"])