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

    mode = st.radio("Nguồn video", ["Webcam realtime", "Upload video"], horizontal=True)
    conf_vid = st.slider("Con_f video", 0.1, 0.9, 0.35, 0.05, key="conf_vid")
    video_imgsz = st.select_slider("Kích thước inference", options=[512, 640, 768, 960], value=640)
    display_every = st.slider("Hiển thị mỗi N frame", 1, 5, 2)
    max_ocr_crops = st.slider("Số crop OCR sau khi detect xong", 5, 100, 30, 5)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Video gốc")
        raw_box = st.empty()
    with c2:
        st.subheader("Detect trên video gốc")
        det_box = st.empty()

    stat_box = st.empty()
    ocr_box = st.empty()

    def show_frame(box, frame):
        view = fit_canvas(frame, 760, 430)
        box.image(bgr2rgb(view), width="stretch")

    def run_ocr_after_video(items):
        ocr = load_ocr()
        rows = []
        items = sorted(items, key=lambda x: x["score"], reverse=True)[:max_ocr_crops]

        for i, item in enumerate(items):
            text, cands = recognize_plate(item["raw"], item["crop"], ocr)
            rows.append({"crop": item["crop"], "text": text, "score": item["score"], "cands": cands})

        return rows

    if mode == "Webcam realtime":
        cam_id = st.number_input("Camera ID", 0, 5, 0)
        max_frames = st.slider("Số frame chạy", 30, 2000, 500, 30)
        start = st.button("Start realtime")

        if start:
            cap = cv2.VideoCapture(int(cam_id))
            all_crops, total_bbox, k = [], 0, 0

            while k < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break

                out, crops = detect_image(frame, conf_vid, False, use_tiles=False, imgsz=video_imgsz)
                all_crops += crops
                total_bbox += len(crops)

                if k % display_every == 0:
                    show_frame(raw_box, frame)
                    show_frame(det_box, out)
                    stat_box.info(f"Frame: {k+1} | Tổng bbox đã detect: {total_bbox}")

                k += 1
                time.sleep(0.001)

            cap.release()

            stat_box.success(f"Detect xong: {k} frame | Tổng bbox: {total_bbox}")
            ocr_box.info("Đang OCR các crop tốt nhất...")

            rows = run_ocr_after_video(all_crops)

            st.subheader("Kết quả OCR sau khi detect xong")
            if not rows:
                st.warning("Không có bbox để OCR.")
            else:
                cols = st.columns(min(len(rows), 5))
                for i, item in enumerate(rows):
                    with cols[i % len(cols)]:
                        st.image(bgr2rgb(fit_canvas(item["crop"], 320, 180)), caption=f"Crop {i+1} | conf={item['score']:.2f}", width="stretch")
                        st.code(item["text"] if item["text"] else "Không đọc được")
                        st.write(item["cands"])

    else:
        vid_file = st.file_uploader("Upload video", type=VIDEO_TYPES, key="vid_file")
        run = st.button("Detect video realtime")

        if vid_file and run:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_file.name).suffix)
            tmp.write(vid_file.read())

            cap = cv2.VideoCapture(tmp.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            all_crops, total_bbox, k = [], 0, 0
            prog = st.progress(0)

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                out, crops = detect_image(frame, conf_vid, False, use_tiles=False, imgsz=video_imgsz)
                all_crops += crops
                total_bbox += len(crops)

                if k % display_every == 0:
                    show_frame(raw_box, frame)
                    show_frame(det_box, out)
                    stat_box.info(f"Frame: {k+1}/{total} | Tổng bbox đã detect: {total_bbox}")

                k += 1
                if total:
                    prog.progress(min(k / total, 1.0))

            cap.release()

            stat_box.success(f"Detect xong: {k} frame | Tổng bbox: {total_bbox}")
            ocr_box.info("Đang OCR các crop tốt nhất...")

            rows = run_ocr_after_video(all_crops)

            st.subheader("Kết quả OCR sau khi detect xong")
            if not rows:
                st.warning("Không có bbox để OCR.")
            else:
                cols = st.columns(min(len(rows), 5))
                for i, item in enumerate(rows):
                    with cols[i % len(cols)]:
                        st.image(bgr2rgb(fit_canvas(item["crop"], 320, 180)), caption=f"Crop {i+1} | conf={item['score']:.2f}", width="stretch")
                        st.code(item["text"] if item["text"] else "Không đọc được")
                        st.write(item["cands"])