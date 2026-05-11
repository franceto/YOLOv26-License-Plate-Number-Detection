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
    video_imgsz = st.select_slider("Kích thước inference", options=[416, 512, 640, 768], value=512)
    display_every = st.slider("Cập nhật giao diện mỗi N frame", 1, 10, 3)

    st.caption("Video chỉ hiển thị 1 khung: frame gốc + bbox detect. OCR chỉ chạy sau khi detect xong video và chỉ OCR top 3 bbox tốt nhất.")

    video_box = st.empty()
    stat_box = st.empty()
    result_box = st.container()

    def show_frame(frame):
        view = fit_canvas(frame, 960, 540)
        video_box.image(bgr2rgb(view), width="stretch")

    def crop_quality(item):
        crop = item["crop"]
        if crop is None or crop.size == 0:
            return 0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        h, w = crop.shape[:2]
        area = h * w
        return item["score"] * 100000 + min(sharp, 2000) * 10 + min(area, 50000) * 0.01

    def update_top3(top3, crops):
        top3 += crops
        top3 = sorted(top3, key=crop_quality, reverse=True)
        return top3[:3]

    def ocr_top3(top3):
        ocr = load_ocr()
        rows = []

        for item in top3:
            text, cands = recognize_plate(item["raw"], item["crop"], ocr)
            item["text"] = text
            item["cands"] = cands
            rows.append(item)

        return rows

    def show_ocr_rows(rows):
        with result_box:
            st.subheader("Kết quả OCR top 3 bbox tốt nhất")
            if not rows:
                st.warning("Không có bbox để OCR.")
            else:
                cols = st.columns(len(rows))
                for i, item in enumerate(rows):
                    with cols[i]:
                        st.image(bgr2rgb(fit_canvas(item["crop"], 320, 180)), caption=f"Top {i+1} | conf={item['score']:.2f}", width="stretch")
                        st.code(item["text"] if item["text"] else "Không đọc được")
                        st.write(item["cands"])

    if mode == "Webcam realtime":
        cam_id = st.number_input("Camera ID", 0, 5, 0)
        max_frames = st.slider("Số frame chạy", 30, 2000, 500, 30)
        start = st.button("Start realtime")

        if start:
            cap = cv2.VideoCapture(int(cam_id))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            top3, total_bbox, k = [], 0, 0

            while k < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break

                out, crops = detect_frame_fast(frame, conf_vid, imgsz=video_imgsz)
                top3 = update_top3(top3, crops)
                total_bbox += len(crops)

                if k % display_every == 0:
                    show_frame(out)
                    stat_box.info(f"Đang detect: frame {k+1} | Tổng bbox: {total_bbox} | Top bbox giữ lại: {len(top3)}")

                k += 1

            cap.release()

            stat_box.success(f"Detect xong: {k} frame | Tổng bbox: {total_bbox}. Đang OCR top 3 bbox...")
            rows = ocr_top3(top3)
            stat_box.success(f"Hoàn tất OCR top {len(rows)} bbox tốt nhất.")
            show_ocr_rows(rows)

    else:
        vid_file = st.file_uploader("Upload video", type=VIDEO_TYPES, key="vid_file")
        run = st.button("Detect video realtime")

        if vid_file and run:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_file.name).suffix)
            tmp.write(vid_file.read())

            cap = cv2.VideoCapture(tmp.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            top3, total_bbox, k = [], 0, 0
            prog = st.progress(0)

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                out, crops = detect_frame_fast(frame, conf_vid, imgsz=video_imgsz)
                top3 = update_top3(top3, crops)
                total_bbox += len(crops)

                if k % display_every == 0:
                    show_frame(out)
                    stat_box.info(f"Đang detect: frame {k+1}/{total} | Tổng bbox: {total_bbox} | Top bbox giữ lại: {len(top3)}")

                k += 1

                if total:
                    prog.progress(min(k / total, 1.0))

            cap.release()

            stat_box.success(f"Detect xong: {k} frame | Tổng bbox: {total_bbox}. Đang OCR top 3 bbox...")
            rows = ocr_top3(top3)
            stat_box.success(f"Hoàn tất OCR top {len(rows)} bbox tốt nhất.")
            show_ocr_rows(rows)