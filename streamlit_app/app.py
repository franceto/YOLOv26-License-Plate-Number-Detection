import streamlit as st
import cv2, tempfile, time, io, hashlib
import numpy as np
from pathlib import Path
from PIL import Image

from src.config import MODEL_PATH, IMAGE_TYPES, VIDEO_TYPES
from src.detector import detect_image
from src.image_utils import fmt_bytes, bgr2rgb
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
    ocr_every = st.slider("OCR mỗi N frame", 1, 30, 10)

    if mode == "Webcam realtime":
        cam_id = st.number_input("Camera ID", 0, 5, 0)
        max_frames = st.slider("Số frame chạy", 30, 1000, 300, 30)
        start = st.button("Start realtime")
        frame_box = st.empty()
        text_box = st.empty()

        if start:
            cap = cv2.VideoCapture(int(cam_id))
            texts = []

            for k in range(max_frames):
                ok, frame = cap.read()
                if not ok:
                    break

                out, crops = detect_image(frame, conf_vid, k % ocr_every == 0)

                for item in crops:
                    if item["text"]:
                        texts.append(item["text"])

                frame_box.image(bgr2rgb(out), width="stretch")
                text_box.write(sorted(set(texts))[-10:])
                time.sleep(0.01)

            cap.release()

    else:
        vid_file = st.file_uploader("Upload video", type=VIDEO_TYPES, key="vid_file")
        run = st.button("Detect video")

        if vid_file and run:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_file.name).suffix)
            tmp.write(vid_file.read())

            cap = cv2.VideoCapture(tmp.name)
            w, h = int(cap.get(3)), int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            prog = st.progress(0)
            texts, k = [], 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                out, crops = detect_image(frame, conf_vid, k % ocr_every == 0)

                for item in crops:
                    if item["text"]:
                        texts.append(item["text"])

                writer.write(out)
                k += 1

                if total:
                    prog.progress(min(k / total, 1.0))

            cap.release()
            writer.release()

            st.video(out_path)
            st.subheader("Kết quả OCR")
            st.write(sorted(set(texts)))