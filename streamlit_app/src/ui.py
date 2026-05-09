import streamlit as st
import cv2
from .image_utils import bgr2rgb, fit_canvas

def inject_css():
    st.markdown("""
    <style>
    .meta-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:10px 0 18px 0;}
    .meta-card{background:#f6f7fb;border-radius:12px;padding:12px 14px;border:1px solid #e5e7eb;}
    .meta-title{font-size:13px;color:#6b7280;}
    .meta-value{font-size:16px;font-weight:600;color:#111827;margin-top:4px;}
    </style>
    """, unsafe_allow_html=True)

def show_meta(meta):
    st.markdown(f"""
    <div class="meta-grid">
        <div class="meta-card"><div class="meta-title">Tên ảnh</div><div class="meta-value">{meta["name"]}</div></div>
        <div class="meta-card"><div class="meta-title">Kích thước ảnh</div><div class="meta-value">{meta["size"]}</div></div>
        <div class="meta-card"><div class="meta-title">Độ phân giải</div><div class="meta-value">{meta["resolution"]}</div></div>
        <div class="meta-card"><div class="meta-title">Định dạng</div><div class="meta-value">{meta["format"]}</div></div>
    </div>
    """, unsafe_allow_html=True)

def show_compare(img, out):
    c1, c2 = st.columns(2)
    img_view = fit_canvas(img, 900, 520)
    out_view = fit_canvas(out, 900, 520)
    c1.image(bgr2rgb(img_view), caption="Ảnh gốc", width="stretch")
    c2.image(bgr2rgb(out_view), caption="Kết quả detect", width="stretch")

def show_plate(item, i):
    view = fit_canvas(item["crop"], 320, 180)
    st.image(bgr2rgb(view), caption=f"Crop {i+1}", width="stretch")
    st.code(item["text"] if item["text"] else "Không đọc được")
    st.write(item["cands"])

def show_plates(crops):
    st.header("2. Trích xuất biển số")

    if len(crops) == 0:
        st.warning("Không phát hiện biển số.")
    elif len(crops) == 1:
        _, mid, _ = st.columns([1, 1.4, 1])
        with mid:
            show_plate(crops[0], 0)
    else:
        cols = st.columns(len(crops))
        for i, item in enumerate(crops):
            with cols[i]:
                show_plate(item, i)