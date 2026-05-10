import streamlit as st
import cv2
from .image_utils import bgr2rgb, fit_canvas

def inject_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #f8fafc 100%);
    }

    h1 {
        font-weight: 800 !important;
        color: #111827 !important;
        letter-spacing: -0.5px;
        padding-bottom: 8px;
    }

    h2, h3 {
        color: #1f2937 !important;
        font-weight: 750 !important;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255,255,255,0.7);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding: 10px 18px;
        border-radius: 12px;
        color: #374151;
        font-weight: 650;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        color: white !important;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border: 1.5px dashed #93c5fd;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.25rem;
        font-weight: 750;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.25);
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.32);
    }

    .stSlider {
        background: white;
        padding: 12px 16px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stImage"] {
        background: white;
        padding: 10px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
    }

    .meta-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 12px 0 20px 0;
    }

    .meta-card {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 16px;
        padding: 15px 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    }

    .meta-title {
        font-size: 13px;
        color: #6b7280;
        font-weight: 600;
    }

    .meta-value {
        font-size: 16px;
        font-weight: 800;
        color: #111827;
        margin-top: 5px;
        word-break: break-word;
    }

    pre {
        background: #0f172a !important;
        color: #67e8f9 !important;
        border-radius: 14px !important;
        padding: 16px !important;
        font-size: 18px !important;
        font-weight: 800 !important;
        text-align: center;
    }

    div[data-testid="stExpander"] {
        background: white;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1500px;
    }

    @media (max-width: 900px) {
        .meta-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 600px) {
        .meta-grid {
            grid-template-columns: 1fr;
        }
    }
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