import streamlit as st
from .config import MODEL_PATH

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO(str(MODEL_PATH))

@st.cache_resource
def load_ocr():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()