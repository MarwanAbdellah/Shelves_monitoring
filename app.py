# app.py
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from shelf_detection import detect_shelves
from product_detection import detect_products
from shelf_analysis import analyze_shelves

shelf_model = YOLO(r"shelves/train3/weights/best.pt")
product_model = YOLO(r"products/train8/weights/best.pt")

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
        margin: auto;
    }
    .stTextArea textarea {
        height: 640px !important;
        font-size: 18px;
        padding: 12px;
        line-height: 1.6;
        background-color: white !important;
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📦 Smart Shelf Analyzer")

uploaded = st.file_uploader("Upload shelf image", type=["jpg", "jpeg", "png"])
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    shelves = detect_shelves(image, shelf_model)
    products = detect_products(image, product_model)
    final_img, shelf_info = analyze_shelves(image, shelves, products)

    col1, col2 = st.columns([2, 1.2])

    with col1:
        st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), caption="Shelf Detection", use_container_width=True)

    with col2:
        st.text_area("🧾 Shelf Summary", shelf_info, height=640)
