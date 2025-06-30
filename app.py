import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load models
shelf_model = YOLO(r"shelves\train3\weights\best.pt")
product_model = YOLO(r"products\train8\weights\best.pt")

# Page setup
st.set_page_config(layout="wide")
st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

st.title("📦 Smart Shelf Analyzer")

uploaded = st.file_uploader("Upload shelf image", type=["jpg", "jpeg", "png"])
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Run predictions
    shelves = shelf_model.predict(image, imgsz=320)[0]
    products = product_model.predict(image, imgsz=320)[0]

    product_boxes = products.boxes.xyxy.cpu().numpy().astype(int)
    shelf_polygons = shelves.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2).astype(int)

    final_img = image.copy()
    shelf_info = ""

    for shelf_id, shelf_poly in enumerate(shelf_polygons):
        count = 0
        product_widths = []
        polygon = shelf_poly.astype(np.int32).reshape(-1, 1, 2)

        # Draw shelf polygon and label
        cv2.polylines(final_img, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        label_pos = tuple(shelf_poly[0])
        cv2.putText(final_img, f"Shelf {shelf_id + 1}", label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)  # Yellow text

        # Count products inside the shelf
        for (x1, y1, x2, y2) in product_boxes:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if cv2.pointPolygonTest(polygon, (int(cx), int(cy)), False) >= 0:
                count += 1
                product_widths.append(x2 - x1)
                cv2.circle(final_img, (cx, cy), 5, (0, 255, 0), -1)

        # Estimate how many can fit using average width
        shelf_width = np.linalg.norm(shelf_poly[0] - shelf_poly[1])
        if product_widths:
            avg_width = np.median(product_widths)
            avg_width = np.median(product_widths)
            adjusted_width = avg_width * 0.85  # allow for spacing between products
            estimated_fit = int(shelf_width // adjusted_width)

        else:
            avg_width = 40
            estimated_fit = 0

        missing = max(estimated_fit - count, 0)

        # Add to shelf summary
        shelf_info += f"Shelf {shelf_id + 1}:\n"
        shelf_info += f"🟩 Detected: {count}\n⬜ Empty: {missing}\n\n"

    # Layout with image and text side-by-side
    col1, col2 = st.columns([2, 1.2])

    with col1:
        st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), caption="Shelf Detection", use_container_width=True)

    with col2:
        st.text_area("🧾 Shelf Summary", shelf_info, height=640)
