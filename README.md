```
# 🧠 Smart Shelf Analyzer with YOLOv8 & YOLOv11

A **Streamlit-based** smart shelf monitoring tool that uses **YOLOv8-OBB** for shelf detection and **YOLOv11** for product detection. It provides real-time insights into product counts per shelf and estimates the number of missing products using spatial analysis.

## 🚀 Features

- 🧠 Detects shelves using YOLOv8n-OBB (oriented bounding boxes)
- 📦 Detects products using YOLOv11
- 📊 Estimates how many products can fit on each shelf
- 🧮 Calculates number of missing (empty) spots
- 🖼️ Visualizes results with bounding boxes and shelf labels
- 🖥️ Streamlit-based UI for easy interaction

## 🧩 Requirements

- Python 3.10+
- Ultralytics (YOLOv8 / YOLOv11)
- OpenCV
- Streamlit
- NumPy

## 📁 Folder Structure

- `app.py` → Main Streamlit app
- `shelves/` → Shelf detection model directory
- `products/` → Product detection model directory
- `uploads/` → Uploaded images for analysis (optional)

## 🛠️ Usage

1. Install requirements:
   ```bash
   pip install ultralytics opencv-python streamlit numpy
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Upload a shelf image and get detailed product count and missing space estimates.

## 📌 Notes

- Empty estimation is based on average product width per shelf.
- Make sure your models are trained and exported to:
  - `shelves/train*/weights/best.pt`
  - `products/train*/weights/best.pt`

## 📸 Example Output

- Shelf 1: 🟩 Detected: 15 ⬜ Empty: 5  
- Shelf 2: 🟩 Detected: 10 ⬜ Empty: 3

## 📃 License

MIT License © 2025 Marwan Abdellah
```
'''