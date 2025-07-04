import cv2
import numpy as np

def detect_shelves(image, shelf_model):
    shelves = shelf_model.predict(image, imgsz=320)[0]
    shelf_polygons = shelves.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2).astype(int)
    return shelf_polygons