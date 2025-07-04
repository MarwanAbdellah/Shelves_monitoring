import numpy as np

def detect_products(image, product_model):
    products = product_model.predict(image, imgsz=320)[0]
    product_boxes = products.boxes.xyxy.cpu().numpy().astype(int)
    return product_boxes