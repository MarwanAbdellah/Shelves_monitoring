import cv2
import numpy as np

def analyze_shelves(image, shelf_polygons, product_boxes):
    result_img = image.copy()
    summary = ""

    for shelf_id, shelf_poly in enumerate(shelf_polygons):
        polygon = shelf_poly.reshape(-1, 1, 2)
        cv2.polylines(result_img, [polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        label_pos = tuple(shelf_poly[0])
        cv2.putText(result_img, f"Shelf {shelf_id + 1}", label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4, cv2.LINE_AA)

        count = 0
        widths = []
        for (x1, y1, x2, y2) in product_boxes:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if cv2.pointPolygonTest(polygon, (int(cx), int(cy)), False) >= 0:
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                widths.append(x2 - x1)
                count += 1

        shelf_width = np.linalg.norm(shelf_poly[0] - shelf_poly[1])
        avg_width = np.median(widths) if widths else 40
        max_fit = int(shelf_width // (avg_width * 0.85))
        missing = max(max_fit - count, 0)

        for i in range(missing):
            px = int(shelf_poly[0][0] + i * avg_width * 0.85)
            py = int((shelf_poly[0][1] + shelf_poly[1][1]) // 2 - avg_width * 0.4)
            cv2.rectangle(result_img, (px, py), (px + int(avg_width), py + int(avg_width * 0.8)), (0, 0, 255), 2)

        summary += f"Shelf {shelf_id + 1}:\n"
        summary += f"🟩 Detected: {count}\n⬜ Empty: {missing}\n\n"

    return result_img, summary