import os
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "result/training/Medium with data augmentation with OBB data/weights/best.pt"
TEST_IMG_DIR = "data/YOLO data/OBB data/test/images"
OUTPUT_DIR = "data/OCR data/cropped_plates"

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model = YOLO(MODEL_PATH)
    
    results = model.predict(source=TEST_IMG_DIR, conf=0.5)

    for r in results:
        img_name = os.path.basename(r.path)
        img = cv2.imread(r.path)
        obb = r.obb

        if obb is None or len(obb) == 0:
            continue

        for i, box in enumerate(obb.xyxyxyxy):
            pts = box.cpu().numpy().astype(np.float32)
            
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            (tl, tr, br, bl) = rect
            
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

            base_name = os.path.splitext(img_name)[0]
            out_name = f"{base_name}_crop_{i}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            
            cv2.imwrite(out_path, warped)

    print(f"Crop and warp complete. Images saved to: {OUTPUT_DIR}")