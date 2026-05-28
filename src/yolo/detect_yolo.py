import cv2
import os
import glob
from ultralytics import YOLO

def crop_plates_from_yolo():
    model = YOLO("result/training/Medium with data augmentation/weights/best.pt")
    input_folder = "data/YOLO data/images/test"
    output_folder = "cropped_images"
    os.makedirs(output_folder, exist_ok=True)

    image_paths = glob.glob(os.path.join(input_folder, "*.*"))
    print(f"Working on {len(image_paths)} images...")

    for path in image_paths:
        img = cv2.imread(path)
        if img is None: continue

        results = model.predict(path, verbose=False)
        result = results[0]

        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)

        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            
            h, w = img.shape[:2]
            pad = 5 
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            cropped_plate = img[y1:y2, x1:x2]
            
            save_path = os.path.join(output_folder, f"{name}_plate_{i}{ext}")
            cv2.imwrite(save_path, cropped_plate)
            
    print(f"Done! Open the folder '{output_folder}' to check the result")

if __name__ == "__main__":
    crop_plates_from_yolo()