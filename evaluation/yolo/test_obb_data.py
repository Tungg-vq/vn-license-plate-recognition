import os
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "result/training/Medium with data augmentation with OBB data/weights/best.pt"
YAML_PATH = "data/YOLO data/OBB data/data.yaml"
TEST_IMG_DIR = "data/YOLO data/OBB data/test/images"

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print("Error: best.pt file not found.")
        exit()

    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data=YAML_PATH,
        split='test',             
        project='result/evaluation',
        name='OBB_Test_Report'
    )
    
    results = model.predict(
        source=TEST_IMG_DIR,
        save=True,               
        project='result/inference',
        name='OBB_Test_Visuals',
        conf=0.5                 
    )

    for r in results:
        img_name = os.path.basename(r.path)
        obb = r.obb 

        if obb is None or len(obb) == 0:
            continue

        points = obb.xyxyxyxy[0].cpu().numpy().astype(int) 
        conf = obb.conf[0].item()

        print(f"{img_name} | Conf: {conf:.2f} | Corners: {points.tolist()}")