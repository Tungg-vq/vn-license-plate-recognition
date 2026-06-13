import os
import pandas as pd
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "result/training/Medium with data augmentation with OBB data/weights/best.pt"
YAML_PATH = "data/YOLO data/OBB data/data.yaml"
TEST_IMG_DIR = "data/YOLO data/OBB data/test/images"

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print("Error: best.pt file not found.")
        exit()

    print("Loading YOLO OBB model...")
    model = YOLO(MODEL_PATH)

    print("\n--- STARTING EVALUATION ON TEST SET ---")
    
    metrics = model.val(
        data=YAML_PATH,
        split='test',             
        project='result/evaluation',
        name='OBB_Test_Report'
    )
    
    res = metrics.results_dict
    
    precision = res.get('metrics/precision(O)', res.get('metrics/precision(B)', 0))
    recall = res.get('metrics/recall(O)', res.get('metrics/recall(B)', 0))
    map50 = res.get('metrics/mAP50(O)', res.get('metrics/mAP50(B)', 0))
    map50_95 = res.get('metrics/mAP50-95(O)', res.get('metrics/mAP50-95(B)', 0))

    print("\n" + "="*50)
    print(" 🏆 FINAL TEST SET METRICS")
    print("="*50)
    print(f" Precision (P)    : {precision:.5f}")
    print(f" Recall (R)       : {recall:.5f}")
    print(f" mAP@50           : {map50:.5f}")
    print(f" mAP@50-95        : {map50_95:.5f}")
    print("="*50)

    csv_path = "result/evaluation/OBB_Test_Report/test_metrics.csv"
    df = pd.DataFrame([res])
    df.to_csv(csv_path, index=False)
    print(f"-> Full metrics saved to: {csv_path}\n")

    print("--- STARTING INFERENCE FOR VISUAL RESULTS ---")
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