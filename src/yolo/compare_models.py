import cv2
import numpy as np
from ultralytics import YOLO


HBB_WEIGHTS = "result/training/Medium with data augmentation/weights/best.pt"  
OBB_WEIGHTS = "result/training/Medium with data augmentation with OBB data/weights/best.pt"     
TEST_IMAGE_PATH = "data/inference data/Tgmt_0704.png" 


model_hbb = YOLO(HBB_WEIGHTS)
model_obb = YOLO(OBB_WEIGHTS)

img = cv2.imread(TEST_IMAGE_PATH)
if img is None:
    print("cannot find the image")
    exit()

results_hbb = model_hbb(img, verbose=False)[0]
results_obb = model_obb(img, verbose=False)[0]

img_with_hbb = results_hbb.plot(line_width=2, font_size=1) 
img_with_obb = results_obb.plot(line_width=2, font_size=1)

cv2.putText(img_with_hbb, "Standard YOLO (HBB)", (20, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
cv2.putText(img_with_obb, "YOLO-OBB (Oriented)", (20, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

combined_img = np.hstack((img_with_hbb, img_with_obb))

OUTPUT_PATH = "Figure2_Comparison.jpg"
cv2.imwrite(OUTPUT_PATH, combined_img)
