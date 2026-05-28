import os
import shutil
import cv2

GT_FILE = "data/OCR data/rec_gt_test.txt"
SRC_IMG_DIR = "data/OCR data/cropped_plates" 

OUT_IMG_DIR = "data/OCR data/bien_1_dong_folder"
OUT_GT_FILE = "data/OCR data/bien_1_dong_label.txt"

os.makedirs(OUT_IMG_DIR, exist_ok=True)

count_1_dong = 0
loi_khong_thay = 0

print("Image scanning...")

with open(GT_FILE, 'r', encoding='utf-8') as f_in, \
     open(OUT_GT_FILE, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        parts = line.strip().split('\t') if '\t' in line else line.strip().split(' ', 1)
        if len(parts) < 2: continue
        
        img_name = os.path.basename(parts[0])
        src_img_path = os.path.join(SRC_IMG_DIR, img_name)
        
        
        if not os.path.exists(src_img_path):
            loi_khong_thay += 1
            if loi_khong_thay <= 5:
                print(f"Not found error: {src_img_path}")
            continue
            
        img = cv2.imread(src_img_path)
        if img is not None:
            h, w = img.shape[:2]
            if h > 0 and (w / h) > 2.2:
                shutil.copy(src_img_path, os.path.join(OUT_IMG_DIR, img_name))
                new_line_path = f"data/OCR data/bien_1_dong_folder/{img_name}"
                f_out.write(f"{new_line_path}\t{parts[1]}\n")
                count_1_dong += 1

print("="*40)
print(f"Found: {count_1_dong} single-line plates.")
print(f"Number of images with incorrect names/not found: {loi_khong_thay}")
print("="*40)