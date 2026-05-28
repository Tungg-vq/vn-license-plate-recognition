import os
GT_FILE = "data/OCR data/rec_gt_test.txt"
OUT_GT_FILE = "data/OCR data/bien_1_dong_label.txt"
FOLDER_1_DONG = "data/OCR data/bien_1_dong_folder"

nhan_goc = {}
with open(GT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        
      
        label = line.split()[-1] 
        
        raw_path = line.replace(label, '').strip()
        img_name = os.path.basename(raw_path)
        nhan_goc[img_name] = label
count = 0
with open(OUT_GT_FILE, 'w', encoding='utf-8', newline='\n') as f_out:
    for img_name in os.listdir(FOLDER_1_DONG):
        if img_name in nhan_goc:
            label = nhan_goc[img_name]
           
            new_path = f"data/OCR data/bien_1_dong_folder/{img_name}"
            f_out.write(f"{new_path}\t{label}\n")
            count += 1

print("="*50)
print(f"Done {count} images")
print("="*50)