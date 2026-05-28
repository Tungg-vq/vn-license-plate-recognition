import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


NUM_IMAGES = 2000  
OUT_DIR = "data/synth_line1"
OUT_TXT = "data/synth_line1_label.txt"

FONT_PATH = "C:\\Windows\\Fonts\\arialbd.ttf" 

os.makedirs(OUT_DIR, exist_ok=True)


def gen_random_text():
    tinh = str(random.randint(11, 99))
    chu = random.choice('ABCDEFGHJKLMNPRSTUVWXYZ')
    so = str(random.randint(1, 9))
  
    if random.random() < 0.7:
        return f"{tinh}-{chu}{so}"
    else:
        return f"{tinh}{chu}{so}"

def adjust_brightness_contrast(img):
    alpha = random.uniform(0.5, 1.5) 
    beta = random.randint(-50, 50)   
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def add_noise(img):
    row, col, ch = img.shape
    gauss = np.random.randn(row, col, ch) * random.randint(10, 30)
    noisy = np.clip(img + gauss, 0, 255).astype(np.uint8)
    return noisy

def add_blur(img):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)

print(f"Generating {NUM_IMAGES} images...")

count = 0
with open(OUT_TXT, 'w', encoding='utf-8', newline='\n') as f_out:
    for i in range(NUM_IMAGES):
        text = gen_random_text()
        
        
        img_w, img_h = 160, 60
        img_pil = Image.new('RGB', (img_w, img_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(img_pil)
        
        
        try:
            
            font_size = random.randint(35, 45)
            font = ImageFont.truetype(FONT_PATH, font_size)
        except:
            print("Lỗi: Không tìm thấy font! Sếp check lại đường dẫn FONT_PATH nhé.")
            break
            
  
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (img_w - text_w) / 2
        y = (img_h - text_h) / 2 - 5
        
        draw.text((x, y), text, font=font, fill=(0, 0, 0)) # Chữ đen
        
       
        img_cv = np.array(img_pil)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
       
        if random.random() < 0.8: img_cv = adjust_brightness_contrast(img_cv)
        if random.random() < 0.4: img_cv = add_blur(img_cv)
        if random.random() < 0.4: img_cv = add_noise(img_cv)
        
     
        img_name = f"synth_{i}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, img_name), img_cv)
        
       
        f_out.write(f"data/synth_line1/{img_name}\t{text}\n")
        count += 1

print(f"Printed {count} images at {OUT_DIR}!")
print(f"Label: {OUT_TXT}")