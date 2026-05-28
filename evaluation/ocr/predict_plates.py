import os
from paddlex import create_model

model = create_model(
    model_name='en_PP-OCRv4_mobile_rec',
    model_dir='inference/model_chot_ha'
)

IMG_DIR = "data/OCR data/final_lines"

if __name__ == '__main__':
    images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Processing {len(images)} images...\n")

    for img_name in images:
        img_path = os.path.join(IMG_DIR, img_name)
        
        try:
            results = model.predict(img_path)
            for res in results:
                text = res['rec_text']
                conf = res['rec_score']
                print(f"{img_name} | Result: {text} | Conf: {conf:.2f}")
        except Exception as e:
            print(f"{img_name} | Error: {e}")