import os
import sys
import cv2
import re
import difflib
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.abspath(r"./PaddleOCR"))

try:
    import tools.infer.utility as utility
    from tools.infer.predict_rec import TextRecognizer
except ImportError:
    print("Error: Please run this script outside the PaddleOCR directory!")
    sys.exit()

def apply_plate_rules_v2(text):
    text = re.sub(r'[^A-Z0-9]', '', text)
    if len(text) < 5: return text
    
    text = re.sub(r'^[A-Z]+(\d{2})', r'\1', text)
    if len(text) >= 9 and re.match(r'^1\d{2}[A-Z]', text):
        text = text[1:]
    text = re.sub(r'^(\d{2})0([A-Z])', r'\1\2', text)

    chars = list(text)
    if len(chars) > 2:
        char_map = {'8':'B', '2':'Z', '6':'G', '0':'D', '5':'S', '1':'Y', '7':'T'}
        if chars[2].isdigit() and chars[2] in char_map:
            chars[2] = char_map[chars[2]]
    text = "".join(chars)

    match = re.match(r'^(\d{2}[A-Z]\d{0,1})(.*)$', text)
    if match:
        head = match.group(1)
        tail = match.group(2)
        
        num_map = {'Q':'0', 'O':'0', 'D':'0', 'U':'0', 'I':'1', 'T':'1', 'Z':'2', 'A':'4', 'S':'5', 'B':'8', 'G':'6'}
        tail_chars = list(tail)
        for i in range(len(tail_chars)):
            if tail_chars[i] in num_map:
                tail_chars[i] = num_map[tail_chars[i]]
            elif tail_chars[i].isalpha():
                tail_chars[i] = '0'
                
        tail = "".join(tail_chars)
        full_len = len(head) + len(tail)
        if (len(head) == 4 and full_len > 9) or (len(head) == 3 and full_len > 8):
            if tail.startswith('0') or tail.startswith('1') or tail.startswith('2'):
                tail = tail[1:]
        text = head + tail
    return text

def init_recognizer(model_dir, use_gpu=False):
    sys.argv = ['']
    args = utility.parse_args()
    args.rec_model_dir = model_dir
    args.rec_char_dict_path = "PaddleOCR/ppocr/utils/en_dict.txt"
    args.use_gpu = use_gpu
    args.rec_algorithm = "CRNN" 
    args.rec_image_shape = "3, 48, 320"
    return TextRecognizer(args)

def predict_license_plate(recognizer, img_list):
    raw_text = ""
    for img in img_list:
        rec_res, _ = recognizer([img])
        if rec_res and isinstance(rec_res, list) and len(rec_res) > 0:
            first_res = rec_res[0]
            if isinstance(first_res, (list, tuple)) and len(first_res) > 0:
                raw_text += str(first_res[0])
                
    raw_text = raw_text.replace('-', '')
    return apply_plate_rules_v2(raw_text)

def calculate_char_accuracy(pred, truth):
    if len(truth) == 0:
        return 0.0
    return difflib.SequenceMatcher(None, pred, truth).ratio()

def evaluate_isolated_ocr():
    OCR_DIR = "./inference/model_phase4"     
    VAL_IMAGES_DIR = "data/eval_chot_ha/" 
    GROUND_TRUTH_FILE = "data/eval_chot_ha_label.txt" 
    
    print("Initializing TextRecognizer module...")
    recognizer = init_recognizer(OCR_DIR, use_gpu=False)
    
    total_images = 0
    exact_matches = 0
    total_char_accuracy = 0.0

    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"Error: File not found: {GROUND_TRUTH_FILE}")
        return

    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"\nStarting evaluation on {len(lines)} validation images...")
    print("-" * 70)
    
    for line in tqdm(lines):
        line = line.strip()
        if not line: 
            continue
        
        parts = line.split('\t')
        if len(parts) < 2:
            parts = line.split(' ', 1)
            if len(parts) < 2:
                continue

        raw_path = parts[0]
        true_text = parts[1].replace("-", "").replace(".", "").upper().strip() 
        
        img_filename = os.path.basename(raw_path)
        img_path = os.path.join(VAL_IMAGES_DIR, img_filename)
        
        if not os.path.exists(img_path):
            img_path = raw_path
            if not os.path.exists(img_path):
                continue
            
        img = cv2.imread(img_path)
        if img is None: 
            continue
        
        total_images += 1
        
        try:
            pred_text = predict_license_plate(recognizer, [img])
            pred_text = pred_text.replace("-", "").replace(".", "").upper().strip()
            
            is_match = (pred_text == true_text)
            if is_match:
                exact_matches += 1
            
            char_acc = calculate_char_accuracy(pred_text, true_text)
            total_char_accuracy += char_acc
            
        except Exception as e:
            print(f"\n[ERROR] | Error processing {img_filename}: {e}")
            
    final_word_acc = (exact_matches / total_images) * 100 if total_images > 0 else 0
    final_char_acc = (total_char_accuracy / total_images) * 100 if total_images > 0 else 0
    
    print("\n" + "=" * 50)
    print(" ISOLATED OCR VALIDATION RESULTS")
    print("=" * 50)
    print(f" Total images evaluated : {total_images}")
    print(f" System (Word) Accuracy : {final_word_acc:.2f} %")
    print(f" Character Accuracy     : {final_char_acc:.2f} %")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_isolated_ocr()