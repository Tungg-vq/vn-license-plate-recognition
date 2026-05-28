import os
import json
import re

GT_FILE = "data/OCR data/rec_gt_test.txt"
PRED_FILE = "output/predicts.txt"

def apply_plate_rules(text):
    text = re.sub(r'[^A-Z0-9]', '', text)
    if len(text) < 5: return text
    
    # 1. CẮT RÁC Ở ĐẦU
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
if __name__ == '__main__':
    gt_dict = {}
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t') if '\t' in line else line.strip().split(' ', 1)
            if len(parts) >= 2:
                base_name = os.path.splitext(os.path.basename(parts[0]))[0]
                gt_dict[base_name] = parts[1]

    raw_preds = {}
    if not os.path.exists(PRED_FILE):
        print(f"Error: {PRED_FILE} not found!")
        exit()

    with open(PRED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_name = os.path.basename(parts[0])
                try:
                    data = json.loads(parts[1])
                    text = data.get('rec_text', '')
                except:
                    text = parts[1] 
                raw_preds[img_name] = text

    predictions = {}
    for img_name in sorted(raw_preds.keys()):
        clean_name = img_name.replace('_line1', '').replace('_line2', '')
        base_name = os.path.splitext(clean_name)[0]

        if base_name not in predictions:
            predictions[base_name] = ""
        predictions[base_name] += raw_preds[img_name]

    total = 0
    correct = 0

    print("="*50)
    print("FINAL EVALUATION REPORT")
    print("="*50)

    for base_name, true_label in gt_dict.items():
        if base_name in predictions:
            total += 1
            raw_pred = predictions[base_name]
            pred_label = apply_plate_rules(raw_pred)
            
            if pred_label == true_label:
                correct += 1
            else:
                print(f"MISMATCH: {base_name}")
                print(f"   - Ground Truth : {true_label}")
                print(f"   - Predicted    : {pred_label}")
                print("-" * 30)

    if total > 0:
        accuracy = (correct / total) * 100
        print("\n" + "="*50)
        print(" SUMMARY:")
        print(f"   - Total Plates Tested : {total}")
        print(f"   - Exact Matches       : {correct}")
        print(f"   - ACCURACY            : {accuracy:.2f}%")
        print("="*50)
    else:
        print("\nNo matching images found!")