import os
import cv2
import difflib

from final_inference import PlateRecognitionPipeline

def calculate_char_accuracy(pred, truth):
    
    if len(truth) == 0:
        return 0.0
    return difflib.SequenceMatcher(None, pred, truth).ratio()

def evaluate_end_to_end():

    YOLO_MODEL = "result/training/Medium with data augmentation with OBB data/weights/best.pt"     
    OCR_DIR = "./inference/model_chot_ha"     
    TEST_IMAGES_DIR = "data/e2e_test_images" 
    GROUND_TRUTH_FILE = "data/e2e_labels.txt" 
    
    print("Initializing the entire Pipeline (YOLO OBB + OCR)...")
    pipeline = PlateRecognitionPipeline(YOLO_MODEL, OCR_DIR, use_gpu=False)
    
    total_images = 0
    exact_matches = 0
    total_char_accuracy = 0.0

   
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"File not found: {GROUND_TRUTH_FILE}")
        return

    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"\nnStarting on  {len(lines)} images...")
    print("-" * 60)
    print(f"{'STATUS':<8} | {'FILE NAME':<20} | {'TRUE':<12} | {'PREDICTED':<12} | {'CHAR ACC'}")
    print("-" * 60)
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        parts = line.split('\t')
        if len(parts) < 2:
            parts = line.split(' ', 1)
            if len(parts) < 2: continue

        img_name = parts[0]
        
       
        true_text = parts[1].replace("-", "").replace(".", "").upper() 
        
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        total_images += 1
        
        try:
           
            _, pred_text = pipeline.process_image(img) 
            
           
            if "not detected" in pred_text.lower():
                pred_text = ""
            else:
                pred_text = pred_text.replace("-", "").replace(".", "").upper()
            
          
            is_match = (pred_text == true_text)
            if is_match:
                exact_matches += 1
            
           
            char_acc = calculate_char_accuracy(pred_text, true_text)
            total_char_accuracy += char_acc
            
        
            status = "[OK]" if is_match else "[FAIL]"
            
            short_name = (img_name[:12] + '...') if len(img_name) > 15 else img_name
            
            print(f"{status:<8} | {short_name:<20} | {true_text:<12} | {pred_text:<12} | {char_acc*100:.1f}%")
            
        except Exception as e:
            print(f"[ERROR]  | {img_name[:20]:<20} | Pipeline processing error: {e}")
            
   
    final_e2e_acc = (exact_matches / total_images) * 100 if total_images > 0 else 0
    final_char_acc = (total_char_accuracy / total_images) * 100 if total_images > 0 else 0
    
    print("\n" + "=" * 50)
    print(" Result: ")
    print("=" * 50)
    print(f" Total images : {total_images}")
    print(f" System Accuracy     : {final_e2e_acc:.2f} %")
    print(f" Character Accuracy  : {final_char_acc:.2f} %")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_end_to_end()