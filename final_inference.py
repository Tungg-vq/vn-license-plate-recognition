import os
import sys
import cv2
import re
import numpy as np
from ultralytics import YOLO


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


def order_points(pts):
   
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


class PlateRecognitionPipeline:
    def __init__(self, yolo_path, ocr_dir, use_gpu=False):
        print("Loading YOLO OBB model...")
        self.detector = YOLO(yolo_path)
        
        print("Loading OCR model...")
        self.recognizer = init_recognizer(ocr_dir, use_gpu=use_gpu)

    def process_image(self, img_path):
        original_img = cv2.imread(img_path)
        if original_img is None:
            print("Imaged not read")
            return None
        
        display_img = original_img.copy()
        
        
        results = self.detector(original_img, verbose=False)
        
        
        if not hasattr(results[0], 'obb') or results[0].obb is None or len(results[0].obb) == 0:
            print("No plates found by YOLO")
            return display_img
            
        
        for obb in results[0].obb:
            
            pts = obb.xyxyxyxy[0].cpu().numpy()
            
           
            crop_img = four_point_transform(original_img, pts)
            
            h, w = crop_img.shape[:2]
           
            if w == 0 or h == 0: continue
            
            ratio = h / float(w)
            ocr_inputs = []
            
            
            if ratio > 0.65:
                mid_y = h // 2
                line1 = crop_img[:mid_y, :]
                line2 = crop_img[mid_y:, :]
                ocr_inputs = [line1, line2]
            else:
                ocr_inputs = [crop_img]
                
            
            final_text = predict_license_plate(self.recognizer, ocr_inputs)
            
           
            pts_int = np.int32(pts).reshape((-1, 1, 2))
            cv2.polylines(display_img, [pts_int], isClosed=True, color=(0, 255, 0), thickness=2)
            
            
            rect = order_points(pts)
            tl_x, tl_y = int(rect[0][0]), int(rect[0][1])
            
            
            cv2.rectangle(display_img, (tl_x, tl_y - 30), (tl_x + 180, tl_y), (0, 255, 0), -1)
            cv2.putText(display_img, final_text, (tl_x + 5, tl_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            print(f"Detected plate: {final_text}")
            
        return display_img

if __name__ == "__main__":
    
    YOLO_MODEL = "result/training/Medium with data augmentation with OBB data/weights/best.pt"     
    TEST_IMAGE = "data/inference data/Picture14.png" 
    OCR_DIR = "./inference/model_chot_ha"     
    
    if os.path.exists(YOLO_MODEL) and os.path.exists(TEST_IMAGE):
       
        pipeline = PlateRecognitionPipeline(YOLO_MODEL, OCR_DIR, use_gpu=False)
        
        
        result_img = pipeline.process_image(TEST_IMAGE)
        
       
        if result_img is not None:
            cv2.imshow("Final result", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Path error")