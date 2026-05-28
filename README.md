# 🚗 Vietnamese License Plate Recognition (End-to-End)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12.0-ee4c2c?logo=pytorch)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.5.0-orange)
![YOLO](https://img.shields.io/badge/YOLO-OBB-yellow)

> *"The best way to predict the future is to create it."*

An end-to-end Computer Vision pipeline designed to accurately detect and recognize Vietnamese vehicle license plates (both 1-line and 2-line formats) under various real-world conditions.

## 🌟 Key Features
- **OBB Detection:** Utilizes YOLO (Oriented Bounding Box) to accurately crop angled/tilted plates.
- **Smart Formatting:** Automatically classifies and reconstructs 2-line plates (motorcycles) into a single logical string before reading.
- **Custom OCR Engine:** Fine-tuned **PaddleOCR** model optimized specifically for Vietnamese license plate fonts and characters.
- **Automated Data Pipeline:** Built-in scripts for data cleaning, deduplication, and preventing train/val leakage.

## 🏗️ Architecture & Pipeline
1. **Input:** Raw vehicle image.
2. **Detection:** YOLO-OBB isolates the license plate and applies perspective transformation.
3. **Processing:** Determines if the plate is 1-line (car) or 2-line (motorcycle). Splits and stitches lines if necessary.
4. **Recognition:** PaddleOCR extracts the text.
5. **Output:** Cleaned text via Regex filtering.

## 📁 Project Structure
```text
CAR_PLATE_DETECTION/
├── data/                # Dataset, txt labels, and test inference images
├── docs/                # Training guides and fix logs
├── evaluation/          # Evaluation scripts and accuracy logs
├── inference/           # Exported OCR model for deployment
├── PaddleOCR/           # Core PaddleOCR library
├── src/                 # Tooling: Data processing, cleaning, and augmentation
├── final_inference.py              # Main pipeline script (End-to-End inference)
└── requirements.txt     # Dependencies
```

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone [https://github.com/Tungg-vq/vn-license-plate-recognition.git](https://github.com/Tungg-vq/vn-license-plate-recognition.git)
cd vn-license-plate-recognition
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download Pre-trained Weights**
Due to GitHub's file size limits, the trained model weights are hosted externally.

[Download YOLO & OCR Weights here](ĐỂ_LINK_GOOGLE_DRIVE_CỦA_ÔNG_VÀO_ĐÂY)

Extract the downloaded file and place the `paddle_ocr_pretrained_model/` and `runs/` folders into the root directory.

## 🚀 Usage

To run the end-to-end pipeline on a sample image:

```bash
python final_inference.py
```
