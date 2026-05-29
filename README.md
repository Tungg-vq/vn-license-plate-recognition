# 🚗 Vietnamese License Plate Recognition (End-to-End)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12.0-ee4c2c?logo=pytorch)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.5.0-orange)
![YOLO](https://img.shields.io/badge/YOLO-OBB-yellow)


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
├── data/                        # Dataset, txt labels, and test inference images
├── evaluation/                  # Evaluation scripts and accuracy logs
├── inference/                   # Exported OCR model for deployment
├── output/                      # Training logs and checkpoints for PaddleOCR
├── paddle_ocr_pretrained_model/ # Base pre-trained model for PaddleOCR
├── PaddleOCR/                   # Core PaddleOCR library
├── result/                      # YOLO-OBB training results and weights (best.pt)
├── runs/                        # YOLO-OBB inference/export logs
├── src/                         # Tooling: Data processing, cleaning, and augmentation
├── .gitignore                   # Git configuration to ignore large/unnecessary files
├── final_inference.py           # Main pipeline script (End-to-End inference)
└── requirements.txt
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

**3. Download Pre-trained Weights & Data**
Due to GitHub's file size limits, the trained model weights and raw datasets are hosted externally on Google Drive.

* 📦 [Download Project Data & Weights Here](https://drive.google.com/drive/folders/1lXlYaD_IJtzu3wlFQ7e-kdiv7yw0E9f8?usp=drive_link)

*Instructions:*
1. Download `result.zip`, `inference.zip`, and `data (1).zip` from the Drive link.
2. Extract the files.
3. Place the extracted `result/`, `inference/`, and `data/` folders directly into the root directory of this project.

## 🚀 Usage

To run the end-to-end pipeline on a sample image:

```bash
python final_inference.py
```

## 🚀 Deployment & Usage

To launch the Smart Gate Guard Console, start the ASGI server using the following command:

```bash
python api_threaded.py
```

Once the server indicates it has started successfully, open your web browser and navigate to:
**👉 `http://localhost:8000`**

### UI Controls:
- **Play Video:** Upload a local `.mp4` CCTV footage file to simulate real-world gate traffic.
- **Webcam:** Switch the input feed to your live local camera.
- **Confirm Check-In / Check-Out:** Manually trigger the state machine to log the vehicle and calculate parking fees based on duration.
