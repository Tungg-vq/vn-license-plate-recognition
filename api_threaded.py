import cv2
import threading
import time
import uvicorn
import datetime
import shutil
import numpy as np
from collections import deque, Counter
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from final_inference import PlateRecognitionPipeline

app = FastAPI()

raw_frame = None
latest_plate_text = "Scanning..."
plate_buffer = deque(maxlen=5)
lock = threading.Lock()
source_lock = threading.Lock()
running = True
parking_db = {}
HOURLY_RATE = 10000

current_source = 0
reload_source = False
ai_awake_until = 0.0

YOLO_MODEL = "result/training/Medium with data augmentation with OBB data/weights/best.pt"
OCR_DIR = "./inference/model_chot_ha"
pipeline = PlateRecognitionPipeline(YOLO_MODEL, OCR_DIR, use_gpu=False)

class ActionRequest(BaseModel):
    plate: str
    action: str

def video_capture_loop():
    global raw_frame, running, current_source, reload_source, ai_awake_until
    camera = cv2.VideoCapture(current_source)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
    
    while running:
        with source_lock:
            if reload_source:
                camera.release()
                camera = cv2.VideoCapture(current_source)
                bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
                plate_buffer.clear()
                reload_source = False
                
        fps = camera.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps > 0 else 0.033
        
        success, frame = camera.read()
        
        if not success:
            if isinstance(current_source, str):
                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = camera.read()
                if not success:
                    time.sleep(0.1)
                    continue
            else:
                time.sleep(0.1)
                continue
                
        if success:
            resized_mask = cv2.resize(frame, (320, 240))
            fg_mask = bg_subtractor.apply(resized_mask)
            motion_score = cv2.countNonZero(fg_mask)
            
            with lock:
                raw_frame = frame.copy()
                if motion_score > 1500:
                    ai_awake_until = time.time() + 3.0
                    
        if isinstance(current_source, str):
            time.sleep(delay)
        else:
            time.sleep(0.01)
            
    camera.release()

def ai_processing_loop():
    global raw_frame, latest_plate_text, running, plate_buffer, ai_awake_until
    while running:
        current_time = time.time()
        is_awake = False
        
        with lock:
            is_awake = current_time < ai_awake_until
            
        if is_awake:
            img_to_process = None
            with lock:
                if raw_frame is not None:
                    img_to_process = raw_frame.copy()

            if img_to_process is not None:
                _, plate_text = pipeline.process_image(img_to_process)
                
                with lock:
                    if plate_text not in ["No plate found", "Không tìm thấy biển số", ""]:
                        plate_buffer.append(plate_text)
                    else:
                        plate_buffer.append("No plate found")
                        
                    if len(plate_buffer) > 0:
                        counter = Counter(plate_buffer)
                        best_plate, count = counter.most_common(1)[0]
                        
                        if count >= 3:
                            latest_plate_text = best_plate
            time.sleep(0.3)
        else:
            with lock:
                latest_plate_text = "💤 AI Sleeping (No Motion)"
                plate_buffer.clear()
            time.sleep(0.5)

def generate_live_stream():
    global raw_frame, latest_plate_text, running, ai_awake_until
    
    placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder_frame, "NO SIGNAL / LOADING...", (120, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
    while running:
        display_frame = None
        current_text = ""
        is_awake = False
        
        with lock:
            if raw_frame is not None:
                display_frame = raw_frame.copy()
            current_text = latest_plate_text
            is_awake = time.time() < ai_awake_until

        if display_frame is None:
            display_frame = placeholder_frame.copy()

        cv2.rectangle(display_frame, (0, 0), (640, 50), (0, 0, 0), -1)
        
        status_text = "AI: AWAKE" if is_awake else "AI: STANDBY"
        status_color = (0, 255, 0) if is_awake else (0, 165, 255)
        cv2.putText(display_frame, status_text, (500, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        cv2.putText(display_frame, f"GATE: {current_text}", (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', display_frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        time.sleep(0.03)

@app.on_event("startup")
def start_threads():
    threading.Thread(target=video_capture_loop, daemon=True).start()
    threading.Thread(target=ai_processing_loop, daemon=True).start()

@app.on_event("shutdown")
def stop_threads():
    global running
    running = False

@app.get("/plate_data")
def get_plate_data():
    with lock:
        return {"plate": latest_plate_text}

@app.post("/action")
def handle_action(req: ActionRequest):
    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    if req.action == "check_in":
        parking_db[req.plate] = current_time
        return {"message": f"[{time_str}] ✅ IN: {req.plate}"}
    
    elif req.action == "check_out":
        if req.plate in parking_db:
            entry_time = parking_db.pop(req.plate)
            duration = (current_time - entry_time).total_seconds()
            hours = max(0.5, round(duration / 3600.0, 2))
            
            if duration < 10: 
                hours = 2.5 
                
            fee = int(hours * HOURLY_RATE)
            return {"message": f"[{time_str}] 🛑 OUT: {req.plate} | Fee: {fee:,} VND"}
        else:
            return {"message": f"⚠️ Error: {req.plate} not found in parking lot!"}

@app.post("/upload_video")
def upload_video(file: UploadFile = File(...)):
    global current_source, reload_source
    file_location = "demo_video.mp4"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with source_lock:
        current_source = file_location
        reload_source = True

    return {"message": "Switched feed to Uploaded Video"}

@app.post("/use_webcam")
def use_webcam():
    global current_source, reload_source
    with source_lock:
        current_source = 0
        reload_source = True
    return {"message": "Switched feed back to Live Webcam"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_live_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Parking Guard Console</title>
        <style>
            body { text-align: center; font-family: 'Segoe UI', Arial, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding-top: 30px; }
            h1 { color: #00E676; margin-bottom: 5px; }
            p { color: #888; margin-top: 0; font-size: 14px; }
            .container { display: flex; justify-content: center; align-items: flex-start; gap: 20px; margin-top: 20px; }
            .video-section { display: flex; flex-direction: column; align-items: center; }
            .video-wrapper { border: 4px solid #333; border-radius: 12px; width: 640px; height: 480px; box-shadow: 0 8px 24px rgba(0,0,0,0.7); background-color: #000; overflow: hidden; display: flex; justify-content: center; align-items: center; }
            img { max-width: 100%; max-height: 100%; object-fit: contain; }
            .source-controls { margin-top: 15px; background: #1e1e1e; padding: 15px; border-radius: 12px; border: 1px solid #333; width: 610px; display: flex; justify-content: space-between; align-items: center; }
            .panel { background: #1e1e1e; padding: 20px; border-radius: 12px; width: 320px; border: 1px solid #333; text-align: left; }
            .plate-box { font-size: 24px; font-weight: bold; color: #FFD700; background: #000; padding: 15px; border-radius: 6px; text-align: center; margin-bottom: 20px; letter-spacing: 1px;}
            button { padding: 10px 15px; border: none; border-radius: 6px; font-size: 14px; font-weight: bold; cursor: pointer; transition: 0.2s; }
            .btn-large { width: 100%; padding: 15px; margin-bottom: 15px; font-size: 16px; }
            .btn-in { background-color: #4CAF50; color: white; }
            .btn-in:hover { background-color: #45a049; }
            .btn-out { background-color: #f44336; color: white; }
            .btn-out:hover { background-color: #da190b; }
            .btn-blue { background-color: #2196F3; color: white; }
            .btn-blue:hover { background-color: #0b7dda; }
            .btn-orange { background-color: #ff9800; color: white; }
            .btn-orange:hover { background-color: #e68a00; }
            #log-box { margin-top: 20px; font-size: 14px; color: #aaa; background: #000; padding: 10px; border-radius: 6px; min-height: 150px; max-height: 250px; overflow-y: auto;}
            input[type=file] { color: #e0e0e0; }
        </style>
    </head>
    <body>
        <h1>🚗 Smart Gate Guard Console</h1>
        <p>Motion-Activated AI Core Active</p>
        <div class="container">
            <div class="video-section">
                <div class="video-wrapper" id="video-wrapper">
                    <img id="video-display" src="/video_feed">
                </div>
                <div class="source-controls">
                    <div>
                        <input type="file" id="video-file" accept="video/*">
                        <button class="btn-blue" onclick="uploadVideo()">🎬 Play Video</button>
                    </div>
                    <button class="btn-orange" onclick="useWebcam()">📸 Webcam</button>
                </div>
            </div>
            <div class="panel">
                <h3>Current Scanned Plate:</h3>
                <div class="plate-box" id="plate-display">Waiting...</div>
                <button class="btn-large btn-in" onclick="confirmAction('check_in')">✅ CONFIRM CHECK-IN</button>
                <button class="btn-large btn-out" onclick="confirmAction('check_out')">🛑 CONFIRM CHECK-OUT</button>
                
                <h3 style="margin-top: 25px;">System Logs:</h3>
                <div id="log-box"></div>
            </div>
        </div>
        <script>
            setInterval(() => {
                fetch('/plate_data').then(r => r.json()).then(d => {
                    document.getElementById('plate-display').innerText = d.plate;
                });
            }, 500);

            function confirmAction(act) {
                let plate = document.getElementById('plate-display').innerText;
                if (plate.includes("Sleeping") || plate.includes("No plate found") || plate.includes("Waiting")) {
                    alert("No valid plate scanned yet!");
                    return;
                }
                fetch('/action', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({plate: plate, action: act})
                }).then(r => r.json()).then(d => {
                    logMessage(d.message);
                });
            }

            function forceReloadStream() {
                let wrapper = document.getElementById('video-wrapper');
                // Xóa thẻ img cũ và nhét thẻ img mới tinh vào
                wrapper.innerHTML = "<img id='video-display' src='/video_feed?rnd=" + Math.random() + "'>";
            }

            function uploadVideo() {
                let input = document.getElementById('video-file');
                if (input.files.length === 0) {
                    alert("Please select a video file first.");
                    return;
                }
                let formData = new FormData();
                formData.append("file", input.files[0]);
                
                // Tạm thời hiển thị chữ loading
                document.getElementById('video-wrapper').innerHTML = "<h3 style='color:red;'>LOADING VIDEO...</h3>";
                
                fetch('/upload_video', {
                    method: 'POST',
                    body: formData
                }).then(r => r.json()).then(d => {
                    logMessage("<span style='color:#2196F3'>ℹ️ " + d.message + "</span>");
                    setTimeout(forceReloadStream, 500);
                });
            }

            function useWebcam() {
                document.getElementById('video-wrapper').innerHTML = "<h3 style='color:orange;'>CONNECTING WEBCAM...</h3>";
                fetch('/use_webcam', {method: 'POST'}).then(r => r.json()).then(d => {
                    logMessage("<span style='color:#ff9800'>ℹ️ " + d.message + "</span>");
                    setTimeout(forceReloadStream, 500);
                });
            }

            function logMessage(msg) {
                let logBox = document.getElementById('log-box');
                logBox.innerHTML = "<div style='margin-bottom:8px;'>" + msg + "</div>" + logBox.innerHTML;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)