import streamlit as st
from PIL import Image
import numpy as np
import datetime
import pandas as pd
import cv2

from final_inference import PlateRecognitionPipeline

st.set_page_config(page_title="Smart Parking Kiosk", page_icon="🚗", layout="wide")

HOURLY_RATE = 10000

if 'parking_db' not in st.session_state:
    st.session_state.parking_db = pd.DataFrame(columns=[
        "License Plate", "Entry Time", "Exit Time", "Duration (Hours)", "Fee (VND)", "Status"
    ])

if 'processed_image_ids' not in st.session_state:
    st.session_state.processed_image_ids = set()

if 'scan_result' not in st.session_state:
    st.session_state.scan_result = None

@st.cache_resource
def load_ai_model():
    YOLO_MODEL = "result/training/Medium with data augmentation with OBB data/weights/best.pt"
    OCR_DIR = "./inference/model_chot_ha"
    return PlateRecognitionPipeline(YOLO_MODEL, OCR_DIR, use_gpu=False)

pipeline = load_ai_model()

def scan_license_plate(image):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result_img, plate_text = pipeline.process_image(opencv_image)
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    return result_img_rgb, plate_text

st.title("🚗 Smart Parking Management Kiosk")
st.markdown("Automated vehicle Entry/Exit control with dynamic billing logic.")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Gate Camera Scan")
    
    input_method = st.radio("Select Input Method:", ("File Upload", "Camera"))
    image_file = st.file_uploader("Upload Vehicle Image", type=['jpg', 'jpeg', 'png']) if input_method == "File Upload" else st.camera_input("Take a picture")

    if image_file is not None:
        img = Image.open(image_file)
        file_id = image_file.file_id

        if file_id not in st.session_state.processed_image_ids:
            with st.spinner('Auto-processing Image...'):
                processed_img, plate_text = scan_license_plate(img)
                st.session_state.processed_image_ids.add(file_id)

                result_data = {
                    "img": processed_img,
                    "plate": plate_text,
                    "type": "error",
                    "fee": 0,
                    "duration": 0,
                    "time": ""
                }

                if plate_text not in ["No plate found", "Không tìm thấy biển số", ""]:
                    current_time = datetime.datetime.now()
                    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    result_data["time"] = current_time_str

                    df = st.session_state.parking_db
                    active_session = df[(df["License Plate"] == plate_text) & (df["Status"] == "Parked")]

                    if not active_session.empty:
                        idx = active_session.index[0]
                        entry_time_str = df.loc[idx, "Entry Time"]
                        entry_time = datetime.datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
                        
                        duration_seconds = (current_time - entry_time).total_seconds()
                        duration_hours = max(0.5, round(duration_seconds / 3600.0, 2))
                        
                        if duration_seconds < 10: 
                            duration_hours = 2.5
                            
                        fee = int(duration_hours * HOURLY_RATE)
                        
                        df.loc[idx, "Exit Time"] = current_time_str
                        df.loc[idx, "Duration (Hours)"] = duration_hours
                        df.loc[idx, "Fee (VND)"] = f"{fee:,}"
                        df.loc[idx, "Status"] = "Checked Out"
                        
                        st.session_state.parking_db = df
                        result_data["type"] = "checkout"
                        result_data["fee"] = fee
                        result_data["duration"] = duration_hours
                        
                    else:
                        new_record = pd.DataFrame([{
                            "License Plate": plate_text,
                            "Entry Time": current_time_str,
                            "Exit Time": "-",
                            "Duration (Hours)": "-",
                            "Fee (VND)": "-",
                            "Status": "Parked"
                        }])
                        st.session_state.parking_db = pd.concat([df, new_record], ignore_index=True)
                        result_data["type"] = "checkin"

                st.session_state.scan_result = result_data

        if st.session_state.scan_result:
            res = st.session_state.scan_result
            if res["type"] == "error":
                st.image(img, caption="Gate Capture", use_container_width=True)
                st.error("Could not recognize any license plate. Please try again.")
            else:
                st.image(res["img"], caption="Processed Image", use_container_width=True)
                if res["type"] == "checkout":
                    st.warning("🔔 VEHICLE CHECK-OUT DETECTED!")
                    st.metric(label="Plate", value=res["plate"])
                    st.metric(label="Total Fee", value=f"{res['fee']:,} VND", delta=f"{res['duration']} hours parked")
                elif res["type"] == "checkin":
                    st.success("✅ VEHICLE CHECK-IN SUCCESSFUL!")
                    st.metric(label="Registered Plate", value=res["plate"])
                    st.info(f"Entry logged at: {res['time']}")

with col2:
    st.subheader("🗄️ Parking Management Database")
    
    view_mode = st.selectbox("Filter View:", ["All Records", "Currently Parked", "History Logs"])
    
    df_display = st.session_state.parking_db
    if view_mode == "Currently Parked":
        df_display = df_display[df_display["Status"] == "Parked"]
    elif view_mode == "History Logs":
        df_display = df_display[df_display["Status"] == "Checked Out"]
        
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    if not st.session_state.parking_db.empty:
        csv = st.session_state.parking_db.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Data to CSV",
            data=csv,
            file_name='parking_logs.csv',
            mime='text/csv',
        )