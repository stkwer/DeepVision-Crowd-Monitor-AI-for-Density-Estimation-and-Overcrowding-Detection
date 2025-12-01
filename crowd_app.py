import streamlit as st
import cv2
from ultralytics import YOLO
import torch
import tempfile
import time

# ---------------------------------------------------------
# 🌈 SUPER ATTRACTIVE FRONTEND — Replace Your Frontend With This
# ---------------------------------------------------------
st.set_page_config(
    page_title="Crowd Detection System",
    page_icon="👥",
    layout="wide"
)

# Glassmorphism + Neon Theme
st.markdown("""
<style>

* {
    font-family: 'Poppins', sans-serif;
}

/* Background Gradient */
body {
    background: linear-gradient(120deg, #0f0f3d, #1b1b4d, #271144);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
}
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Sidebar */
.css-1d391kg, .sidebar .sidebar-content {
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0px 0px 15px rgba(0,255,255,0.2);
}

/* Title */
h1 {
    color: #00eaff !important;
    text-shadow: 0px 0px 15px #00eaff;
}

/* Subtitle */
.sub-text {
    color: #c5e7ff;
    font-size: 20px;
    margin-top: -10px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00f2fe, #4facfe);
    color: #000;
    padding: 0.8rem 1.6rem;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 700;
    border: none;
    transition: 0.25s;
}
.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0px 0px 20px #00eaff;
}

/* People Counter Box */
.counter-box {
    background: rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    color: #ffffff;
    box-shadow: 0px 0px 25px rgba(0,255,255,0.15);
    animation: popIn 0.8s ease-in-out;
}
@keyframes popIn {
    0% {transform: scale(0.7); opacity: 0;}
    100% {transform: scale(1); opacity: 1;}
}

.video-frame {
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0px 0px 18px rgba(0,255,255,0.25);
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# Animated Heading
st.markdown("""
<div style="text-align:center; padding:10px 0;">
    <h1>👥 Real-Time Crowd Detection System</h1>
    <p class="sub-text">AI Powered Monitoring • Real-time Alerts • Smart Vision System</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 🎛️ SIDEBAR UI
# ---------------------------------------------------------
st.sidebar.header("⚙️ Control Panel")
crowd_limit = st.sidebar.slider("Set Crowd Limit", 1, 100, 10)
use_webcam = st.sidebar.toggle("Use Webcam")
alarm_enabled = st.sidebar.toggle("Enable Alarm")

video_file = None
if not use_webcam:
    video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

start_button = st.sidebar.button("▶️ Start Detection")


# ---------------------------------------------------------
# MODEL LOAD
# ---------------------------------------------------------
model = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

alarm_triggered = False

def play_alarm_once():
    st.warning("🚨 Overcrowded! Alarm Triggered!")
    st.audio("https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg")


# ---------------------------------------------------------
# MAIN SCREEN LAYOUT
# ---------------------------------------------------------
col_left, col_right = st.columns([2.5, 1])

with col_right:
    live_count_box = st.empty()

with col_left:
    stframe = st.empty()


# ---------------------------------------------------------
# 🔍 DETECTION LOGIC (your same logic)
# ---------------------------------------------------------
if start_button:
    if use_webcam:
        cap = cv2.VideoCapture(0)
        st.info("📹 Using Webcam…")
    else:
        if video_file is None:
            st.error("Please upload a video file!")
            st.stop()
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(video_file.read())
        cap = cv2.VideoCapture(temp.name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        boxes = results[0].boxes.xyxy
        crowd_count = len(boxes)

        # Draw
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,200), 2)
            cv2.putText(frame, "Person", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Counter Box
        live_count_box.markdown(
            f"<div class='counter-box'>👥 People Detected:<br><span style='font-size:45px'>{crowd_count}</span></div>",
            unsafe_allow_html=True
        )

        # Alarm Logic
        if alarm_enabled:
            if crowd_count > crowd_limit and not alarm_triggered:
                play_alarm_once()
                alarm_triggered = True
            elif crowd_count <= crowd_limit:
                alarm_triggered = False

        # Display Frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        time.sleep(0.04)

    cap.release()
    st.success("✔️ Detection Finished!")
