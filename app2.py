import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import smtplib
import ssl
import time
import os
import tempfile
import av  # Required for webcam fix
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from ultralytics import YOLO


# --- Page Configuration ---
st.set_page_config(
    page_title="Crowd Counting Dashboard",
    page_icon="👥",
    layout="wide"
)

# --- Global Styles / Theming (UI Only) ---
st.markdown(
    """
    <style>
    :root {
        --primary-color: #4F46E5;
        --primary-soft: rgba(79, 70, 229, 0.16);
        --accent-color: #F97316;
        --accent-soft: rgba(248, 171, 110, 0.18);
        --teal-color: #22D3EE;
        --bg-deep: #020617;
        --bg-card: rgba(15, 23, 42, 0.96);
        --border-subtle: rgba(148, 163, 184, 0.45);
    }

    /* ---------- GLOBAL TYPOGRAPHY ---------- */
    html, body, .stApp {
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 16px;
    }

    h1 {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    h2 {
        font-size: 1.9rem;
        font-weight: 600;
    }

    h3 {
        font-size: 1.25rem;
        font-weight: 600;
    }

    p, label, span, li, div, input, textarea {
        font-size: 0.97rem;
    }

    /* ---------- APP BACKGROUND ---------- */
    .stApp {
        background: radial-gradient(circle at top left, #111827, #020617);
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 2.4rem;
        padding-bottom: 3.2rem;
        max-width: 1250px;
    }

    /* Main central panel so it doesn't look empty */
    .block-container > div:first-child {
        background: radial-gradient(circle at top left,
                    rgba(79, 70, 229, 0.32),
                    rgba(15, 23, 42, 0.98));
        border-radius: 1.8rem;
        padding: 1.9rem 2.2rem 2.4rem;
        box-shadow: 0 30px 80px rgba(15, 23, 42, 1);
        border: 1px solid rgba(129, 140, 248, 0.55);
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #020617 14%, #020617 40%, #020617 100%);
        border-right: 1px solid var(--border-subtle);
        box-shadow: 12px 0 40px rgba(15, 23, 42, 0.95);
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* Mode radio buttons styled as pills, new color combo */
    div[role="radiogroup"] > label {
        background: radial-gradient(circle at top left,
                        rgba(34, 211, 238, 0.20),
                        rgba(15, 23, 42, 0.96));
        border-radius: 999px;
        padding: 0.4rem 1.1rem;
        border: 1px solid rgba(56, 189, 248, 0.85);
        margin-bottom: 0.35rem;
        cursor: pointer;
    }

    div[role="radiogroup"] > label:hover {
        background: radial-gradient(circle at top left,
                        rgba(45, 212, 191, 0.35),
                        rgba(15, 23, 42, 0.98));
        border-color: rgba(45, 212, 191, 0.95);
    }

    div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0;
        font-weight: 500;
    }

    /* Inputs */
    input, textarea {
        background: rgba(15, 23, 42, 0.96) !important;
        border-radius: 0.8rem !important;
        border: 1px solid rgba(55, 65, 81, 0.9) !important;
    }

    input:focus, textarea:focus {
        border-color: var(--teal-color) !important;
        box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.65) !important;
    }

    /* ---------- NUMBER INPUT (+ / -) ---------- */
    div[data-testid="stNumberInput"] {
        margin-bottom: 0.4rem;
    }

    div[data-testid="stNumberInput"] button {
        background: #020617 !important;
        color: #f9fafb !important;
        border-radius: 0.6rem !important;
        border: 1px solid rgba(148, 163, 184, 0.9) !important;
        font-size: 1.1rem !important;
        width: 2.1rem;
        height: 2.1rem;
    }

    div[data-testid="stNumberInput"] button:hover {
        background: #22c55e !important;
        color: #020617 !important;
    }

    /* ---------- TOP MODE PILL ---------- */
    .mode-pill {
        border-radius: 999px;
        padding: 0.45rem 1.1rem;
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.7);
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.8rem;
        color: #e5e7eb;
        margin-bottom: 0.65rem;
    }

    .mode-pill span {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: #22c55e;
        box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.4);
    }

    /* ---------- METRICS (Crowd Limit, etc.) NEW COLORS ---------- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg,
                        rgba(14, 165, 233, 0.85),
                        rgba(129, 140, 248, 0.95));
        border-radius: 1.2rem;
        padding: 1.1rem 1.4rem;
        border: 1px solid rgba(191, 219, 254, 0.9);
        box-shadow: 0 20px 55px rgba(15, 23, 42, 1);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #e5e7eb;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.9rem;
        font-weight: 700;
        color: #fef9c3;
    }

    /* ---------- FILE UPLOADER (Browse text visible & vivid) ---------- */
    div[data-testid="stFileUploader"] {
        background: linear-gradient(135deg,
                        rgba(15, 23, 42, 0.98),
                        rgba(37, 99, 235, 0.96));
        border-radius: 1.3rem;
        padding: 1.2rem 1.3rem 1.05rem;
        border: 1px dashed rgba(191, 219, 254, 0.95);
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.95);
    }

    div[data-testid="stFileUploader"] * {
        color: #e5e7eb !important;
    }

    div[data-testid="stFileUploader"] span {
        opacity: 0.96 !important;
    }

    div[data-testid="stFileUploader"] small {
        color: #cbd5f5 !important;
        opacity: 0.9 !important;
    }

    /* Browse files button */
    div[data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #22c55e, #0ea5e9) !important;
        color: #0f172a !important;
        border-radius: 999px !important;
        padding: 0.35rem 0.9rem !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 10px 28px rgba(45, 212, 191, 0.55);
    }

    div[data-testid="stFileUploader"] button:hover {
        filter: brightness(1.05);
        box-shadow: 0 14px 38px rgba(45, 212, 191, 0.75);
    }

    /* ---------- IMAGES & VIDEO (bigger, stronger) ---------- */
    div[data-testid="stImage"] img {
        border-radius: 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.6);
        box-shadow: 0 20px 60px rgba(15, 23, 42, 1);
    }

    video {
        border-radius: 1.2rem;
        box-shadow: 0 20px 60px rgba(15, 23, 42, 1);
        width: 100% !important;
        height: auto !important;
    }

    /* ---------- BUTTONS ---------- */
    .stButton>button {
        border-radius: 999px;
        padding: 0.65rem 1.5rem;
        background: linear-gradient(135deg, var(--primary-color), #EC4899);
        border: none;
        color: #f9fafb;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 16px 40px rgba(79, 70, 229, 0.9);
    }

    .stButton>button:hover {
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 24px 60px rgba(79, 70, 229, 1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Model Definition (CSRNet) ---
class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M',
                              256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = self.make_layers(self.frontend_feat)
        self.backend = self.make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        layers = []
        dilation_rate = 2 if dilation else 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(
                    in_channels, v, kernel_size=3, padding=dilation_rate,
                    dilation=dilation_rate
                )
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


# --- Model Loading (Cached) ---
@st.cache_resource
def load_models(csrnet_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load CSRNet (for dense crowds)
    csrnet_model = CSRNet().to(device)
    csrnet_model.load_state_dict(torch.load(csrnet_path, map_location=device))
    csrnet_model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # 2. Load YOLOv8 (for sparse individuals)
    yolo_model = YOLO("yolov8n.pt")  # 'n' is the small, fast "nano" model

    return csrnet_model, yolo_model, device, transform


# --- Core Processing Function (FOR AI MODEL - CSRNet) ---
def process_frame_ai(model, device, transform, frame):
    # This function expects a BGR frame from cv2
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    img_tensor = transform(pil_img).to(device)
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        pred_density_map = model(img_tensor)

    # Force the minimum count to be 0
    pred_count = max(0, pred_density_map.sum().item())
    return pred_count


# --- Email Alert Function ---
def send_email_alert(sender, password, receiver, count, limit):
    subject = "CROWD ALERT - OVERCROWDING DETECTED"
    body = f"An overcrowding event was detected.\nEstimated Count: {count}\nLimit: {limit}"

    # Create the email message
    msg = f"Subject: {subject}\n\n{body}"

    try:
        # Connect to Gmail's secure SSL server
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender, password)  # Login with your email and 16-digit App Password
            server.sendmail(sender, receiver, msg)  # Send the email
        st.toast("Email alert sent successfully!", icon="✅")
    except Exception as e:
        st.error(f"Failed to send email: {e}")


# --- Helper to draw on image ---
def draw_on_image(image_np, count, limit):
    color = (0, 255, 0)  # Green
    alert_text = ""
    if count > limit:
        color = (0, 0, 255)  # Red
        alert_text = "!!! ALERT: OVERCROWDED !!!"

    # --- FONT SIZE INCREASED ---
    # Increased fontScale from 2 to 3, and thickness from 3 to 4
    cv2.putText(
        image_np,
        f"Count: {count}",
        (50, 80),  # Moved Y down a bit
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        color,
        4,
    )
    if alert_text:
        # Also increased alert font size
        cv2.putText(
            image_np,
            alert_text,
            (50, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 0, 255),
            4,
        )
    # -----------------------------
    return image_np


# --- Initialize Session State for Alerts ---
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = 0


# --- Load Model ---
MODEL_PATH = "best_model_finetuned.pth"
if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found! Make sure 'best_model_finetuned.pth' is in the same folder as app.py"
    )
    st.stop()

csrnet_model, yolo_model, device, transform = load_models(MODEL_PATH)
if csrnet_model is None:
    st.error("Failed to load models. App cannot start.")
    st.stop()

# --- Hardcode the threshold so it's hidden from the UI ---
HYBRID_THRESHOLD = 10


# --- Dashboard UI ---
st.markdown(
    """
    <div class="mode-pill">
        <span></span>
        <div>Crowd Intelligence Dashboard</div>
    </div>
    <div style="display:flex;align-items:flex-end;justify-content:space-between;margin-bottom:0.8rem;">
        <div>
            <h1 style="margin:0;">Crowd Counting &amp; Overcrowding Alert System</h1>
            <p style="margin:0.35rem 0 0;color:#d1d5db;font-size:0.98rem;">
                Real-time monitoring of crowd density by AB.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- THIS IS THE FIX (Part 1) ---
# Create one single, empty placeholder for the audio
# We will place this *inside* each tab
# alert_placeholder = st.empty() <-- DELETE THIS GLOBAL ONE


# --- Sidebar for Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    app_mode = st.radio("Choose Mode", ["Image", "Video", "Live Webcam"])

    st.divider()
    st.header("Alert Settings")
    CROWD_LIMIT = st.number_input("Crowd Limit Threshold", min_value=1, value=100)
    ALERT_COOLDOWN = st.number_input("Alert Cooldown (seconds)", min_value=10, value=300)

    st.divider()
    st.header("Email Alerts (Optional)")
    st.info("Requires a 16-digit Google App Password.")
    EMAIL_SENDER = st.text_input("Your Gmail Address")
    EMAIL_PASSWORD = st.text_input("Your Google App Password", type="password")
    EMAIL_RECEIVER = st.text_input("Email to Send Alerts To")

    email_enabled = EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER
    if email_enabled:
        st.success("Email alerts are enabled.")


# --- Function to check and send alerts ---
# --- THIS IS THE FIX (Part 2) ---
# The function now requires the placeholder as an argument
def check_and_send_alert(count, limit, alert_placeholder):
    if count > limit:
        current_time = time.time()
        if (current_time - st.session_state.last_alert_time) > ALERT_COOLDOWN:
            st.session_state.last_alert_time = current_time

            st.toast(f"🚨 ALERT! Count: {count}", icon="🚨")

            # --- SOUND ALERT UPDATED ---
            # We use the *specific placeholder* passed to this function
            if os.path.exists("alert.mp3"):
                alert_placeholder.audio("alert.mp3", autoplay=True)
            else:
                print("Alert sound file 'alert.mp3' not found.")
            # ---------------------------

            if email_enabled:
                send_email_alert(EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER, count, limit)

            return True  # Alert was sent
    return False


# --- Main App Logic ---


# --- IMAGE TAB ---
if app_mode == "Image":
    st.markdown(
        """
        <div style="margin-top:0.75rem;margin-bottom:0.75rem;">
            <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:.18em;color:#e5e7eb;margin-bottom:0.1rem;">
                Mode • Image
            </div>
            <h2 style="margin:0;color:#F9FAFB;">Process a Single Image</h2>
            <p style="margin:0.15rem 0 0;color:#E5E7EB;font-size:0.9rem;opacity:0.9;">
                Upload a frame to estimate the number of people in the scene.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.metric("Crowd Limit", CROWD_LIMIT)
    with info_col2:
        st.metric("Alert Cooldown (s)", ALERT_COOLDOWN)

    # --- THIS IS THE FIX (Part 3) ---
    # Create a *local* placeholder for this tab
    alert_placeholder = st.empty()

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        pil_image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(pil_image)

        image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        with st.spinner("Processing image..."):

            if image_np_bgr.shape[0] * image_np_bgr.shape[1] > 2000000:
                processing_frame = cv2.resize(image_np_bgr, (800, 600), interpolation=cv2.INTER_AREA)
            else:
                processing_frame = image_np_bgr.copy()

            # --- Hybrid Logic ---
            yolo_results = yolo_model(processing_frame, classes=[0], verbose=False)
            num_yolo_persons = (yolo_results[0].boxes.cls == 0).sum().item()

            if 0 < num_yolo_persons <= HYBRID_THRESHOLD:
                final_count = num_yolo_persons
            else:
                pred_count = process_frame_ai(csrnet_model, device, transform, processing_frame)
                final_count = round(pred_count)
            # --------------------

            result_image = draw_on_image(image_np_bgr.copy(), final_count, CROWD_LIMIT)

            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_image, caption="Original Image (Resized if > 2MP)", use_container_width=True)
            with col2:
                st.image(result_image_rgb, caption="Processed Image", use_container_width=True)

            st.divider()
            st.metric("Estimated Crowd Count", final_count)

            if final_count > CROWD_LIMIT:
                st.error(f"OVERCROWDING DETECTED! Count: {final_count} > Limit: {CROWD_LIMIT}")
                # --- THIS IS THE FIX (Part 4) ---
                # Pass the local placeholder to the function
                check_and_send_alert(final_count, CROWD_LIMIT, alert_placeholder)
            else:
                st.success(f"Crowd count is normal. Count: {final_count} <= Limit: {CROWD_LIMIT}")


# --- VIDEO TAB (Real-time playback) ---
elif app_mode == "Video":
    st.markdown(
        """
        <div style="margin-top:0.75rem;margin-bottom:0.75rem;">
            <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:.18em;color:#e5e7eb;margin-bottom:0.1rem;">
                Mode • Video
            </div>
            <h2 style="margin:0;color:#F9FAFB;">Process a Video File</h2>
            <p style="margin:0.15rem 0 0;color:#E5E7EB;font-size:0.9rem;opacity:0.9;">
                Stream through a saved video and visualize live crowd estimates.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.metric("Crowd Limit", CROWD_LIMIT)
    with info_col2:
        st.metric("Alert Cooldown (s)", ALERT_COOLDOWN)

    st.header("Process a Video File")
    # --- THIS IS THE FIX (Part 3) ---
    alert_placeholder = st.empty()

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if uploaded_video:

        video_path = os.path.join(".", "temp_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.info("Processing video... This will play in real-time and will not hang.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1 / fps if fps > 0 else 0.03

            FRAME_PROCESS_RATE = 5
            skip_interval = 1 if (fps == 0 or fps < FRAME_PROCESS_RATE) else int(fps // FRAME_PROCESS_RATE)

            st.subheader("Video Output")
            video_placeholder = st.empty()

            last_final_count = 0
            last_alert_text = ""
            last_color = (0, 255, 0)
            frame_counter = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_counter += 1

                try:
                    processing_frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_AREA)
                except Exception as e:
                    st.error(f"Error resizing frame: {e}")
                    break

                if frame_counter % skip_interval == 0:

                    # --- Hybrid Logic ---
                    yolo_results = yolo_model(processing_frame, classes=[0], verbose=False)
                    num_yolo_persons = (yolo_results[0].boxes.cls == 0).sum().item()

                    if 0 < num_yolo_persons <= HYBRID_THRESHOLD:
                        final_count = num_yolo_persons
                    else:
                        pred_count = process_frame_ai(csrnet_model, device, transform, processing_frame)
                        final_count = round(pred_count)
                    # ----------------------

                    last_final_count = final_count
                    last_color = (0, 255, 0)
                    last_alert_text = ""

                    # --- THIS IS THE FIX (Part 4) ---
                    if check_and_send_alert(final_count, CROWD_LIMIT, alert_placeholder):
                        last_alert_text = "!!! ALERT: OVERCROWDED !!!"
                        last_color = (0, 0, 255)

                # Draw on the *original* high-res frame
                frame_with_info = draw_on_image(frame, last_final_count, CROWD_LIMIT)

                frame_rgb = cv2.cvtColor(frame_with_info, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                time.sleep(frame_delay)

            cap.release()
            st.success("Video playback complete.")
            os.remove(video_path)


# --- WEBCAM TAB (This is the working version) ---
elif app_mode == "Live Webcam":
    st.markdown(
        """
        <div style="margin-top:0.75rem;margin-bottom:0.75rem;">
            <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:.18em;color:#e5e7eb;margin-bottom:0.1rem;">
                Mode • Live Webcam
            </div>
            <h2 style="margin:0;color:#F9FAFB;">Live Webcam Feed</h2>
            <p style="margin:0.15rem 0 0;color:#E5E7EB;font-size:0.9rem;opacity:0.9;">
                Use your camera for continuous, real-time crowd monitoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.metric("Crowd Limit", CROWD_LIMIT)
    with info_col2:
        st.metric("Alert Cooldown (s)", ALERT_COOLDOWN)

    st.header("Live Webcam Feed")
    # --- THIS IS THE FIX (Part 3) ---
    alert_placeholder = st.empty()

    st.info("Click 'START' to access your webcam. The model will process the feed in real-time.")

    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.frame_counter = 0
            self.skip_interval = 6  # Process 1 in 6 frames (approx 5fps)
            self.last_final_count = 0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            try:
                processing_frame = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(f"Error resizing webcam frame: {e}")
                return frame

            self.frame_counter += 1

            if self.frame_counter % self.skip_interval == 0:

                # --- Hybrid Logic ---
                yolo_results = yolo_model(processing_frame, classes=[0], verbose=False)
                num_yolo_persons = (yolo_results[0].boxes.cls == 0).sum().item()

                if 0 < num_yolo_persons <= HYBRID_THRESHOLD:
                    self.last_final_count = num_yolo_persons
                else:
                    pred_count = process_frame_ai(csrnet_model, device, transform, processing_frame)
                    self.last_final_count = round(pred_count)
                # ----------------------

                # --- THIS IS THE FIX (Part 4) ---
                # We are inside a class, so we can't access the placeholder
                # We will revert this ONE function to use st.html, as it's the only way.
                if self.last_final_count > CROWD_LIMIT:
                    current_time = time.time()
                    if (current_time - st.session_state.last_alert_time) > ALERT_COOLDOWN:
                        st.session_state.last_alert_time = current_time

                        # We can't use st.toast or st.audio from here,
                        # so we use st.html. This is the only way.
                        js_code = """
                        <script>
                            // 1. Play a beep sound
                            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                            if (audioContext.state === 'suspended') {
                                audioContext.resume();
                            }
                            const oscillator = audioContext.createOscillator();
                            oscillator.type = 'sine';
                            oscillator.frequency.setValueAtTime(800, audioContext.currentTime); // 800 Hz beep
                            oscillator.connect(audioContext.destination);
                            oscillator.start();
                            setTimeout(() => oscillator.stop(), 200); // Beep for 200ms

                            // 2. Speak the alert
                            const utterance = new SpeechSynthesisUtterance('Alert! Overcrowding detected!');
                            utterance.lang = 'en-US';
                            window.speechSynthesis.speak(utterance);
                        </script>
                        """
                        # We can't call st.html from here, so we print to console
                        # And the email will still send
                        print(f"ALERT! Overcrowding detected! Count: {self.last_final_count}")

                        if email_enabled:
                            send_email_alert(
                                EMAIL_SENDER,
                                EMAIL_PASSWORD,
                                EMAIL_RECEIVER,
                                self.last_final_count,
                                CROWD_LIMIT,
                            )

            # Draw on the original (non-resized) image
            img_with_info = draw_on_image(img, self.last_final_count, CROWD_LIMIT)

            # Convert the processed numpy array back to a VideoFrame
            new_frame = av.VideoFrame.from_ndarray(img_with_info, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

    webrtc_streamer(
        key="webcam",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )