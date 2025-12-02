# crowd_streamlit_advanced.py
import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter
import tempfile, os, threading, datetime
import pandas as pd
import smtplib
from email.message import EmailMessage
from twilio.rest import Client
from pathlib import Path

# -------------------------
# Helper: send email via Gmail (use App Password)
# -------------------------
def send_email_alert(subject, body, to_email, from_email, app_password, attach_path=None):
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        msg.set_content(body)
        if attach_path and Path(attach_path).exists():
            with open(attach_path, "rb") as f:
                data = f.read()
            msg.add_attachment(data, maintype="image", subtype="jpeg", filename=Path(attach_path).name)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(from_email, app_password)
            smtp.send_message(msg)
        return True, "Email sent"
    except Exception as e:
        return False, str(e)

# -------------------------
# Helper: send SMS via Twilio
# -------------------------
def send_sms_alert(account_sid, auth_token, from_number, to_number, body):
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(body=body, from_=from_number, to=to_number)
        return True, message.sid
    except Exception as e:
        return False, str(e)

# -------------------------
# Draw heatmap from detected person centers
# -------------------------
def overlay_heatmap_on_frame(frame_bgr, centers, sigma=25, radius=30, alpha=0.5):
    h, w = frame_bgr.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for (cx, cy) in centers:
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(heat, (int(cx), int(cy)), radius, 1, -1)
    heat = gaussian_filter(heat, sigma=sigma)
    if heat.max() > 0:
        heat_col = cv2.applyColorMap((255 * (heat / heat.max())).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame_bgr, 1 - alpha, heat_col, alpha, 0)
    else:
        overlay = frame_bgr
    return overlay, heat

# -------------------------
# Setup Streamlit UI
# -------------------------
st.set_page_config(layout="wide", page_title="Advanced Crowd Monitor")
st.title("Advanced Crowd Monitor — Streamlit (YOLOv8 + Alerts)")

# Left panel - Controls
with st.sidebar:
    st.header("1) Camera & Model")
    camera_option = st.selectbox("Camera source", ["Laptop Webcam (0)", "Android IP Camera (RTSP/HTTP URL)"])
    ip_cam_url = st.text_input("If IP camera: enter URL (rtsp://... or http://...)", "")
    model_choice = st.selectbox("YOLO model", ["yolov8n.pt (fast)", "yolov8s.pt", "yolov8m.pt", "yolov8x.pt (accurate)"])
    load_model_button = st.button("Load Model")

    st.markdown("---")
    st.header("2) Alerts & Credentials")
    st.subheader("Email (Gmail)")
    gmail_user = st.text_input("Gmail address (from)", value=st.secrets.get("GMAIL_USER", ""))
    gmail_app_password = st.text_input("Gmail App Password", value=st.secrets.get("GMAIL_APP_PASSWORD", ""), type="password")
    alert_email_to = st.text_input("Alert recipient email", value=st.secrets.get("ALERT_EMAIL_TO", ""))
    st.markdown("**Use Gmail App Password**: https://support.google.com/accounts/answer/185833")

    st.subheader("SMS (Twilio)")
    twilio_sid = st.text_input("Twilio Account SID", value=st.secrets.get("TWILIO_SID", ""))
    twilio_token = st.text_input("Twilio Auth Token", value=st.secrets.get("TWILIO_TOKEN", ""), type="password")
    twilio_from = st.text_input("Twilio From Number (E.164)", value=st.secrets.get("TWILIO_FROM", ""))
    alert_phone_to = st.text_input("Alert phone number (E.164)", value=st.secrets.get("ALERT_PHONE", ""))

    st.markdown("---")
    st.header("3) Detection & Logging")
    alert_threshold = st.slider("Alert threshold (person count)", min_value=1, max_value=500, value=10)
    heatmap_sigma = st.slider("Heatmap sigma (blur)", 1, 100, 25)
    save_snapshots = st.checkbox("Save snapshot on alert", value=True)
    record_video = st.checkbox("Record processed video", value=False)
    save_folder = st.text_input("Save folder (optional)", value=str(Path.cwd()))
    cooldown_seconds = st.number_input("Alert cooldown (sec)", min_value=5, max_value=3600, value=60)

# Right panel - Live frames + status
col1, col2 = st.columns([2,1])
frame_display = col1.empty()
status_area = col2.empty()
log_area = col2.empty()

# Global states in session_state
if "running" not in st.session_state:
    st.session_state["running"] = False
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False
if "last_alert_time" not in st.session_state:
    st.session_state["last_alert_time"] = 0
if "log_df" not in st.session_state:
    st.session_state["log_df"] = pd.DataFrame(columns=["timestamp", "camera", "count", "saved_snapshot", "note"])

# Load model when requested
model = None
if load_model_button or (not st.session_state["model_loaded"] and st.session_state["running"]):
    with st.spinner("Loading YOLO model..."):
        model_path = model_choice
        # the ultralytics module accepts model names (it will auto-download)
        model = YOLO(model_path)
    st.session_state["model_loaded"] = True
    st.success(f"Model {model_path} loaded.")

# Start / Stop buttons
start_button = col2.button("Start Monitoring", key="start")
stop_button = col2.button("Stop Monitoring", key="stop")

# Prepare save folder
Path(save_folder).mkdir(parents=True, exist_ok=True)

# Main monitoring loop
def monitor_loop():
    # pick camera
    cam_src = 0 if camera_option.startswith("Laptop") else ip_cam_url
    # use DirectShow on Windows for webcam
    if cam_src == 0:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_src)
    if not cap.isOpened():
        status_area.error(" Could not open camera. Check URL or device.")
        return

    # Video writer if requested
    writer = None
    if record_video:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = Path(save_folder) / f"crowd_record_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        writer = cv2.VideoWriter(str(out_name), fourcc, fps, (w, h))

    # run loop
    alert_triggered = False
    while st.session_state["running"]:
        ret, frame = cap.read()
        if not ret:
            status_area.error("Video read error. Stopping.")
            break

        # run detection (use RGB frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # run model inference
        try:
            results = model(frame_rgb, verbose=False)
        except Exception as e:
            status_area.error(f"Model inference error: {e}")
            break

        centers = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                # ensure class 0 = person for COCO
                if model.names[cls] == "person":
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cx = int((x1 + x2) / 2)
                    cy = int(y2)  # bottom-center
                    centers.append((cx, cy))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        count = len(centers)
        overlay, heat = overlay_heatmap_on_frame(frame, centers, sigma=heatmap_sigma, radius=30, alpha=0.5)
        cv2.putText(overlay, f"Count: {count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        # Alert logic with cooldown
        now = time.time()
        can_alert = (now - st.session_state["last_alert_time"]) > cooldown_seconds
        if count >= alert_threshold and can_alert:
            note = ""
            # Save snapshot if requested
            snapshot_path = ""
            if save_snapshots:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_path = str(Path(save_folder) / f"alert_snapshot_{ts}.jpg")
                cv2.imwrite(snapshot_path, overlay)
            # Send Email (non-blocking via thread)
            if gmail_user and gmail_app_password and alert_email_to:
                subj = f"[ALERT] Overcrowding detected ({count})"
                body = f"Overcrowding detected at {datetime.datetime.now().isoformat()}\nCount = {count}"
                threading.Thread(target=send_email_alert, args=(subj, body, alert_email_to, gmail_user, gmail_app_password, snapshot_path)).start()
                note += "email_sent;"
            # Send SMS
            if twilio_sid and twilio_token and twilio_from and alert_phone_to:
                sms_body = f"ALERT: Overcrowding detected. Count={count} at {datetime.datetime.now().isoformat()}"
                threading.Thread(target=send_sms_alert, args=(twilio_sid, twilio_token, twilio_from, alert_phone_to, sms_body)).start()
                note += "sms_sent;"
            st.session_state["last_alert_time"] = now
            alert_triggered = True

        # Logging
        st.session_state["log_df"] = st.session_state["log_df"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "camera": str(cam_src),
            "count": count,
            "saved_snapshot": snapshot_path if (count >= alert_threshold and save_snapshots) else "",
            "note": note
        }, ignore_index=True)

        # Show frame in streamlit
        frame_display.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        # Write video
        if writer is not None:
            writer.write(overlay)

        # small sleep to yield
        time.sleep(0.01)

    # cleanup
    try:
        cap.release()
    except:
        pass
    if writer is not None:
        writer.release()

# Buttons logic
if start_button:
    if not st.session_state["model_loaded"]:
        with st.spinner("Loading model before starting..."):
            model = YOLO(model_choice)
            st.session_state["model_loaded"] = True
    st.session_state["running"] = True
    # run monitor in separate thread so Streamlit UI remains responsive
    threading.Thread(target=monitor_loop, daemon=True).start()
    st.success("Monitoring started.")

if stop_button:
    st.session_state["running"] = False
    st.success("Stopping monitoring...")

# Show live log, and allow save or download
with st.expander("Live log & controls"):
    st.write("Last 10 log entries:")
    st.dataframe(st.session_state["log_df"].tail(10))
    if st.button("Save log to CSV"):
        csv_path = Path(save_folder) / f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        st.session_state["log_df"].to_csv(csv_path, index=False)
        st.success(f"Saved log to {csv_path}")
    if st.button("Clear log"):
        st.session_state["log_df"] = pd.DataFrame(columns=["timestamp", "camera", "count", "saved_snapshot", "note"])
        st.info("Log cleared.")
