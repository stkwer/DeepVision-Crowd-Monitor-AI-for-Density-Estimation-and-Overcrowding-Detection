import os
import time
from datetime import datetime
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as transforms
import yagmail
from PIL import Image

import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None

from model import CSRNet


# ===============================
# MODEL LOADING
# ===============================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet(load_weights=True).to(device)
    model.eval()

    checkpoint = torch.load("best_model.pth", map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model, device


# ===============================
# UTILITIES
# ===============================
def preprocess(img: Image.Image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)


def generate_heatmap_and_mask(output, base_img_bgr: np.ndarray):
    """
    Returns overlay image and normalized density mask (0–255) resized to base image size.
    """
    density = output.squeeze().detach().cpu().numpy()
    density = np.maximum(density, 0)
    if density.max() > 0:
        density_norm = (density / density.max()) * 255.0
    else:
        density_norm = density
    density_uint8 = density_norm.astype(np.uint8)
    heatmap = cv2.applyColorMap(density_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (base_img_bgr.shape[1], base_img_bgr.shape[0]))
    overlay = cv2.addWeighted(base_img_bgr, 0.6, heatmap, 0.4, 0)
    return overlay, density_uint8


def get_bounding_boxes_from_density(density_uint8_resized, min_rel_thresh=0.5, min_area=150):
    """
    Approximate bounding boxes by thresholding the density map and finding contours.
    density_uint8_resized: 2D uint8 mask (0–255) same size as frame.
    """
    if density_uint8_resized.max() == 0:
        return []

    # Normalize to 0–1 then threshold
    norm = density_uint8_resized.astype(np.float32) / 255.0
    mask = (norm > min_rel_thresh).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
    return boxes


def send_email_alert(sender_email, app_password, receiver_email, count):
    try:
        yag = yagmail.SMTP(user=sender_email, password=app_password)
        subject = "Crowd Alert Triggered"
        body = (
            f"Overcrowding detected.\n\n"
            f"Estimated crowd count: {count}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Please take appropriate action."
        )
        yag.send(to=receiver_email, subject=subject, contents=body)
        st.sidebar.success("Alert email sent successfully.")
    except Exception as e:
        st.sidebar.error(f"Email send failed: {e}")


def get_system_stats():
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    gpu_text = "N/A"
    gpu_val = None

    if GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_val = gpu.load * 100.0
                gpu_text = f"{gpu_val:.1f}%"
        except Exception:
            gpu_text = "Error"
    return cpu, ram, gpu_text


# ===============================
# STREAMLIT LAYOUT CONFIG
# ===============================
st.set_page_config(page_title="CSRNet Crowd Density Monitor", layout="wide")

st.title("CSRNet Crowd Density Monitoring System")
st.markdown(
    "This application estimates crowd density using a CSRNet model and visualizes "
    "results as heatmaps and crowd counts over time."
)

# Sidebar controls
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Mode", ["Image Upload", "Video Upload", "Webcam Stream"])
threshold = st.sidebar.slider("Crowd Alert Threshold", 10, 2000, 200, 10)

enable_email = st.sidebar.checkbox("Enable Email Alerts", value=False)
if enable_email:
    st.sidebar.subheader("Email Settings")
    sender_email = st.sidebar.text_input("Sender Gmail", placeholder="you@gmail.com")
    app_password = st.sidebar.text_input("App Password (Gmail App Password)", type="password")
    receiver_email = st.sidebar.text_input("Receiver Email", placeholder="receiver@gmail.com")
else:
    sender_email = app_password = receiver_email = None

st.sidebar.header("Advanced Features")
show_fps = st.sidebar.checkbox("Show FPS on video/webcam", value=True)
show_fps_chart = st.sidebar.checkbox("Show FPS chart", value=False)
enable_bboxes = st.sidebar.checkbox("Show approximate bounding boxes", value=False)
enable_perf_panel = st.sidebar.checkbox("Show performance panel (CPU / RAM / GPU)", value=False)

output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

model, device = load_model()


# ===============================
# IMAGE MODE
# ===============================
if mode == "Image Upload":
    st.subheader("Image Analysis")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_cv = np.array(image)[:, :, ::-1]  # RGB to BGR

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        inp = preprocess(image).to(device)
        with torch.no_grad():
            output = model(inp)

        count = int(output.sum().item())
        overlay, density_uint8 = generate_heatmap_and_mask(output, img_cv)

        if enable_bboxes:
            # Resize density mask to frame size for boxes
            density_resized = cv2.resize(
                density_uint8,
                (img_cv.shape[1], img_cv.shape[0])
            )
            boxes = get_bounding_boxes_from_density(density_resized)
            for (x, y, w, h) in boxes:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)

        with col2:
            st.image(
                overlay,
                caption=f"Heatmap Overlay — Estimated Count: {count}",
                channels="BGR",
                use_container_width=True
            )

        if count > threshold:
            st.error(f"Overcrowding detected. Estimated count: {count}")
            if enable_email and sender_email and app_password and receiver_email:
                send_email_alert(sender_email, app_password, receiver_email, count)


# ===============================
# VIDEO MODE
# ===============================
elif mode == "Video Upload":
    st.subheader("Video Analysis")

    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if video_file:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(video_file.read())
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)
        if not cap.isOpened():
            st.error("Failed to open uploaded video.")
        else:
            st.info("Processing video. Please wait, graphs will update as frames are processed.")

            fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = os.path.join(
                output_dir,
                f"video_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

            frame_placeholder = st.empty()
            live_chart_placeholder = st.empty()
            fps_chart_placeholder = st.empty()
            progress_bar = st.progress(0)

            if enable_perf_panel:
                st.sidebar.subheader("System Performance")
                cpu_placeholder = st.sidebar.empty()
                ram_placeholder = st.sidebar.empty()
                gpu_placeholder = st.sidebar.empty()
            else:
                cpu_placeholder = ram_placeholder = gpu_placeholder = None

            times = []
            counts = []
            alerts = []
            fps_times = []
            fps_values = []

            frame_idx = 0
            prev_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                now_time = time.time()
                proc_fps = 1.0 / (now_time - prev_time) if frame_idx > 0 else 0.0
                prev_time = now_time

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                inp = preprocess(pil_img).to(device)

                with torch.no_grad():
                    output = model(inp)

                count = int(output.sum().item())
                t_sec = frame_idx / fps

                overlay, density_uint8 = generate_heatmap_and_mask(output, frame)

                # Bounding boxes
                if enable_bboxes:
                    density_resized = cv2.resize(
                        density_uint8,
                        (frame.shape[1], frame.shape[0])
                    )
                    boxes = get_bounding_boxes_from_density(density_resized)
                    for (x, y, w, h) in boxes:
                        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)

                cv2.putText(
                    overlay,
                    f"Count: {count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

                if show_fps:
                    cv2.putText(
                        overlay,
                        f"FPS: {proc_fps:.1f}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                alert_flag = count > threshold
                if alert_flag:
                    cv2.putText(
                        overlay,
                        "CROWD ALERT",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2
                    )
                    if enable_email and sender_email and app_password and receiver_email:
                        send_email_alert(sender_email, app_password, receiver_email, count)

                writer.write(overlay)
                frame_placeholder.image(overlay, channels="BGR", use_container_width=True)

                times.append(t_sec)
                counts.append(count)
                alerts.append(1 if alert_flag else 0)
                fps_times.append(t_sec)
                fps_values.append(proc_fps)

                if frame_idx % max(int(fps // 2), 1) == 0:
                    df_live = pd.DataFrame({"time_s": times, "count": counts})
                    live_chart_placeholder.line_chart(
                        df_live.set_index("time_s"),
                        height=250
                    )

                    if show_fps_chart and fps_values:
                        df_fps = pd.DataFrame({"time_s": fps_times, "fps": fps_values})
                        fps_chart_placeholder.line_chart(
                            df_fps.set_index("time_s"),
                            height=250
                        )

                if enable_perf_panel and cpu_placeholder is not None:
                    cpu, ram, gpu_text = get_system_stats()
                    cpu_placeholder.metric("CPU Usage", f"{cpu:.1f}%")
                    ram_placeholder.metric("RAM Usage", f"{ram:.1f}%")
                    gpu_placeholder.metric("GPU Load", gpu_text)

                frame_idx += 1
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            writer.release()

            st.success(f"Processed video saved to: {out_path}")
            st.video(out_path)

            if times:
                st.subheader("Crowd Count Over Time (Summary)")
                df = pd.DataFrame(
                    {"time_s": times, "count": counts, "alert": alerts}
                )
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("Crowd count vs time")
                    st.line_chart(df.set_index("time_s")["count"], height=300)

                with col2:
                    st.markdown("Overcrowding events (1 = alert, 0 = normal)")
                    st.bar_chart(df.set_index("time_s")["alert"], height=300)

                st.markdown("Summary statistics")
                st.write(
                    {
                        "max_count": int(np.max(counts)),
                        "min_count": int(np.min(counts)),
                        "mean_count": float(np.mean(counts)),
                        "total_alert_frames": int(np.sum(alerts)),
                    }
                )


# ===============================
# WEBCAM STREAM MODE
# ===============================
elif mode == "Webcam Stream":
    st.subheader("Webcam Analysis")
    st.info("Webcam will start. Stop the Streamlit app to end the session.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        st.error("Unable to access webcam.")
    else:
        fps_target = 15.0
        out_path = os.path.join(
            output_dir,
            f"webcam_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_target, (1280, 720))

        frame_placeholder = st.empty()
        live_chart_placeholder = st.empty()
        fps_chart_placeholder = st.empty()

        if enable_perf_panel:
            st.sidebar.subheader("System Performance")
            cpu_placeholder = st.sidebar.empty()
            ram_placeholder = st.sidebar.empty()
            gpu_placeholder = st.sidebar.empty()
        else:
            cpu_placeholder = ram_placeholder = gpu_placeholder = None

        times = []
        counts = []
        alerts = []
        fps_times = []
        fps_values = []

        start_time = time.time()
        prev_time = start_time
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now_time = time.time()
            proc_fps = 1.0 / (now_time - prev_time) if frame_idx > 0 else 0.0
            prev_time = now_time

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            inp = preprocess(pil_img).to(device)

            with torch.no_grad():
                output = model(inp)

            count = int(output.sum().item())
            t_sec = time.time() - start_time

            overlay, density_uint8 = generate_heatmap_and_mask(output, frame)

            if enable_bboxes:
                density_resized = cv2.resize(
                    density_uint8,
                    (frame.shape[1], frame.shape[0])
                )
                boxes = get_bounding_boxes_from_density(density_resized)
                for (x, y, w, h) in boxes:
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)

            cv2.putText(
                overlay,
                f"Count: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            if show_fps:
                cv2.putText(
                    overlay,
                    f"FPS: {proc_fps:.1f}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            alert_flag = count > threshold
            if alert_flag:
                cv2.putText(
                    overlay,
                    "CROWD ALERT",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )
                if enable_email and sender_email and app_password and receiver_email:
                    send_email_alert(sender_email, app_password, receiver_email, count)

            writer.write(overlay)
            frame_placeholder.image(overlay, channels="BGR", use_container_width=True)

            times.append(t_sec)
            counts.append(count)
            alerts.append(1 if alert_flag else 0)
            fps_times.append(t_sec)
            fps_values.append(proc_fps)

            if frame_idx % 5 == 0:
                df_live = pd.DataFrame({"time_s": times, "count": counts})
                live_chart_placeholder.line_chart(
                    df_live.set_index("time_s"),
                    height=250
                )

                if show_fps_chart and fps_values:
                    df_fps = pd.DataFrame({"time_s": fps_times, "fps": fps_values})
                    fps_chart_placeholder.line_chart(
                        df_fps.set_index("time_s"),
                        height=250
                    )

            if enable_perf_panel and cpu_placeholder is not None:
                cpu, ram, gpu_text = get_system_stats()
                cpu_placeholder.metric("CPU Usage", f"{cpu:.1f}%")
                ram_placeholder.metric("RAM Usage", f"{ram:.1f}%")
                gpu_placeholder.metric("GPU Load", gpu_text)

            frame_idx += 1

        cap.release()
        writer.release()

        st.success(f"Webcam session saved to: {out_path}")

        if times:
            st.subheader("Crowd Count Over Time (Summary)")
            df = pd.DataFrame(
                {"time_s": times, "count": counts, "alert": alerts}
            )
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("Crowd count vs time")
                st.line_chart(df.set_index("time_s")["count"], height=300)

            with col2:
                st.markdown("Overcrowding events (1 = alert, 0 = normal)")
                st.bar_chart(df.set_index("time_s")["alert"], height=300)

            st.markdown("Summary statistics")
            st.write(
                {
                    "max_count": int(np.max(counts)),
                    "min_count": int(np.min(counts)),
                    "mean_count": float(np.mean(counts)),
                    "total_alert_frames": int(np.sum(alerts)),
                }
            )
