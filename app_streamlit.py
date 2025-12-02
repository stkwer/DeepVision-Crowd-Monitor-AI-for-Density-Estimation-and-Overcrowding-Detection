# app_streamlit.py
import os, tempfile, cv2, numpy as np, torch, streamlit as st
from csrnet import CSRNet
from inference_utils import (
    load_model, predict_density, make_overlay, find_blob_peaks,
    draw_peaks, put_text
)

st.set_page_config(page_title="DeepVision Crowd Monitor", layout="wide")
st.title("DeepVision Crowd Monitor – Milestone 3")

# --- Sidebar controls ---
weights_path = st.sidebar.text_input("Weights path (.pth)", "/content/drive/MyDrive/checkpoints_A/best_mae.pth")
scale = st.sidebar.number_input("Calibration scale", value=1.0, step=0.0001, format="%.4f")
clip_pct = st.sidebar.number_input("Clip percentile", value=99.5, step=0.1)
alpha = st.sidebar.slider("Overlay alpha", min_value=0.0, max_value=1.0, value=0.5)
peak_pct = st.sidebar.number_input("Peak percentile (blobs)", value=99.4, step=0.1)
min_sep = st.sidebar.number_input("Min separation (px)", value=10, step=1)
blob_sigma = st.sidebar.number_input("Blob blur sigma", value=4, step=1)
alert_thresh = st.sidebar.number_input("Alert threshold (count)", value=1000.0, step=10.0)
font_scale = st.sidebar.number_input("Text size", value=1.2, step=0.1)
use_cpu = st.sidebar.checkbox("Force CPU", value=False)

device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
st.sidebar.write(f"Device: **{device}**")

# preload model lazily on first use
@st.cache_resource(show_spinner=False)
def _load(weights, dev):
    return load_model(CSRNet, weights, dev)

mode = st.tabs(["🖼️ Image", "🎞️ Video"])

with mode[0]:
    st.subheader("Single Image")
    imfile = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
    if imfile is not None and weights_path.strip():
        file_bytes = np.frombuffer(imfile.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        model = _load(weights_path, device)
        out = predict_density(model, img, device, out_size=(384,384), scale=scale)
        den, pred = out["density"], out["pred_count"]
        overlay = make_overlay(img, den, clip_pct=clip_pct, alpha=alpha)
        pts = find_blob_peaks(den, peak_percentile=peak_pct, min_sep=int(min_sep), blob_sigma=int(blob_sigma))
        vis = draw_peaks(overlay, pts)
        vis = put_text(vis, f"Pred: {pred:.1f}", org=(10, 34), scale=font_scale)

        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
        c2.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"Overlay + Blobs (count≈{pred:.1f})", use_column_width=True)

        if pred >= alert_thresh:
            st.error(f"⚠️ ALERT: predicted crowd {pred:.1f} ≥ {alert_thresh}")
        else:
            st.success(f"OK: predicted {pred:.1f}")

with mode[1]:
    st.subheader("Video (MP4/AVI)")
    vfile = st.file_uploader("Upload a short video", type=["mp4","avi","mov","mkv"])
    if vfile is not None and weights_path.strip():
        model = _load(weights_path, device)
        # temp write
        tdir = tempfile.mkdtemp()
        in_path = os.path.join(tdir, "in.mp4")
        with open(in_path, "wb") as f: f.write(vfile.read())

        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(tdir, "out.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v"); writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

        frame_counts = []
        with st.spinner("Processing video..."):
            while True:
                ok, frame = cap.read()
                if not ok: break
                out = predict_density(model, frame, device, out_size=(384,384), scale=scale)
                den, pred = out["density"], out["pred_count"]
                overlay = make_overlay(frame, den, clip_pct=clip_pct, alpha=alpha)
                pts = find_blob_peaks(den, peak_percentile=peak_pct, min_sep=int(min_sep), blob_sigma=int(blob_sigma))
                vis = draw_peaks(overlay, pts)
                vis = put_text(vis, f"Count: {pred:.1f}", org=(10, 34), scale=font_scale)
                if pred >= alert_thresh:
                    vis = put_text(vis, f"ALERT ≥ {alert_thresh}", org=(10, 70), scale=font_scale, color=(0,255,255))
                writer.write(vis); frame_counts.append(pred)

        cap.release(); writer.release()
        st.video(out_path)
        st.write(f"Frames: {len(frame_counts)} | Mean count: {np.mean(frame_counts):.1f}")




