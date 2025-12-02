import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
# Assuming 'model' and 'CSRNet' are defined in model.py
from model import CSRNet 
import time
import os
import tempfile 

# -----------------------------------------
# Custom CSS for Professional & Smooth Look (SKY BLUE & PINK THEME)
# -----------------------------------------
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
    }

    /* Page Background - Soft Sky Blue */
    [data-testid="stAppViewContainer"] {
        background-color: #E0FFFF; /* Light Sky Blue */
        color: #212529; /* Dark text for readability */
    }

    /* Sidebar - Clean White Panel */
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        padding: 20px;
        border-right: 1px solid #DEE2E6; /* Subtle border */
    }
    
    [data-testid="stSidebar"] h2 {
        color: #FF69B4; /* Hot Pink accent */
        margin-bottom: 20px;
    }
    
    /* Main Title - Hot Pink */
    h1 {
        color: #FF69B4 !important; /* Hot Pink */
        text-align: center;
        font-weight: 800 !important; 
        padding-top: 10px;
        margin-bottom: 25px;
        letter-spacing: 1px;
        text-shadow: none; 
    }
    
    /* Subtitles and Headers */
    h3, h4 {
        color: #495057 !important; /* Secondary dark text */
    }

    /* Radio buttons (Mode Selection) - Clean Tabs */
    .stRadio > div {
        display: flex;
        gap: 20px; 
        justify-content: center;
        margin-bottom: 20px;
    }

    .stRadio label {
        background: #FFFFFF;
        padding: 12px 25px;
        border-radius: 8px;
        border: 1px solid #CED4DA; /* Standard border */
        color: #495057 !important;
        font-weight: 600;
        font-size: 16px;
        transition: 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); /* Softer shadow */
    }

    .stRadio label:hover {
        background: #F1F3F5;
        cursor: pointer;
        border-color: #FF69B4; /* Pink border on hover */
        transform: translateY(-2px);
    }
    
    /* Selected radio button style - Hot Pink Highlight */
    .stRadio label[aria-checked="true"] {
        background: #FF69B4 !important; /* Hot Pink */
        color: #FFFFFF !important; /* White text on pink */
        border-color: #FF69B4;
        box-shadow: 0 4px 10px rgba(255, 105, 180, 0.3);
    }
    
    /* Primary Buttons - Hot Pink */
    .stButton>button {
        background-color: #FF69B4; /* Hot Pink */
        color: #FFFFFF; /* White text */
        font-size: 17px;
        font-weight: 700;
        border-radius: 8px;
        padding: 12px 20px;
        transition: 0.3s ease;
        border: none;
        box-shadow: 0 4px 10px rgba(255, 105, 180, 0.2);
    }

    .stButton>button:hover {
        background-color: #FF1493; /* Deep Pink on hover */
        transform: scale(1.02);
        box-shadow: 0 6px 15px rgba(255, 105, 180, 0.4);
    }

    /* Upload Widget Styling */
    .stFileUploader {
        background: #FFFFFF;
        border: 2px dashed #FF69B4; /* Pink dashed border */
        border-radius: 12px;
        padding: 20px;
        transition: 0.3s;
    }
    
    /* Uploaded image/video styling */
    img, video {
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); /* Lighter shadow */
    }
    
    /* Metric Display */
    [data-testid="stMetricValue"] {
        font-size: 3.5em; 
        font-weight: 800;
        color: #FF69B4 !important; /* Hot Pink */
        text-shadow: none;
    }
    [data-testid="stMetricLabel"] {
        color: #495057 !important; /* Dark text */
    }
    
    /* --- IMPRESSIVE DIALOG BOX (Custom High Alert) --- */
    /* Kept the standard red alert for critical warnings, as red/pink can be confusing */
    .custom-alert {
        background-color: #F8D7DA; /* Light red background */
        color: #721C24; /* Dark red text */
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        margin-bottom: 15px;
        font-size: 1.1em;
        font-weight: 700;
        border: 1px solid #F5C6CB; /* Red border */
        box-shadow: 0 0 15px rgba(220, 53, 69, 0.3); /* Soft red glow */
        animation: pulse 1.5s infinite alternate; 
    }
    
    @keyframes pulse {
      0% { transform: scale(1); opacity: 0.9; }
      100% { transform: scale(1.01); opacity: 1; }
    }
    
    .stAlert {
        border-radius: 8px;
    }
    
    /* Sidebar Threshold Text */
    .stSidebar p {
        color: #FF69B4 !important; /* Hot Pink */
    }

</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Streamlit Configuration
# -----------------------------------------
st.set_page_config(page_title="Crowd Analytics Platform", layout="wide")
st.title("Crowd Analytics Platform 📊")

st.markdown("""
<p style='text-align: center; color: #495057;'>
    Harnessing <b>CSRNet</b> and <b>YOLOv8</b> for high-precision crowd density mapping and real-time counting.
</p>
""", unsafe_allow_html=True)

# -----------------------------------------
# Sidebar Configuration (Dynamic Ruler/Threshold)
# -----------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration Panel")
    
    # Dynamic Ruler Implementation
    crowd_threshold = st.slider(
        "Alert Frequency Threshold (Scale 1-200)",
        min_value=1,
        max_value=200,
        value=30, # Default value, as requested
        step=5,
        help="Set the crowd count above which a high-frequency alert will be triggered in Video Mode."
    )
    st.markdown(f"""
        <p style='color: #FF69B4; font-weight: bold; margin-top: 5px;'>
            Current Alert Threshold: {crowd_threshold} people
        </p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Model Info:**\n- CSRNet (Density)\n- YOLOv8n (Detection)")


# -----------------------------------------
# Load Models (Cached) - LOGIC UNCHANGED
# -----------------------------------------
@st.cache_resource
def load_csrnet_model():
    try:
        model = CSRNet()
        # NOTE: This function assumes weights.pth is in the execution path or handles its own path.
        # We keep the loading logic as it was provided to us.
        checkpoint = torch.load("weights.pth", map_location="cpu")
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load CSRNet model. Check 'weights.pth' and 'model.py'. Error: {e}")
        st.stop() 

@st.cache_resource
def load_yolo_model():
    try:
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"Failed to load YOLO model. Check 'yolov8n.pt'. Error: {e}")
        st.stop()

# Initialize models
csrnet_model = load_csrnet_model()
yolo_model = load_yolo_model()

# Preprocessing for CSRNet
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -----------------------------------------
# Mode Selection - LOGIC UNCHANGED
# -----------------------------------------
mode = st.radio("Select Analysis Mode:", ("🖼️ Image Density Analysis", "🎥 Real-Time Video Stream"))
st.markdown("---")

# ============================================================
# 🖼️ IMAGE MODE - LOGIC UNCHANGED
# ============================================================
if mode == "🖼️ Image Density Analysis":
    uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Predict Crowd Count"):
            with st.spinner("Analyzing crowd... please wait ⏳"):
                img_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = csrnet_model(img_tensor)
                    count = int(output.sum().item())
                    density_map = output.squeeze().numpy()

                density_norm = (density_map - np.min(density_map)) / (
                    np.max(density_map) - np.min(density_map) + 1e-5
                )
                density_colored = cv2.applyColorMap(
                    (density_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
                )

                # st.success(f"✅ Estimated Crowd Count: **{count}**")
                if count >= crowd_threshold:
                 st.markdown(
                    f"<div class='custom-alert'>🚨 **HIGH ALERT!** The estimated crowd count of <b>{count}</b> exceeds the configured threshold of <b>{crowd_threshold}</b>. Immediate attention may be required.</div>",
                    unsafe_allow_html=True
                 )
                else:
                    st.success(f"✅ Analysis complete. Count ({count}) is below the high-alert threshold.")
                st.image(density_colored, caption="Predicted Density Map", channels="BGR", use_container_width=True)
    
# ============================================================
# 🎥 VIDEO MODE - LOGIC UNCHANGED
# ============================================================
elif mode == "🎥 Real-Time Video Stream":
    st.markdown("### ⚙️ Select Source")
    # Updated label to match the original structure's requirement
    video_option = st.radio("", ("📁 Upload Video File", "📷 Use Webcam")) 

    # Placeholder for the video stream and alert box
    stframe = st.empty()
    alert_placeholder = st.empty()
    count_placeholder = st.empty()
    
    # ---------------------------------------------
    # 📁 Upload Video (Logic UNCHANGED)
    # ---------------------------------------------
    # if video_option == "📁 Upload Video File":
    #     uploaded_video = st.file_uploader("Upload an MP4, AVI, or MOV video file", type=["mp4", "avi", "mov"])

    #     if uploaded_video is not None:
    #         # Use tempfile for robust handling of uploaded files
    #         tfile = tempfile.NamedTemporaryFile(delete=False)
    #         tfile.write(uploaded_video.read())
    #         temp_video_path = tfile.name
    #         tfile.close()

    #         cap = cv2.VideoCapture(temp_video_path)
    #         last_crowd_count = 0
    #         prev_time = time.time()
            
    #         st.info("🎬 Processing video stream... Display will update every 2 seconds for crowd density calculation.")

    #         # Audio for alert (kept as in the original code)
    #         beep_sound = """
    #         <audio autoplay>
    #             <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
    #         </audio>
    #         """

    #         while cap.isOpened():
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break

    #             # Resize frame for consistency 
    #             frame = cv2.resize(frame, (680, 360))
                
    #             # --- YOLO Detection for visual bounding boxes ---
    #             results = yolo_model(frame, verbose=False)
    #             # Ensure the results object has boxes before trying to access them
    #             boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 and hasattr(results[0], 'boxes') else []

    #             for (x1, y1, x2, y2) in boxes:
    #                 # Draw subtle bounding boxes (BGR format for OpenCV) - Pink: (180, 105, 255)
    #                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) 

    #             # --- CSRNet Density Calculation (Every 2 seconds) ---
    #             current_time = time.time()
    #             if current_time - prev_time >= 2:
    #                 pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #                 # CSRNet often requires square input, resize if needed
    #                 pil_img_resized = pil_img.resize((512, 512))
    #                 img_tensor = transform(pil_img_resized).unsqueeze(0)
                    
    #                 with torch.no_grad():
    #                     output = csrnet_model(img_tensor)
    #                     last_crowd_count = int(output.sum().item())
                        
    #                 prev_time = current_time

    #             # --- Dynamic Alert Logic and Display ---
    #             if last_crowd_count > crowd_threshold:
    #                 alert_placeholder.markdown(
    #                     f"<div class='custom-alert'>🔥 **CRITICAL OVERLOAD** | Count: {last_crowd_count} | Threshold: {crowd_threshold}</div>", 
    #                     unsafe_allow_html=True
    #                 )
    #                 st.markdown(beep_sound, unsafe_allow_html=True) # Siren sound
    #             else:
    #                 alert_placeholder.empty()
                    
    #             # Update Count Metric
    #             count_placeholder.metric(label="Real-time Estimated Count", value=f"{last_crowd_count}", delta_color="off")


    #             # Display frame in Streamlit
    #             stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    #                           caption=f"Live View (Last CSRNet update: {last_crowd_count})",
    #                           channels="RGB", use_container_width=True)
                
    #             # Smooth Animation for the video frame placeholder
    #             time.sleep(1/30) 

    #         cap.release()
    #         os.unlink(temp_video_path)
    #         stframe.empty()
    #         count_placeholder.empty()
    #         alert_placeholder.empty()
    #         st.success("✅ Video processing complete!")
    if video_option == "📁 Upload Video File":
        uploaded_video = st.file_uploader("Upload an MP4, AVI, or MOV video file", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            # Use tempfile for robust handling of uploaded files
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            temp_video_path = tfile.name
            tfile.close()

            cap = cv2.VideoCapture(temp_video_path)
            last_crowd_count = 0
            prev_time = time.time()

            st.info("🎬 Processing video... please wait")

            beep_sound = """
            <audio autoplay>
                <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
            </audio>
            """

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # ✔ OLD LOGIC: Resize to 640×360
                frame = cv2.resize(frame, (640, 360))

                # ✔ YOLO Detection (same as old)
                results = yolo_model(frame, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # ✔ CSRNet every 2 seconds (old logic)
                current_time = time.time()
                if current_time - prev_time >= 2:
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # ✔ OLD LOGIC: NO 512×512 resize
                    img_tensor = transform(pil_img).unsqueeze(0)

                    with torch.no_grad():
                        output = csrnet_model(img_tensor)
                        last_crowd_count = int(output.sum().item())

                    prev_time = current_time

                # ✔ OLD THRESHOLD: Fixed 30
                if last_crowd_count > 30:
                    alert_placeholder.warning(
                        f"🚨 **Alert:** High crowd detected! Count = {last_crowd_count}", 
                        icon="⚠️"
                    )
                    st.markdown(beep_sound, unsafe_allow_html=True)
                else:
                    alert_placeholder.empty()

                # ✔ OLD TEXT ON SCREEN
                cv2.putText(frame, f"Crowd Count: {last_crowd_count}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # ✔ Streamlit video output
                stframe.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True
                )

            cap.release()
            os.unlink(temp_video_path)
            st.success("✅ Video processing complete!")

    # ---------------------------------------------
    # 📷 WEBCAM MODE (REVERTED TO ORIGINAL LOCAL LOGIC)
    # ---------------------------------------------
    elif video_option == "📷 Use Webcam":
        st.markdown("### 🎥 Run Local Webcam (CSRNet + Face Detection)")
        st.info("Press the button below to start your webcam window. Press 'Q' to stop.")

        if st.button("▶️ Start Webcam"):
            # Imports inside the button handler as per your original design
            import cv2
            import torch
            import numpy as np
            from torchvision import transforms
            from model import CSRNet

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.write(f"✅ Using device: {device}")

            # NOTE: Keeping the Windows-specific path as requested
            weights_path = r"C:\22-7359\Deep Vision-2\Root\weights.pth" 

            model = CSRNet().to(device)
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.eval()
            st.write("✅ Model and weights loaded successfully.")

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not open webcam.")
                # We use st.stop() instead of your original flow control to exit streamlit app gracefully
                st.stop()

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output_crowd.avi', fourcc, 20.0, (640, 480))

            # NOTE: Keeping the local cascade classifier path
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            st.success("✅ Webcam started — **Look for the separate OpenCV window.** Press 'Q' on that window to exit.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ No frame captured.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                # Face detection boxes (Pink)
                for (x, y, w, h) in faces:
                    # BGR color for Pink #FF69B4: (180, 105, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 

                frame_resized = cv2.resize(frame, (512, 512))
                img_tensor = transform(frame_resized).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img_tensor)

                density_map = output.squeeze().cpu().numpy()
                count = np.sum(density_map)

                # Text output on the frame (Pink)
                cv2.putText(frame, f"People Count: {int(count)}", (20, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255,0), 3)

                out.write(frame)
                # This opens the new window for local execution
                cv2.imshow("Crowd Counting + Face Detection (Press Q to exit)", frame) 

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
            st.success("🛑 Webcam stopped and output video saved as `output_rowd.avi`.")