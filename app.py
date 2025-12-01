import streamlit as st
from components.header import show_header
from components.footer import show_footer
import streamlit as st
# import bcrypt
import json
import os
import cv2
import numpy as np
import onnxruntime as ort
from torchvision import transforms
# import winsound

import streamlit as st
import base64

import time
from scipy.ndimage import label, find_objects

from email_alert import send_email_alert
import threading

# from werkzeug.security import generate_password_hash
from werkzeug.security import generate_password_hash, check_password_hash

# ----- Cross-platform sound handling -----
try:
    import winsound  # Works only on Windows
    SOUND_AVAILABLE = True
except:
    winsound = None
    SOUND_AVAILABLE = False


# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="CrowdVision AI Dashboard",
    page_icon="",
    layout="wide"
)


def fire_and_forget_email(count):
    threading.Thread(target=send_email_alert, args=(count,), daemon=True).start()



# ---------- GLOBAL STYLING ----------
def set_app_theme(background_path=None):
    st.markdown("""
                
        <style>
        /* General App Styling */
        .stApp {
            background-color: #121212;
            color: #F5F5F5;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Optional background image */
        """ + (
            f""".stApp {{
                background-image: url("data:image/jpg;base64,{base64.b64encode(open(background_path, "rb").read()).decode()}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-position: center;
            }}""" if background_path else ""
        ) + """
        
        /* Titles */
        h1, h2, h3 {
            color: #1E88E5;
            font-weight: 700;
        }

        /* Buttons */
        div.stButton > button {
            background-color: #00ffff;
            color: white;
            padding: 0.6em 1.2em;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        div.stButton > button:hover {
            background-color: #0ef;
            transform: scale(1.05);
            box-shadow: 0 0 5px cyan, 0 0 25px cyan, 0 0 50px cyan, 0 0 100px cyan,
    0 0 200px cyan;
        }

        /* File uploader */
        .stFileUploader label {
            color: #F5F5F5;
            font-weight: 600;
        }

        /* Text inputs */
        .stTextInput input {
            background-color: #1e1e1e;
            color: #fff;
            border: 1px solid #1E88E5;
            border-radius: 6px;
        }

        /* Footer */
        footer {
            visibility: hidden;
        }

        .custom-footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #1E1E1E;
            color: #ccc;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
        }
        .metric-box {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.15);
    }
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,255,255,0.3);
    }
    .metric-value {
        font-size: 36px;
        color: #00ffff;
        font-weight: 700;
    }
    .metric-label {
        font-size: 16px;
        color: white;
        margin-top: 5px;
    }
        </style>
    """, unsafe_allow_html=True)

# ✅ Apply theme with optional background image
set_app_theme("assets/backgroundimage.jpg")


# ✅ Add fixed footer (optional)
st.markdown("""
    <div class="custom-footer">
        © 2025 CrowdVision AI | Smart Real-Time Crowd Analytics
    </div>
""", unsafe_allow_html=True)


def set_global_background(image_file):
    """
    Sets a full-page background image for all Streamlit pages (using a local image)
    """
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Optional - make text visible */
        h1, h2, h3, h4, h5, h6, p, span, label {{
            color: white !important;
            text-shadow: 1px 1px 2px black;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Call the function with your local image path
set_global_background("assets/backgroundimage.jpg")





# -------------------------------
# Initialize session state
# -------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

# -------------------------------
# Show Header
# -------------------------------
show_header()

# -------------------------------
# Page Routing Logic
# -------------------------------
page = st.session_state["page"]

# --------- HOME PAGE ----------
# ---------------------------------------------------------------------------------------
if page == "home":
    st.write("")  # Spacer
    
    st.markdown(
        """
        <div style="
            background-color: rgba(0,0,0,0.4); 
            padding: 100px; 
            text-align:center;
            border-radius: 10px;
        ">
            <h1 style="color:white;font-size:60px;">CrowdVision AI</h1>
            <p style="color:white;font-size:24px;">Real-time crowd monitoring with AI-powered alerts</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    
# ----------------------------------------------------------------------------------------
USER_FILE = "users.json"

# Ensure user file exists
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

# # -------------------------------
# # Signup
# # -------------------------------
# if st.session_state["page"] == "signup":
#     st.title(" Signup")
#     new_user = st.text_input("Choose Username", key="new_user")
#     new_pass = st.text_input("Choose Password", type="password", key="new_pass")
#     confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")

#     if st.button("Create Account", key="btn_signup_page"):
#         if new_pass != confirm_pass:
#             st.error("Passwords do not match")
#         else:
#             # Load existing users
#             with open(USER_FILE, "r") as f:
#                 users = json.load(f)
#             if new_user in users:
#                 st.error("Username already exists")
#             else:
#                 # Hash the password
#                 # hashed_pass = bcrypt.hashpw(new_pass.encode(), bcrypt.gensalt())
#                 # users[new_user] = hashed_pass.decode()
#                 hashed_pass = generate_password_hash(new_pass)
#                 users[new_user] = hashed_pass
#                 with open(USER_FILE, "w") as f:
#                     json.dump(users, f)
#                 st.success(f"Account created for {new_user}!")
#                 st.session_state["page"] = "login"

# # -------------------------------
# # Login
# # -------------------------------
# elif st.session_state["page"] == "login":
#     st.title(" Login")
#     username = st.text_input("Username", key="username_login")
#     password = st.text_input("Password", type="password", key="password_login")

#     if st.button("Login", key="btn_login_page"):
#         with open(USER_FILE, "r") as f:
#             users = json.load(f)
#         if username in users:
#             hashed_pass = users[username].encode()
#             # if bcrypt.checkpw(password.encode(), hashed_pass):
#             #     st.session_state["logged_in"] = True
#             if username in users and check_password_hash(users[username], password):
#                 st.session_state["logged_in"] = True
#                 st.session_state["username"] = username
#                 st.success(f"Welcome, {username}!")
#                 st.session_state["page"] = "dashboard"
#             else:
#                 st.error("Invalid username or password")
#         else:
#             st.error("Invalid username or password")
# -------------------------------
# Signup
# -------------------------------
if st.session_state["page"] == "signup":
    st.title(" Signup")
    new_user = st.text_input("Choose Username", key="new_user")
    new_pass = st.text_input("Choose Password", type="password", key="new_pass")
    confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")

    if st.button("Create Account", key="btn_signup_page"):
        if new_pass != confirm_pass:
            st.error("Passwords do not match")
        else:
            # Load existing users
            with open(USER_FILE, "r") as f:
                users = json.load(f)

            if new_user in users:
                st.error("Username already exists")
            else:
                # Hash the password (Werkzeug)
                hashed_pass = generate_password_hash(new_pass)
                users[new_user] = hashed_pass

                with open(USER_FILE, "w") as f:
                    json.dump(users, f)

                st.success(f"Account created for {new_user}!")
                st.session_state["page"] = "login"
# -------------------------------
# Login
# -------------------------------
elif st.session_state["page"] == "login":
    st.title(" Login")
    username = st.text_input("Username", key="username_login")
    password = st.text_input("Password", type="password", key="password_login")

    if st.button("Login", key="btn_login_page"):
        with open(USER_FILE, "r") as f:
            users = json.load(f)

        if username in users:
            stored_hash = users[username]

            # Validate password
            if check_password_hash(stored_hash, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"Welcome, {username}!")
                st.session_state["page"] = "dashboard"
            else:
                st.error("Invalid username or password")
        else:
            st.error("Invalid username or password")

# --------- DASHBOARD PAGE ----------
# --------- DASHBOARD PAGE ----------
elif page == "dashboard":
    st.title("Dashboard")

    # --------- LOGIN CHECK ---------
    if not st.session_state["logged_in"]:
        st.warning("You must login first!")
        st.session_state["page"] = "login"

    else:
        st.write(f"Welcome to Dashboard, {st.session_state['username']}!")
       

        

        # --------- SIDEBAR SETTINGS ---------
        with st.sidebar:
            st.header("Settings")
            CROWD_LIMIT = st.slider("Crowd Limit", 10, 100, 37)
            SCALE = st.slider("Video Scale", 0.2, 1.0, 0.5)

        # Initialize email cooldown state (Step 3)
        if "email_cooldown" not in st.session_state:
            st.session_state["email_cooldown"] = 0

        # --------- CHOOSE INPUT MODE ---------
        mode = st.radio("Select Input Mode", ["Recorded Video", "Live Webcam", "Image"], horizontal=True)

        # --------- LOAD MODEL (shared for all modes) ---------
        if "onnx_session" not in st.session_state:
            try:
                st.session_state["onnx_session"] = ort.InferenceSession(
                    "checkpoints/csrnet.onnx", providers=['CUDAExecutionProvider']
                )
            except Exception:
                st.session_state["onnx_session"] = ort.InferenceSession(
                    "checkpoints/csrnet.onnx", providers=['CPUExecutionProvider']
                )

        session = st.session_state["onnx_session"]
        input_name = session.get_inputs()[0].name

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # --------- MODE 1: RECORDED VIDEO ---------
        if mode == "Recorded Video":
            uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"], key="upload_video")

            # -----------------------------------------------------------
            # 🔥 Create the 3 Dynamic Metric Boxes (BEFORE reading video)
            # -----------------------------------------------------------
                        # -----------------------------------------------------------

            if uploaded_file is not None:
                
                if "last_email_time" not in st.session_state:
                    st.session_state["last_email_time"] = 0

                tfile = "temp_video.mp4"
                with open(tfile, 'wb') as f:
                    f.write(uploaded_file.read())

                cap = cv2.VideoCapture(tfile)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                out_file = "processed_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

                stframe = st.empty()
                frame_count = 0

                # FPS counters for live updating
                fps_calc = 0
                fps_start = time.time()

                col1, col2, col3 = st.columns(3)
                with col1:
                    crowd_box = st.empty()
                with col2:
                    status_box = st.empty()
                with col3:
                    fps_box = st.empty()

                # Initial empty UI
                crowd_box.markdown("""
                    <div class="metric-box">
                        <div class="metric-value">0</div>
                        <div class="metric-label">Crowd Count</div>
                    </div>
                """, unsafe_allow_html=True)

                status_box.markdown("""
                    <div class="metric-box">
                        <div class="metric-value">Normal</div>
                        <div class="metric-label">Status</div>
                    </div>
                """, unsafe_allow_html=True)

                fps_box.markdown("""
                    <div class="metric-box">
                        <div class="metric-value">0.00</div>
                        <div class="metric-label">FPS</div>
                    </div>
                """, unsafe_allow_html=True)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1

                    frame_scaled = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
                    img_rgb = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2RGB)
                    h, w, _ = img_rgb.shape
                    new_w = (w // 8) * 8
                    new_h = (h // 8) * 8
                    img_resized = cv2.resize(img_rgb, (new_w, new_h))
                    tensor = transform(img_resized).unsqueeze(0).numpy().astype(np.float32)

                    preds = session.run(None, {input_name: tensor})[0]
                    density_map = preds.squeeze(0).squeeze(0)
                    count = float(np.sum(density_map))



                    # ---------------- EMAIL ALERT LOGIC --------------------
                    # ---------------- EMAIL ALERT LOGIC (time-based throttle) -----------------
                    try:
                        current_time = time.time()
                        THROTTLE_SECONDS = 2   # Send alert every 10 seconds if still overcrowded

                        if count > CROWD_LIMIT:
                            if current_time - st.session_state["last_email_time"] >= THROTTLE_SECONDS:
                                import threading
                                from email_alert import send_email_alert

                                threading.Thread(target=send_email_alert, args=(count,), daemon=True).start()
                                st.toast("📩 Email alert sent")
                                st.sidebar.success("📩 Email alert sent")

                                st.session_state["last_email_time"] = current_time

                    except Exception as e:
                        print("Email alert error:", e)

                    # -------------------------------------------------------


                    density_resized = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_CUBIC)
                    density_norm = np.clip(density_resized, 0, np.percentile(density_resized, 99))
                    density_norm = density_norm / (density_norm.max() + 1e-8)
                    density_uint8 = (density_norm * 255).astype(np.uint8)
                    density_blur = cv2.GaussianBlur(density_uint8, (5, 5), 0)
                    density_color = cv2.applyColorMap(density_blur, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(frame_scaled, 0.7, density_color, 0.3, 0)


                    # -----------------------------------------------------------
                    # 🔥 UPDATE DYNAMIC METRIC BOXES (INSIDE LOOP)
                    # -----------------------------------------------------------
                    status_text = "⚠️ Overcrowded" if count > CROWD_LIMIT else "Normal"
                    status_color = "#ff4c4c" if count > CROWD_LIMIT else "#00ff99"

                    crowd_box.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">{count:.1f}</div>
                            <div class="metric-label">Crowd Count</div>
                        </div>
                    """, unsafe_allow_html=True)

                    status_box.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color:{status_color};">{status_text}</div>
                            <div class="metric-label">Status</div>
                        </div>
                    """, unsafe_allow_html=True)

                    # fps_box.markdown(f"""
                    #     <div class="metric-box">
                    #         <div class="metric-value">{fps_calc:.2f}</div>
                    #         <div class="metric-label">FPS</div>
                    #     </div>
                    # """, unsafe_allow_html=True)
                    fps_box.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{fps:.2f}</div>
                        <div class="metric-label">FPS</div>
                    </div>
                """, unsafe_allow_html=True)
                    # -----------------------------------------------------------


                    # Update FPS every 10 frames
                    if frame_count % 10 == 0:
                        fps_end = time.time()
                        fps_calc = 10 / (fps_end - fps_start)
                        fps_start = fps_end


                    # Static count on video
                    cv2.putText(overlay, f"Count: {count:.1f}", (25, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                    if count > CROWD_LIMIT:
                        cv2.putText(overlay, "!!! ALERT: OVERCROWDING !!!", (25, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        # winsound.Beep(1000, 500)
                        if SOUND_AVAILABLE:
                            winsound.Beep(1000, 400)  # Play sound only on Windows
                        else:
                            pass  # or print("Sound disabled in Docker")


                    display_frame = cv2.resize(overlay, (1200, 500))
                    stframe.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")

                    out.write(cv2.resize(overlay, (width, height)))

                cap.release()
                out.release()

                st.success(f"Video processed! Total frames: {frame_count}")

                with open(out_file, "rb") as f:
                    st.download_button("Download Processed Video", f, file_name="processed_video.mp4")


        # --------- MODE 2: LIVE WEBCAM ---------
        elif mode == "Live Webcam":
            st.write("### Live Crowd Monitoring (Webcam)")
            run = st.checkbox("Start Webcam", key="start_cam")

            if run:
                cap = cv2.VideoCapture(0)
                stframe = st.empty()

                fps = 0
                fps_start = time.time()
                frame_count = 0

                if not cap.isOpened():
                    st.error("Cannot open webcam.")
                else:
                
                    

                    # Load model
                    if "onnx_session" not in st.session_state:
                        try:
                            st.session_state["onnx_session"] = ort.InferenceSession("checkpoints/csrnet.onnx", providers=["CUDAExecutionProvider"])
                            st.toast("Running on GPU (CUDA)")
                        except Exception:
                            st.session_state["onnx_session"] = ort.InferenceSession("checkpoints/csrnet.onnx", providers=["CPUExecutionProvider"])
                            st.toast("Running on CPU")

                    session = st.session_state["onnx_session"]
                    input_name = session.get_inputs()[0].name

                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                    ])

                    CROWD_LIMIT = st.session_state.get("CROWD_LIMIT", 5)
                    frame_count = 0
                    fps = 0
                    fps_start = time.time()

                    # ---------- STEP 1: Dynamic Metric Boxes Setup ----------
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        crowd_box = st.empty()
                    with col2:
                        status_box = st.empty()
                    with col3:
                        fps_box = st.empty()

                    # Initial Values for boxes
                    crowd_box.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">0</div>
                            <div class="metric-label">Crowd Count</div>
                        </div>
                    """, unsafe_allow_html=True)

                    status_box.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">Normal</div>
                            <div class="metric-label">Status</div>
                        </div>
                    """, unsafe_allow_html=True)

                    fps_box.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">0.00</div>
                            <div class="metric-label">FPS</div>
                        </div>
                    """, unsafe_allow_html=True)
                    # --------------------------------------------------------

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1
                        if frame_count % 3 != 0:
                            continue

                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, _ = img_rgb.shape

                        img_resized = cv2.resize(img_rgb, (256, 256))
                        tensor = transform(img_resized).unsqueeze(0).numpy()

                        preds = session.run(None, {input_name: tensor})[0]
                        density_map = preds.squeeze(0).squeeze(0)
                        density_map = cv2.resize(density_map, (w, h))

                        # Smooth + normalize
                        density_blur = cv2.GaussianBlur(density_map, (15, 15), 0)
                        norm_map = cv2.normalize(density_blur, None, 0, 1, cv2.NORM_MINMAX)

                        thresh_val = 0.9
                        mask = (norm_map > thresh_val).astype(np.uint8)

                        labeled, num_features = label(mask)
                        objects = find_objects(labeled)

                        detected_heads = []
                        for obj in objects:
                            ys, xs = obj
                            y1, y2 = ys.start, ys.stop
                            x1, x2 = xs.start, xs.stop
                            area = (x2 - x1) * (y2 - y1)
                            if area > 150:
                                detected_heads.append((x1, y1, x2, y2))

                        # Merge overlapping boxes
                        merged_boxes = []
                        for box in detected_heads:
                            x1, y1, x2, y2 = box
                            merged = False
                            for mb in merged_boxes:
                                mx1, my1, mx2, my2 = mb
                                if not (x2 < mx1 or x1 > mx2 or y2 < my1 or y1 > my2):
                                    mb[:] = [min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2)]
                                    merged = True
                                    break
                            if not merged:
                                merged_boxes.append([x1, y1, x2, y2])

                        # Draw expanded green boxes
                        for (x1, y1, x2, y2) in merged_boxes:
                            box_w = x2 - x1
                            box_h = y2 - y1
                            expand_ratio = 14
                            x1_new = int(max(0, x1 - (expand_ratio - 1) * box_w / 2))
                            y1_new = int(max(0, y1 - (expand_ratio - 1) * box_h / 2))
                            x2_new = int(min(frame.shape[1], x2 + (expand_ratio - 1) * box_w / 2))
                            y2_new = int(min(frame.shape[0], y2 + (expand_ratio - 1) * box_h / 2))
                            cv2.rectangle(frame, (x1_new, y1_new), (x2_new, y2_new), (0, 255, 0), 3)

                        head_count = len(merged_boxes)

                        # ---------- Step 4: Email alert logic ----------
                        try:
                            if head_count > CROWD_LIMIT and st.session_state.get("email_cooldown", 0) == 0:
                                import threading
                                from email_alert import send_email_alert
                                threading.Thread(target=send_email_alert, args=(head_count,), daemon=True).start()
                                st.toast("📩 Email alert sent")
                                st.sidebar.success("📩 Email alert sent")
                                st.session_state["email_cooldown"] = 30 * int(fps) if fps and fps > 0 else 600
                            else:
                                if st.session_state.get("email_cooldown", 0) > 0:
                                    st.session_state["email_cooldown"] -= 1
                        except Exception as e:
                            print("Email alert error (webcam):", e)

                        # ---------- STEP 2: Update Dynamic Metric Boxes ----------
                        status_text = "⚠️ Overcrowded" if head_count > CROWD_LIMIT else " Normal"
                        status_color = "#ff4c4c" if head_count > CROWD_LIMIT else "#00ff99"

                        crowd_box.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-value">{head_count}</div>
                                <div class="metric-label">Crowd Count</div>
                            </div>
                        """, unsafe_allow_html=True)

                        status_box.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-value" style="color:{status_color};">{status_text}</div>
                                <div class="metric-label">Status</div>
                            </div>
                        """, unsafe_allow_html=True)

                        fps_box.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-value">{fps:.2f}</div>
                                <div class="metric-label">FPS</div>
                            </div>
                        """, unsafe_allow_html=True)
                        # --------------------------------------------------------

                        # FPS update
                        if frame_count % 10 == 0:
                            fps_end = time.time()
                            fps = 10 / (fps_end - fps_start)
                            fps_start = fps_end

                        # Transparent overlay text (no black box)
                        cv2.putText(frame, f"Count: {head_count}", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)

                        if head_count > CROWD_LIMIT:
                            cv2.putText(frame, "!!! ALERT: OVERCROWDING !!!", (20, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                            # winsound.Beep(1000, 400)
                            if SOUND_AVAILABLE:
                                winsound.Beep(1000, 400)  # Play sound only on Windows
                            else:
                                pass  # or print("Sound disabled in Docker")


                        # Resize frame for uniform display
                        display_frame = cv2.resize(frame, (1200, 500))

                        # Show in Streamlit
                        stframe.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")

                    cap.release()

        # --------- MODE 3: IMAGE ---------
        elif mode == "Image":
            uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="upload_image")

            if uploaded_image is not None:
                import io
                from PIL import Image

                # Read uploaded image
                img_pil = Image.open(uploaded_image).convert("RGB")
                img = np.array(img_pil)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                orig_h, orig_w = img_bgr.shape[:2]

                # --- Preprocess for CSRNet ---
                img_resized = cv2.resize(img, (256, 256))
                tensor = transform(img_resized).unsqueeze(0).numpy().astype(np.float32)

                # --- Inference with ONNX model ---
                preds = session.run(None, {input_name: tensor})[0]
                density_map = preds.squeeze(0).squeeze(0)

                # --- Resize back to original size ---
                density_resized = cv2.resize(density_map, (orig_w, orig_h))

                # ✅ Normalize and clip outliers for better visualization
                density_norm = np.clip(density_resized, 0, np.percentile(density_resized, 99))
                density_norm = density_norm / (density_norm.max() + 1e-8)

                # --- Convert to color heatmap ---
                density_uint8 = (density_norm * 255).astype(np.uint8)
                density_color = cv2.applyColorMap(density_uint8, cv2.COLORMAP_JET)

                # --- Overlay the density map on original image ---
                overlay = cv2.addWeighted(img_bgr, 0.6, density_color, 0.4, 0)

                # ✅ Compute realistic count
                # CSRNet outputs density per 1/8th pixel (depending on training scale)
                # Empirical correction: divide by 100–400 based on dataset
                raw_count = np.sum(density_resized)
                corrected_count = raw_count / 100.0   # try 80–200 range for tuning

                # ---------- Step 4: Email alert logic for Image ----------
                try:
                    if corrected_count > CROWD_LIMIT and st.session_state.get("email_cooldown", 0) == 0:
                        import threading
                        from email_alert import send_email_alert
                        threading.Thread(target=send_email_alert, args=(corrected_count,), daemon=True).start()
                        st.toast("📩 Email alert sent")
                        st.sidebar.success("📩 Email alert sent")
                        st.session_state["email_cooldown"] = 30
                    else:
                        if st.session_state.get("email_cooldown", 0) > 0:
                            st.session_state["email_cooldown"] -= 1
                except Exception as e:
                    print("Email alert error (image):", e)

                # --- Draw the count text ---
                cv2.putText(
                    overlay,
                    f"Count: {corrected_count:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # --- Display in Streamlit ---
                display_img = cv2.resize(overlay, (1200, 500))
                st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), channels="RGB")

       

# --------- ABOUT PAGE ----------
elif page == "about":
    st.title("About CrowdVision AI")

    st.markdown(
        """
        <div style="
            background-color: rgba(255, 255, 255, 0.05);
            padding: 30px;
            border-radius: 10px;
        ">
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "**CrowdVision AI** is a state-of-the-art real-time crowd monitoring system powered by deep learning (CSRNet) and optimized with ONNX. It provides accurate crowd counting, dynamic density heatmaps, and real-time alerts for overcrowding."
    )

    st.markdown("### Key Features:")
    st.markdown("""
    - Real-time crowd counting with high accuracy  
    - Dynamic density heatmaps for visual analysis  
    - Immediate alerts for overcrowding situations  
    - Video upload, live processing, and downloadable results  
    - Customizable settings like crowd limit and video scale
    """)

    st.markdown(
        "Designed for event organizers, security teams, and public safety monitoring, CrowdVision AI provides actionable insights to manage crowds efficiently."
    )

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------
# Show Footer (fixed at bottom)
# -------------------------------
show_footer()
