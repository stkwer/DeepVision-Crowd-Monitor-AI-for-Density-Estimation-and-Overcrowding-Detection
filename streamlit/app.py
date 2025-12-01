import base64
import hashlib
import json
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from string import Template
import bcrypt
import re

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO
import torch

st.set_page_config(page_title="People Counter AI", page_icon="👥", layout="wide")

# state for alert triggering
if 'alert_triggered' not in st.session_state:
    st.session_state['alert_triggered'] = False

import threading
try:
    from playsound import playsound
    _HAS_PLAYSOUND = True
except Exception:
    _HAS_PLAYSOUND = False

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "yolov8peoplecounter" / "yolov8s.pt"
COCO_PATH = REPO_ROOT / "yolov8peoplecounter" / "coco.txt"
AUTH_DB_PATH = REPO_ROOT / "streamlit" / "auth.db"

# Session configuration
SESSION_TIMEOUT_HOURS = 24

MODERN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --primary-light: #818cf8;
    --secondary: #8b5cf6;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --dark: #1e293b;
    --darker: #0f172a;
    --light: #f8fafc;
    --gray: #64748b;
}

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.main-header {
    text-align: center;
    padding: 2rem 0 3rem 0;
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.main-header h1 {
    font-size: 3rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(135deg, #fff 0%, #e0e7ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.main-header p {
    font-size: 1.1rem;
    margin-top: 0.5rem;
    color: rgba(255, 255, 255, 0.8);
}

.control-panel {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.video-container {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    margin-bottom: 2rem;
}

.video-container video {
    border-radius: 12px;
    width: 100%;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
}

.stFileUploader {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1rem;
    border: 2px dashed rgba(255, 255, 255, 0.3);
}

.stFileUploader:hover {
    border-color: rgba(255, 255, 255, 0.5);
    background: rgba(255, 255, 255, 0.12);
}

.stFileUploader label {
    color: white !important;
    font-weight: 600;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-top: 2rem;
}

.stat-card {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}

.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
}

.stat-label {
    font-size: 0.9rem;
    color: rgba(255, 255, 255, 0.7);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

.stat-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: white;
    margin-top: 0.5rem;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
}

.logout-button {
    background: rgba(239, 68, 68, 0.1) !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    color: #ef4444 !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1rem !important;
    border-radius: 8px !important;
    transition: all 0.3s !important;
}

.logout-button:hover {
    background: rgba(239, 68, 68, 0.2) !important;
    border-color: rgba(239, 68, 68, 0.5) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3) !important;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
}

.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.6);
}

.stRadio > div {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1rem;
}

.stRadio label {
    color: white !important;
    font-weight: 600;
}

.stCheckbox label {
    color: white !important;
    font-weight: 600;
}

.stSlider > div > div > div {
    background: rgba(255, 255, 255, 0.3);
}

.stSlider > div > div > div > div {
    background: white;
}

.stSlider label {
    color: white !important;
    font-weight: 600;
}

.alert-box {
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
    backdrop-filter: blur(10px);
}

.alert-success {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid rgba(16, 185, 129, 0.5);
}

.alert-warning {
    background: rgba(245, 158, 11, 0.2);
    border: 1px solid rgba(245, 158, 11, 0.5);
}

.alert-danger {
    background: rgba(239, 68, 68, 0.2);
    border: 1px solid rgba(239, 68, 68, 0.5);
}

.alert-icon {
    font-size: 2rem;
}

.placeholder-box {
    background: rgba(255, 255, 255, 0.05);
    border: 2px dashed rgba(255, 255, 255, 0.3);
    border-radius: 16px;
    padding: 3rem;
    text-align: center;
    color: rgba(255, 255, 255, 0.7);
}

.placeholder-box h3 {
    color: white;
    margin-bottom: 1rem;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

st.markdown(MODERN_CSS, unsafe_allow_html=True)


def ensure_auth_db() -> None:
    with sqlite3.connect(str(AUTH_DB_PATH)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email.strip()) is not None


def validate_username(username: str) -> Tuple[bool, str]:
    """Validate username and return (is_valid, error_message)"""
    username = username.strip()
    if not username:
        return False, "Username cannot be empty"
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
    if len(username) > 30:
        return False, "Username must be less than 30 characters"
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        return False, "Username can only contain letters, numbers, underscores, and hyphens"
    return True, ""


def validate_password(password: str) -> Tuple[bool, str]:
    """Validate password strength and return (is_valid, error_message)"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter (A-Z)"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter (a-z)"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number (0-9)"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character (!@#$%^&*)"
    return True, ""


def check_rate_limit(identifier: str) -> Tuple[bool, int]:
    """Check if login attempts are within rate limit. Returns (allowed, remaining_attempts)"""
    max_attempts = 5
    lockout_minutes = 15

    # Initialize rate limiting state if not exists
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = {}
    if 'lockout_until' not in st.session_state:
        st.session_state.lockout_until = {}

    current_time = time.time()

    # Check if user is currently locked out
    if identifier in st.session_state.lockout_until:
        if current_time < st.session_state.lockout_until[identifier]:
            remaining_time = int((st.session_state.lockout_until[identifier] - current_time) / 60)
            return False, remaining_time
        else:
            # Lockout period expired, reset
            del st.session_state.lockout_until[identifier]
            if identifier in st.session_state.login_attempts:
                del st.session_state.login_attempts[identifier]

    # Get current attempt count
    attempts = st.session_state.login_attempts.get(identifier, 0)
    remaining_attempts = max_attempts - attempts

    if attempts >= max_attempts:
        # Lock out the user
        st.session_state.lockout_until[identifier] = current_time + (lockout_minutes * 60)
        return False, lockout_minutes

    return True, remaining_attempts


def record_failed_attempt(identifier: str) -> None:
    """Record a failed login attempt"""
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = {}

    st.session_state.login_attempts[identifier] = st.session_state.login_attempts.get(identifier, 0) + 1


def reset_login_attempts(identifier: str) -> None:
    """Reset login attempts on successful login"""
    if 'login_attempts' in st.session_state and identifier in st.session_state.login_attempts:
        del st.session_state.login_attempts[identifier]
    if 'lockout_until' in st.session_state and identifier in st.session_state.lockout_until:
        del st.session_state.lockout_until[identifier]


def user_exists(username: str, email: str) -> bool:
    normalized_username = username.strip().lower()
    normalized_email = email.strip().lower()
    with sqlite3.connect(str(AUTH_DB_PATH)) as conn:
        row = conn.execute(
            "SELECT 1 FROM users WHERE LOWER(username)=? OR LOWER(email)=?",
            (normalized_username, normalized_email),
        ).fetchone()
    return row is not None


def user_exists_by_username(username: str) -> bool:
    normalized_username = username.strip().lower()
    with sqlite3.connect(str(AUTH_DB_PATH)) as conn:
        row = conn.execute(
            "SELECT 1 FROM users WHERE LOWER(username)=?",
            (normalized_username,),
        ).fetchone()
    return row is not None


def create_jwt_token(user: str) -> str:
    """Create a JWT token for the user session using base64 encoding"""
    payload = {
        "user": user,
        "iat": int(time.time()),
        "exp": int(time.time()) + (SESSION_TIMEOUT_HOURS * 3600)
    }
    # Use base64 encoding for token (simple but effective)
    token_str = json.dumps(payload)
    token = base64.b64encode(token_str.encode()).decode('utf-8')
    return token

def save_persistent_auth(authenticated: bool, user: str = None) -> None:
    """Save authentication state using JWT token in URL parameters"""
    if authenticated and user:
        token = create_jwt_token(user)

        # Update URL with JWT token
        js_code = f"""
        <script>
        try {{
            const url = new URL(window.location.href);
            url.searchParams.set('auth_token', '{token}');
            window.history.replaceState(null, null, url.toString());
        }} catch (e) {{
            console.warn('Failed to update URL with auth token:', e);
        }}
        </script>
        """
        st.markdown(js_code, unsafe_allow_html=True)


def verify_jwt_token(token: str) -> Optional[str]:
    """Verify JWT token and return username if valid using base64 decoding"""
    try:
        # Decode base64 token
        token_bytes = base64.b64decode(token)
        payload = json.loads(token_bytes.decode('utf-8'))
        
        # Check expiration
        if payload.get("exp", 0) > time.time():
            user = payload.get("user")
            if user and user_exists_by_username(user):
                return user
    except Exception:
        pass  # Token invalid or expired
    return None

def load_persistent_auth() -> dict:
    """Load authentication state from JWT token in URL parameters"""
    try:
        # Get auth token from URL parameters
        query_params = st.query_params
        token = query_params.get('auth_token')

        if token:
            user = verify_jwt_token(token)
            if user:
                return {
                    "authenticated": True,
                    "user": user,
                    "token": token
                }
    except Exception as e:
        # Silently fail for auth loading errors
        pass

    return None


def clear_persistent_auth() -> None:
    """Clear authentication state from URL parameters"""
    js_code = """
    <script>
    try {
        const url = new URL(window.location.href);
        url.searchParams.delete('auth_token');
        url.searchParams.delete('user');
        window.history.replaceState(null, null, url.toString());
    } catch (e) {
        console.warn('Failed to clear auth from URL:', e);
    }
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)


def create_user(username: str, email: str, password: str) -> Optional[str]:
    cleaned_username = username.strip()
    cleaned_email = email.strip().lower()
    hashed = hash_password(password)
    try:
        with sqlite3.connect(str(AUTH_DB_PATH)) as conn:
            conn.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (cleaned_username, cleaned_email, hashed),
            )
            conn.commit()
        return cleaned_username
    except sqlite3.IntegrityError:
        return None


def authenticate_credentials(identifier: str, password: str) -> Optional[str]:
    normalized_identifier = identifier.strip().lower()
    if not normalized_identifier or not password:
        return None

    with sqlite3.connect(str(AUTH_DB_PATH)) as conn:
        row = conn.execute(
            "SELECT username, password FROM users WHERE LOWER(username)=? OR LOWER(email)=?",
            (normalized_identifier, normalized_identifier),
        ).fetchone()

    if row:
        username, stored_hash = row
        password_encoded = password.encode('utf-8')
        stored_hash_encoded = stored_hash.encode('utf-8')
        
        # Try bcrypt first (new format)
        try:
            if bcrypt.checkpw(password_encoded, stored_hash_encoded):
                return username
        except (ValueError, TypeError):
            # Not a valid bcrypt hash, try legacy SHA256 format
            try:
                legacy_hash = hashlib.sha256(password_encoded).hexdigest()
                if legacy_hash == stored_hash:
                    # Password matches legacy hash - upgrade to bcrypt
                    new_hash = hash_password(password)
                    with sqlite3.connect(str(AUTH_DB_PATH)) as conn:
                        conn.execute("UPDATE users SET password=? WHERE username=?", (new_hash, username))
                        conn.commit()
                    return username
            except Exception:
                pass
    
    return None


@st.cache_resource(show_spinner=False)
def load_model(model_path: Path) -> YOLO:
    return YOLO(str(model_path))


@st.cache_resource(show_spinner=False)
def get_inference_settings() -> Dict[str, object]:
    """Return a dict with device, half precision, and default imgsz for inference."""
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = 0
            use_half = True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            use_half = False
        else:
            device = 'cpu'
            use_half = False
    except Exception:
        device = 'cpu'
        use_half = False

    return {"device": device, "half": use_half, "imgsz": 480}


@st.cache_data(show_spinner=False)
def load_class_names(labels_path: Path) -> List[str]:
    with labels_path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


@st.cache_data(show_spinner=False)
def encode_video_base64(video_hash: str, video_bytes: bytes) -> str:
    return base64.b64encode(video_bytes).decode("utf-8")


@st.cache_data(show_spinner=False)
def encode_beep_audio(beep_path: str) -> str:
    with open(beep_path, "rb") as bf:
        return base64.b64encode(bf.read()).decode("utf-8")


def detect_people(
    frame: np.ndarray,
    model: YOLO,
    class_names: List[str],
    imgsz: Optional[int] = None,
    device: Optional[object] = None,
    half: Optional[bool] = None,
) -> List[Tuple[int, int, int, int]]:
    """Detect persons in a frame.

    Args:
        imgsz: target width for resizing inference. If None, will use default from `get_inference_settings`.
    """
    detections: List[Tuple[int, int, int, int]] = []

    # Determine device and precision
    settings = get_inference_settings()
    device = device if device is not None else settings.get("device")
    half = half if half is not None else settings.get("half")
    imgsz = imgsz if imgsz is not None else settings.get("imgsz")

    h, w = frame.shape[:2]

    # Resize input for faster processing when imgsz is provided
    if imgsz and imgsz > 0:
        new_w = imgsz
        new_h = max(1, int(imgsz * h / w))
        interpolation = cv2.INTER_AREA if new_w < w else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
        results = model.predict(resized, conf=0.15, iou=0.3, verbose=False, half=half, device=device, imgsz=new_w)
    else:
        resized = frame
        results = model.predict(frame, conf=0.15, iou=0.3, verbose=False, half=half, device=device)
    # Choose device and half precision appropriately
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = 0
            use_half = True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple MPS device
            device = 'mps'
            use_half = False
        else:
            device = 'cpu'
            use_half = False
    except Exception:
        # Fallback to CPU if torch backend check fails
        device = 'cpu'
        use_half = False

    results = model.predict(resized, conf=0.15, iou=0.3, verbose=False, half=use_half, device=device)

    if not results:
        return detections

    scale_x = w / 480
    scale_x = w / resized.shape[1]
    scale_y = h / resized.shape[0]
    
    for box in results[0].boxes:
        cls_id = int(box.cls)
        if 0 <= cls_id < len(class_names) and class_names[cls_id] == "person":
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append((int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)))

    return detections


def apply_alert_overlay(frame: np.ndarray, people_count: int, threshold: int, triggered: bool) -> None:
    if threshold <= 0:
        return

    height, width = frame.shape[:2]
    banner_height = max(int(height * 0.12), 80)
    overlay = frame.copy()
    
    # Color based on alert status
    if triggered:
        color = (68, 68, 239)  # Red in BGR
    else:
        color = (129, 185, 16)  # Green in BGR

    cv2.rectangle(overlay, (0, 0), (width, banner_height), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    title = "ALERT: Threshold Exceeded!" if triggered else "Normal Count"
    detail = f"People: {people_count} / {threshold}"

    cv2.putText(
        frame,
        title,
        (30, int(banner_height * 0.4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        detail,
        (30, int(banner_height * 0.75)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def annotate_frame(
    frame: np.ndarray,
    model: YOLO,
    class_names: List[str],
    show_alert_overlay: bool = False,
    threshold: int = 0,
    imgsz: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    annotated = frame.copy()
    # Use provided imgsz or fallback to default settings
    person_boxes = detect_people(annotated, model, class_names, imgsz=imgsz)

    # Draw bounding boxes
    for (x1, y1, x2, y2) in person_boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            annotated,
            "PERSON",
            (x1, max(y1 - 15, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    people_count = len(person_boxes)

    # Apply alert overlay if enabled
    banner_height = 0
    if show_alert_overlay:
        apply_alert_overlay(annotated, people_count, threshold, people_count >= threshold)
        banner_height = max(int(annotated.shape[0] * 0.12), 80)

    count_y = banner_height + 50 if banner_height else 50

    cv2.putText(
        annotated,
        f"COUNT: {people_count}",
        (30, count_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated, people_count


def process_video(
    input_path: Path,
    model: YOLO,
    class_names: List[str],
    threshold: int,
    show_alerts: bool,
    frame_skip: int = 3,
    imgsz: int = 480,
) -> Tuple[Path, Dict[str, float], List[float]]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open the uploaded video file.")

    try:
        return process_video_capture(cap, model, class_names, threshold, show_alerts, frame_skip=frame_skip, imgsz=imgsz)
    finally:
        cap.release()


def check_video_format(input_path: Path) -> bool:
    if shutil.which("ffprobe") is None:
        return False
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt", "-of", "default=noprint_wrappers=1", str(input_path)],
            capture_output=True,
            text=True,
            timeout=5
        )
        output = result.stdout.lower()
        return "h264" in output and "yuv420p" in output
    except Exception:
        return False


def transcode_with_ffmpeg(input_path: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg and ensure it is in PATH.")
    output_path = Path(tempfile.gettempdir()) / f"people_counter_transcoded_{uuid.uuid4().hex}.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg transcoding failed.")
    return output_path


def process_video_capture(
    cap: cv2.VideoCapture,
    model: YOLO,
    class_names: List[str],
    threshold: int,
    show_alerts: bool,
    frame_skip: int = 3,
    imgsz: int = 480,
) -> Tuple[Path, Dict[str, float], List[float]]:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 1e-2:
        fps = 24.0

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("The uploaded video contains no readable frames.")

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = Path(tempfile.gettempdir()) / f"people_counter_output_{uuid.uuid4().hex}.mp4"
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError("Failed to initialize the video writer.")

    frame_counts: List[int] = []
    alert_times: List[float] = []
    frame_index = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    prev_above = False
    
    # Fast processing configuration is provided by args
    if frame_skip < 1:
        frame_skip = 1
    target_width = imgsz
    last_count = 0
    last_annotated_frame = None

    while ret:
        frame_index += 1
        
        if frame_index % frame_skip == 0:
            annotated_frame, count = annotate_frame(
                frame,
                model,
                class_names,
                show_alert_overlay=show_alerts,
                threshold=threshold,
                imgsz=target_width,
            )
            last_count = count
            last_annotated_frame = annotated_frame
        else:
            # Use last annotated frame for skipped frames
            if last_annotated_frame is not None:
                annotated_frame = last_annotated_frame
            else:
                annotated_frame = frame.copy()
            count = last_count
        
        frame_counts.append(count)
        if threshold > 0:
            current_time = frame_index / float(fps)
            if count >= threshold and not prev_above:
                alert_times.append(current_time)
                prev_above = True
            elif count < threshold:
                prev_above = False
        writer.write(annotated_frame)

        if total_frames:
            progress = min(frame_index / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"⚙️ Processing: {int(progress*100)}%")

        ret, frame = cap.read()

    progress_bar.empty()
    status_text.empty()
    writer.release()

    if check_video_format(output_path):
        final_output_path = output_path
    else:
        final_output_path = transcode_with_ffmpeg(output_path)
        output_path.unlink(missing_ok=True)

    summary: Dict[str, float] = {
        "frames": float(frame_index),
        "max": float(max(frame_counts)) if frame_counts else 0.0,
        "average": float(np.mean(frame_counts)) if frame_counts else 0.0,
    }

    return final_output_path, summary, alert_times


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_path = Path(tempfile.gettempdir()) / f"people_counter_upload_{uuid.uuid4().hex}{suffix}"
    with temp_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())
    return temp_path


def render_stats(summary: Dict[str, float], threshold: int) -> None:
    frames = int(summary.get("frames", 0))
    peak = int(summary.get("max", 0))
    average = summary.get("average", 0.0)
    
    st.markdown("""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Frames</div>
                <div class="stat-value">""" + str(frames) + """</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Peak Count</div>
                <div class="stat-value">""" + str(peak) + """</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average</div>
                <div class="stat-value">""" + f"{average:.1f}" + """</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Threshold</div>
                <div class="stat-value">""" + str(threshold) + """</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Alert status
    if peak >= threshold:
        st.markdown(f"""
            <div class="alert-box alert-danger">
                <div class="alert-icon">🚨</div>
                <div>
                    <strong>Alert Triggered!</strong><br>
                    Peak count ({peak}) exceeded threshold ({threshold})
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="alert-box alert-success">
                <div class="alert-icon">✅</div>
                <div>
                    <strong>All Clear</strong><br>
                    Peak count ({peak}) within threshold ({threshold})
                </div>
            </div>
        """, unsafe_allow_html=True)


def process_uploaded_video(
    model: YOLO,
    class_names: List[str],
    show_alerts: bool,
    threshold: int,
    frame_skip: int = 3,
    imgsz: int = 480,
) -> None:
    st.markdown("### 📹 Video Processing")
    
    uploaded_file = st.file_uploader(
        "Upload your video file",
        type=["mp4", "mov", "avi", "mkv"],
        help="Supported formats: MP4, MOV, AVI, MKV",
    )

    if uploaded_file is None:
        st.info("👆 Select a video file to begin analysis")
        return

    try:
        input_path = save_uploaded_file(uploaded_file)
    except Exception as exc:
        st.error(f"❌ Failed to save file: {exc}")
        st.stop()

    with st.spinner("🔄 Analyzing video..."):
        try:
            output_path, summary, alert_times = process_video(
                input_path,
                model,
                class_names,
                threshold=threshold,
                show_alerts=show_alerts,
                frame_skip=frame_skip,
                imgsz=imgsz,
            )
        except Exception as exc:
            st.error(f"❌ Processing failed: {exc}")
            input_path.unlink(missing_ok=True)
            st.stop()

    # Read processed video
    with output_path.open("rb") as file:
        video_bytes = file.read()

    # Display video and download button
    col1, col2 = st.columns([5, 1])

    with col1:
        if len(video_bytes) == 0:
            st.error("❌ Processed video is empty. Please try with a different input.")
        else:
            video_hash = hashlib.md5(video_bytes).hexdigest()
            b64_video = encode_video_base64(video_hash, video_bytes)
            serialized_video = json.dumps(b64_video)
            serialized_alerts = json.dumps(alert_times)
            beep_path = REPO_ROOT / "beep.mp3"
            serialized_beep = 'null'
            if beep_path.exists():
                b64_beep = encode_beep_audio(str(beep_path))
                serialized_beep = json.dumps(b64_beep)

            audio_flag = json.dumps(True)
            video_id = f"processed-video-{uuid.uuid4().hex}"
            video_template = Template("""
                <div class="video-container">
                    <video id="$video_id" width="100%" height="auto" controls muted playsinline autoplay preload="auto"></video>
                </div>
                <script>
                    (function() {
                        const videoElement = document.getElementById('$video_id');
                        if (!videoElement) return;

                        const base64String = $serialized_video;
                        const alertTimes = $serialized_alerts;
                        const beepBase64 = $serialized_beep;
                        const audioEnabled = $audio_flag;
                        const byteCharacters = atob(base64String);
                        const byteLength = byteCharacters.length;
                        const sliceSize = 1024 * 256;
                        const byteArrays = [];

                        for (let offset = 0; offset < byteLength; offset += sliceSize) {
                            const slice = byteCharacters.slice(offset, offset + sliceSize);
                            const sliceLength = slice.length;
                            const byteNumbers = new Array(sliceLength);
                            for (let i = 0; i < sliceLength; i++) {
                                byteNumbers[i] = slice.charCodeAt(i);
                            }
                            byteArrays.push(new Uint8Array(byteNumbers));
                        }

                        const blob = new Blob(byteArrays, { type: 'video/mp4' });
                        const objectUrl = URL.createObjectURL(blob);
                        videoElement.src = objectUrl;
                        videoElement.load();

                        const attemptPlay = () => {
                            const playPromise = videoElement.play();
                            if (playPromise !== undefined) {
                                playPromise.catch(() => {});
                            }
                        };

                        videoElement.addEventListener('loadeddata', attemptPlay, { once: true });
                        attemptPlay();

                        let beepUrl = null;
                        if (beepBase64 && beepBase64 !== null && beepBase64 !== 'null') {
                            try {
                                const byteCharactersB = atob(beepBase64);
                                const byteNumbersB = new Array(byteCharactersB.length);
                                for (let i = 0; i < byteCharactersB.length; i++) {
                                    byteNumbersB[i] = byteCharactersB.charCodeAt(i);
                                }
                                const beepArray = new Uint8Array(byteNumbersB);
                                const beepBlob = new Blob([beepArray], { type: 'audio/mpeg' });
                                beepUrl = URL.createObjectURL(beepBlob);
                            } catch (e) {
                                console.warn('Failed to prepare beep audio', e);
                                beepUrl = null;
                            }
                        }

                        try {
                            let lastTriggered = -1;
                            const beepPlayer = () => {
                                if (!beepUrl) return;
                                try {
                                    const audio = new Audio(beepUrl);
                                    audio.currentTime = 0;
                                    const playPromise = audio.play();
                                    if (playPromise !== undefined) {
                                        playPromise.catch(() => {});
                                    }
                                } catch (e) {
                                    console.warn('Beep play failed', e);
                                }
                            };

                            videoElement.addEventListener('timeupdate', () => {
                                const t = videoElement.currentTime;
                                for (let i = 0; i < alertTimes.length; i++) {
                                    if (Math.abs(t - alertTimes[i]) < 0.2 && lastTriggered !== i) {
                                        beepPlayer();
                                        lastTriggered = i;
                                        break;
                                    }
                                }
                            });
                        } catch (e) {
                            console.warn('Beep setup failed', e);
                        }

                        const cleanup = () => {
                            URL.revokeObjectURL(objectUrl);
                            if (beepUrl) URL.revokeObjectURL(beepUrl);
                        };

                        window.addEventListener('pagehide', cleanup);
                        window.addEventListener('beforeunload', cleanup);
                    })();
                </script>
            """
            )
            components.html(
                video_template.substitute(
                    video_id=video_id,
                    serialized_video=serialized_video,
                    serialized_alerts=serialized_alerts,
                    serialized_beep=serialized_beep,
                    audio_flag=audio_flag,
                ),
                height=520,
                scrolling=False,
            )

    with col2:
        st.download_button(
            label="⬇️ Download",
            data=video_bytes,
            file_name=f"processed_{Path(uploaded_file.name).stem}.mp4",
            mime="video/mp4",
        )

    # Display statistics
    render_stats(summary, threshold)

    # Cleanup
    input_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)


def process_webcam_stream(
    model: YOLO,
    class_names: List[str],
    threshold: int,
    show_alerts: bool,
    imgsz: int = 640,
) -> None:
    # Initialize webcam session state
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = True
    
    st.info("📷 Camera access requested - please allow camera permission in your browser")

    container = st.empty()
    stats_container = st.empty()
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("⏹️ Stop Stream", type="secondary", use_container_width=True):
            st.session_state.webcam_running = False
            st.rerun()
    
    with col2:
        st.write("")  # Spacer

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Unable to access webcam. Please check permissions and ensure no other app is using the camera.")
        st.session_state.webcam_running = False
        return

    frame_count = 0
    try:
        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Lost connection to webcam.")
                break

            # Optionally scale the frame for faster inference
            if imgsz and imgsz > 0:
                new_w = imgsz
                new_h = int(imgsz * frame.shape[0] / frame.shape[1])
                frame = cv2.resize(frame, (new_w, new_h))
            else:
                # keep high res for accurate mode
                frame = cv2.resize(frame, (1280, 720))
            annotated_frame, people_count = annotate_frame(
                frame,
                model,
                class_names,
                show_alert_overlay=show_alerts,
                threshold=threshold,
                imgsz=imgsz,
            )

            # Display frame
            container.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                width=500,
            )
            
            # Show status and play beep when threshold is reached
            if show_alerts and threshold > 0:
                if people_count >= threshold:
                    if not st.session_state.get('alert_triggered', False):
                        if _HAS_PLAYSOUND:
                            try:
                                beep_path = Path(r"C:\Users\saura\People_counter\beep.mp3")
                                if beep_path.exists():
                                    threading.Thread(target=lambda: playsound(str(beep_path)), daemon=True).start()
                            except Exception:
                                pass
                        st.session_state['alert_triggered'] = True
                    stats_container.markdown(f"""
                        <div class="alert-box alert-danger">
                            <div class="alert-icon">🚨</div>
                            <div><strong>Alert!</strong> {people_count} people detected</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.session_state['alert_triggered'] = False
                    stats_container.markdown(f"""
                        <div class="alert-box alert-success">
                            <div class="alert-icon">✅</div>
                            <div><strong>Normal</strong> {people_count} people detected</div>
                        </div>
                    """, unsafe_allow_html=True)

            frame_count += 1
            if frame_count % 10 == 0:  # Refresh every 10 frames to allow button clicks
                time.sleep(0.03)
    finally:
        cap.release()
        st.session_state.webcam_running = False


def render_people_counter_app() -> None:
    # Header with logout button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("""
            <div class="main-header">
                <h1>👥 AI People Counter</h1>
                <p>Real-time crowd monitoring with YOLOv8 computer vision</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="text-align: right; padding-top: 1rem;">', unsafe_allow_html=True)
        if st.button("🚪 Logout", key="header_logout", help="Sign out of your account"):
            st.session_state["authenticated"] = False
            st.session_state["user"] = None
            st.session_state["session_start"] = None
            clear_persistent_auth()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Apply custom styling to the logout button
        st.markdown("""
        <style>
        div[data-testid="column"]:has(button[key="header_logout"]) button {
            background: rgba(239, 68, 68, 0.1) !important;
            border: 1px solid rgba(239, 68, 68, 0.3) !important;
            color: #ef4444 !important;
            font-size: 0.9rem !important;
            padding: 0.5rem 1rem !important;
            border-radius: 8px !important;
            transition: all 0.3s !important;
        }
        div[data-testid="column"]:has(button[key="header_logout"]) button:hover {
            background: rgba(239, 68, 68, 0.2) !important;
            border-color: rgba(239, 68, 68, 0.5) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3) !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # Check for required files
    if not MODEL_PATH.exists():
        st.error("❌ Model file not found. Please place 'yolov8s.pt' in the 'yolov8peoplecounter' folder.")
        st.stop()

    if not COCO_PATH.exists():
        st.error("❌ COCO labels file not found. Please ensure 'coco.txt' is in 'yolov8peoplecounter' folder.")
        st.stop()

    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model = load_model(MODEL_PATH)
        class_names = load_class_names(COCO_PATH)

    # Triage device information for users
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            st.success(f"✅ CUDA GPU detected (devices={torch.cuda.device_count()}). Using GPU accelerated inference.")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            st.info("⚡ Apple MPS device detected. Using MPS for inference.")
        else:
            # No GPU detected — run on CPU without showing a prominent warning.
            # We intentionally do not show a warning message here to avoid cluttering the UI.
            pass
    except Exception:
        st.info("⚠️ Unable to detect GPU status; defaulting to CPU.\nIf you have a GPU, please ensure your environment has CUDA installed and visible.")

    # Control Panel - Clean Production UI
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1], gap="medium")
    
    with col1:
        mode = st.radio(
            "📊 Input Source",
            ["Upload Video", "Live Webcam"],
            horizontal=True,
        )
    
    with col2:
        show_alerts = st.checkbox("🔔 Enable Alerts", value=True)

    # Performance mode: affects frame skipping and inference image size
    perf_option = st.selectbox(
        "⚡ Performance Mode",
        ["Balanced (default)", "Fast", "Accurate"],
        index=0,
        help="Choose a tradeoff between speed and accuracy. Fast uses more skipping and smaller images.",
    )
    
    with col3:
        threshold = st.slider("Alert Threshold", min_value=1, max_value=100, value=10, label_visibility="collapsed")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Process based on mode
    if mode == "Upload Video":
        # Map UI perf option to frame_skip + imgsz
        if perf_option == "Fast":
            ui_frame_skip, ui_imgsz = 5, 320
        elif perf_option == "Accurate":
            ui_frame_skip, ui_imgsz = 1, 0
        else:
            ui_frame_skip, ui_imgsz = 3, 480

        process_uploaded_video(
            model,
            class_names,
            show_alerts=show_alerts,
            threshold=threshold,
            frame_skip=ui_frame_skip,
            imgsz=ui_imgsz,
        )
    else:
        # Initialize webcam state if not already done
        if 'webcam_running' not in st.session_state:
            st.session_state.webcam_running = False
        
        if st.button("▶️ Start Live Webcam", type="primary", use_container_width=False):
            st.session_state.webcam_running = True
            # Map UI perf option to webcam scaling
            if perf_option == "Fast":
                ui_cam_imgsz = 320
            elif perf_option == "Accurate":
                ui_cam_imgsz = 0
            else:
                ui_cam_imgsz = 640
            process_webcam_stream(model, class_names, threshold, show_alerts, imgsz=ui_cam_imgsz)


def initialize_auth_state() -> None:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "session_start" not in st.session_state:
        st.session_state["session_start"] = None

    # Check for session timeout
    if st.session_state["authenticated"] and st.session_state["session_start"]:
        current_time = time.time()
        session_duration = current_time - st.session_state["session_start"]
        if session_duration > (SESSION_TIMEOUT_HOURS * 3600):
            # Session expired
            st.session_state["authenticated"] = False
            st.session_state["user"] = None
            st.session_state["session_start"] = None
            clear_persistent_auth()
            st.warning("Your session has expired. Please log in again.")
            st.rerun()

    # Load persistent authentication from URL parameters
    auth_data = load_persistent_auth()
    if auth_data and auth_data.get("authenticated") and auth_data.get("user"):
        # Verify the stored credentials are still valid
        username = auth_data.get("user")
        if username and user_exists_by_username(username):
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            # Set session start time if not set
            if not st.session_state["session_start"]:
                st.session_state["session_start"] = time.time()
            # Re-save to maintain URL state
            save_persistent_auth(True, username)


def render_authentication() -> None:
    st.markdown("""
        <div class="main-header">
            <h1>👥 AI People Counter</h1>
            <p>Secure Access Required</p>
        </div>
    """, unsafe_allow_html=True)

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        st.subheader("Sign In")
        with st.form("login_form"):
            identifier = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Sign In", use_container_width=True)

        if submit_login:
            allowed, remaining = check_rate_limit(identifier)
            if not allowed:
                if remaining > 5:
                    st.error(f"Too many failed attempts. Try again in {remaining} minutes.")
                else:
                    st.error(f"Too many failed attempts. {remaining} attempts remaining.")
                st.stop()

            if not identifier or not password:
                st.error("Please enter both username/email and password.")
                record_failed_attempt(identifier)
            else:
                username = authenticate_credentials(identifier, password)
                if username:
                    reset_login_attempts(identifier)
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = username
                    st.session_state["session_start"] = time.time()
                    save_persistent_auth(True, username)
                    st.success(f"Welcome, {username}!")
                    st.rerun()
                else:
                    record_failed_attempt(identifier)
                    allowed, remaining = check_rate_limit(identifier)
                    if allowed:
                        st.error(f"Invalid credentials. {remaining} attempts remaining.")
                    else:
                        st.error(f"Too many failed attempts. Try again in {remaining} minutes.")

    with register_tab:
        st.subheader("Create Account")
        st.markdown("""
        **Password Requirements:**
        - Minimum 8 characters
        - Uppercase letter (A-Z)
        - Lowercase letter (a-z)
        - Number (0-9)
        - Special character (!@#$%^&*)
        """)
        
        with st.form("register_form"):
            username = st.text_input("Username", help="3-30 characters, letters/numbers/-/_")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_register = st.form_submit_button("Create Account", use_container_width=True)

        if submit_register:
            errors = []

            username_valid, username_error = validate_username(username)
            if not username_valid:
                errors.append(username_error)

            if not validate_email(email):
                errors.append("Invalid email format.")

            password_valid, password_error = validate_password(password)
            if not password_valid:
                errors.append(password_error)

            if password != confirm_password:
                errors.append("Passwords do not match.")

            if user_exists(username, email):
                errors.append("Username or email already in use.")

            if errors:
                for error in errors:
                    st.error(error)
            else:
                created_username = create_user(username, email, password)
                if created_username:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = created_username
                    st.session_state["session_start"] = time.time()
                    save_persistent_auth(True, created_username)
                    st.success("Account created successfully!")
                    st.rerun()
                else:
                    st.error("Unable to create account. Please try again.")


def render_user_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state['user']}")
        st.divider()
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["user"] = None
            st.session_state["session_start"] = None
            clear_persistent_auth()
            st.rerun()


def main() -> None:
    initialize_auth_state()
    ensure_auth_db()
    if not st.session_state["authenticated"]:
        render_authentication()
        return
    render_user_sidebar()
    render_people_counter_app()



if __name__ == "__main__":
    main()