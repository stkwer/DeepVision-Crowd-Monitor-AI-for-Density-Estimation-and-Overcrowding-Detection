# 👥 AI People Counter 🚀

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat-square&logo=streamlit)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-00BFFF?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)

### ✨ Real-time AI-Powered People Detection & Counting with Secure Authentication ✨

**Transform your videos and live feeds into intelligent people counting solutions with enterprise-grade security.**

[🎯 Quick Start](#-quick-start) • [📖 Documentation](#-features) • [🔐 Security](#-security-features) • [⚙️ Configuration](#-configuration--tuning)

</div>

---

## 🎯 Quick Start

### 🚀 Get Running in 3 Steps

```bash
# 1️⃣ Install dependencies
pip install -r requirements.txt

# 2️⃣ Launch the application
cd streamlit
streamlit run app.py

# 3️⃣ Open browser (automatically opens at http://localhost:8501)
```

**First time?** Create an account, then start counting! 👤➡️👥

---

## 📋 Features

<table>
<tr>
<td width="50%">

### 🔐 **Authentication & Security**
- ✅ Secure login/registration system
- ✅ Bcrypt password hashing with salt
- ✅ JWT token-based sessions (24h timeout)
- ✅ Rate limiting (5 attempts = 15min lockout)
- ✅ Password strength validation
- ✅ SQL injection prevention

### 🎥 **Video Processing**
- ✅ Upload & process pre-recorded videos
- ✅ Supported formats: MP4, MOV, AVI, MKV
- ✅ Frame-by-frame analysis
- ✅ Customizable performance modes
- ✅ Progress tracking & ETA

</td>
<td width="50%">

### 📹 **Live Detection**
- ✅ Real-time webcam counting
- ✅ Multi-person tracking
- ✅ Bounding box visualization
- ✅ Smooth FPS optimization
- ✅ GPU acceleration support

### 🚨 **Smart Alerts**
- ✅ Configurable crowd thresholds
- ✅ Visual & audio notifications
- ✅ Alert logging & history
- ✅ Custom threshold profiles

### 🎨 **Modern UI/UX**
- ✅ Beautiful gradient design
- ✅ Responsive layout
- ✅ Dark mode optimization
- ✅ Real-time statistics
- ✅ Download processed videos

</td>
</tr>
</table>

---

## 📦 Installation

### Prerequisites
- **Python 3.8+** 🐍
- **Webcam** (optional, for live mode) 📷
- **FFmpeg** (for video processing) 🎬
- **NVIDIA GPU** (optional, for faster inference) ⚡

### Step-by-Step Setup

```bash
# Clone or download the repository
git clone https://github.com/yourusername/People_counter.git
cd People_counter

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Download YOLOv8 model (if not present)
# The model will auto-download on first run
```

### Requirements
```
streamlit>=1.28.0        # Web UI framework
opencv-python>=4.8.0    # Computer vision
ultralytics>=8.0.0      # YOLOv8 model
torch>=2.0.0            # Deep learning
bcrypt>=4.0.0           # Password hashing
PyJWT>=2.8.0            # Token management
```

---

## 🔧 Usage Guide

### 1️⃣ **Authentication Flow**

```
┌─────────────────────┐
│   Launch App        │
└──────────┬──────────┘
           │
      ┌────▼────┐
      │ Register │──→ Create new account
      │ or Login │
      └────┬────┘
           │
      ┌────▼─────────────────────┐
      │ Dashboard (Authenticated) │
      └────┬─────────────────────┘
           │
   ┌───────┼───────┐
   │       │       │
▼──┴──▼   │    ▼──┴─▼
Upload   Live   Settings
Video   Webcam
```

### 2️⃣ **Video Processing**

1. 📤 Upload a video file (MP4, MOV, AVI, MKV)
2. ⚙️ Choose performance mode:
   - 🚀 **Fast**: Quick analysis, lower accuracy
   - ⚡ **Balanced**: Good speed & accuracy mix
   - 🎯 **Accurate**: Precise counting, slower
3. 🔧 Set crowd alert threshold (optional)
4. ▶️ Process and download results

### 3️⃣ **Live Webcam Counting**

1. 📹 Click "Live Webcam" mode
2. ✅ Grant camera permissions
3. 👀 Real-time people count display
4. 🎬 Stream statistics and alerts
5. ⏹️ Stop when done

### 4️⃣ **Configure Alerts**

```python
# Set threshold in sidebar
Crowd Alert Threshold: 10 people
    │
    ├─ Count < 10 → ✅ Normal
    └─ Count ≥ 10 → 🚨 Alert (sound + notification)
```

---

## 📁 Project Structure

```
People_counter/
│
├── 🌐 streamlit/
│   ├── app.py                 # 🎯 Main Streamlit application
│   └── auth.db               # 🗄️ User authentication database
│
├── 🤖 yolov8peoplecounter/
│   ├── main.py               # Core detection engine
│   ├── tracker.py            # Multi-person tracking
│   ├── yolov8s.pt           # Pre-trained YOLOv8 model
│   └── coco.txt             # COCO dataset class labels
│
├── 📄 AUTHENTICATION.md       # 🔐 Auth system documentation
├── requirements.txt           # 📦 Python dependencies
├── run_app.py                # ⚡ Quick start script
├── run_app.sh                # 🐧 Linux/Mac launcher
└── run_app.bat               # 🪟 Windows launcher
```

---

## 🔐 Security Features

### 🛡️ Password Security

| Feature | Implementation |
|---------|-----------------|
| 🔒 Hashing | Bcrypt (salted, industry-standard) |
| 💪 Strength | 8+ chars, uppercase, lowercase, number, special |
| 🔐 Storage | SQLite encrypted database |
| ⏰ Expiry | 24-hour session timeout |
| 🚫 Attempts | Rate limiting after 5 failed tries |

### 🎫 Token Management

```
Login → bcrypt verify → JWT token generated
         ↓
      24-hour validity
         ↓
    Auto-refresh on activity
         ↓
    Logout → Token invalidated
```

### ✅ Security Checklist

- ✅ Bcrypt with salt for all passwords
- ✅ JWT tokens for session management
- ✅ Rate limiting on failed logins
- ✅ Input validation & sanitization
- ✅ SQL injection prevention
- ✅ CSRF protection ready
- ⚠️ Set `JWT_SECRET` environment variable in production

**🔒 For Production Deployment:**
```bash
export JWT_SECRET="your-long-random-secret-key-here"
streamlit run streamlit/app.py
```

---

## ⚙️ Configuration & Tuning

### 🚀 Performance Modes

Choose the right balance for your use case:

| Mode | Speed | Accuracy | Frame Skip | Image Size | Best For |
|------|-------|----------|-----------|-----------|----------|
| 🚀 Fast | ⚡⚡⚡ | ⭐ | High | Low | Quick analysis |
| ⚡ Balanced | ⚡⚡ | ⭐⭐ | Medium | Medium | Most users |
| 🎯 Accurate | ⭐ | ⭐⭐⭐ | Low | High | Precision needed |

### 💾 GPU Acceleration

**Enable CUDA for 3-5x speedup:**

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

The app automatically detects GPU and displays status in the UI. ⚡

### 🔧 Configuration Options

```python
SESSION_TIMEOUT_HOURS = 24          # Session expiration
MODEL_CONFIDENCE = 0.5              # Detection confidence threshold
VIDEO_FPS_TARGET = 30               # Target processing FPS
MAX_WORKERS = 4                     # Parallel processing threads
```

---

## 🎬 Model Information

### 🤖 YOLOv8s Model

- **Size**: ~42MB
- **Speed**: ~25ms per frame (GPU)
- **Accuracy**: 80%+ mAP on COCO dataset
- **Classes**: 80 (including person, car, dog, etc.)
- **Framework**: PyTorch (ultralytics)

**Auto-download**: Model downloads automatically on first run if not present.

**Manual download:**
```bash
cd yolov8peoplecounter
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

---

## 📊 Example Output

```
📹 Processing: video.mp4
├─ 🎬 Total Frames: 1500
├─ ⏱️ Duration: 50s @ 30fps
├─ 👥 Total People Detected: 2,347
├─ 📈 Average per Frame: 15.6
├─ 🚨 Alert Threshold: 20
├─ ⚠️ Alerts Triggered: 3
└─ ✅ Status: Complete

Download: processed_video_with_annotations.mp4
```

---

## 🐛 Troubleshooting

### ❌ "Account locked for X minutes"
→ Too many login attempts. Wait for cooldown period.

### ❌ "Invalid credentials"
→ Check username/email and password. Verify caps lock.

### ❌ "GPU not detected"
→ Install PyTorch with CUDA support. App will use CPU as fallback.

### ❌ "Video won't process"
→ Ensure FFmpeg is installed. Check video format compatibility.

### ❌ "Webcam permission denied"
→ Allow camera access in browser/OS settings.

### ❌ "Session expired"
→ Login again. Sessions last 24 hours.

**For more details**, see [AUTHENTICATION.md](AUTHENTICATION.md) 📖

---

## 🌟 Tips & Tricks

| Tip | Benefit |
|-----|---------|
| 🚀 Use **Fast Mode** for long videos | Reduce processing time by 50% |
| 💾 Enable GPU (NVIDIA) | 3-5x speed boost |
| 🎯 Lower alert threshold | More sensitive crowd detection |
| 📱 Use balanced mode | Best quality-to-speed ratio |
| 🔄 Batch process videos | Queue multiple files |

---

## 📖 Documentation

- 🔐 **[Authentication System](AUTHENTICATION.md)** - Detailed auth & security info
- 🤖 **YOLOv8 Docs** - https://docs.ultralytics.com/
- 🌐 **Streamlit Docs** - https://docs.streamlit.io/
- 🐍 **Python Docs** - https://docs.python.org/3/

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. 🍴 Fork the repository
2. 🌿 Create feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to branch (`git push origin feature/amazing-feature`)
5. 🎉 Open Pull Request

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 📞 Support & Contact

- 🐛 **Issues**: Open GitHub issue for bugs
- 💡 **Suggestions**: Share ideas via discussions
- 📧 **Email**: [your-email@example.com]

---

<div align="center">

### ⭐ If you find this useful, please give it a star! 🌟

**Built with ❤️ using Streamlit & YOLOv8**

![Python](https://img.shields.io/badge/Made%20with-Python-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-00BFFF?style=flat-square)

</div>
