**🧠 AI Crowd Monitoring System (YOLOv8 + CSRNet + Streamlit)**
This project is an AI-powered crowd monitoring system that detects people, estimates crowd density, and provides an overcrowding alert using a combination of YOLOv8 and CSRNet.
A Streamlit web interface allows users to upload images or videos for real-time monitoring.

**🔥 Key Features**
Real-time people detection using YOLOv8
Density-based crowd estimation using CSRNet
Overcrowding alert when threshold is crossed
Interactive Streamlit UI
Supports image & video input
Trained on the ShanghaiTech dataset for high accuracy

**🛠️ Technologies Used**
Python
YOLOv8 (Ultralytics)
CSRNet (PyTorch)
Streamlit
OpenCV
NumPy

**📁 Dataset**
The system is trained on the ShanghaiTech Crowd Counting dataset, which provides high-density crowd scenes and ground-truth density maps essential for accurate crowd estimation.

**🧩 Project Workflow**
User uploads an image or video
YOLOv8 detects visible individuals
CSRNet generates a density map and estimates total count
System compares count with threshold
Displays crowd count + overcrowded alert if needed

**▶️ How to Run the Application**
streamlit run crowd_app.py

**📌 Project Status**
✔️ Completed — fully functional AI-based crowd monitoring system.
