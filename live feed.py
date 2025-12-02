import cv2
import numpy as np
from ultralytics import YOLO
import time

# === CONFIGURATION ===
MODEL_NAME = "yolov8n.pt"  # lightweight YOLOv8 model
OUTPUT_FILE = "crowd_monitor_output.avi"
ALERT_THRESHOLD = 5  # number of people to trigger overcrowding alert

# === INITIALIZE ===
model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(0)  # 0 = default webcam

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

print("🎥 Crowd monitoring started... Press 'Q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True)
    person_centers = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                person_centers.append((cx, cy))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Generate heatmap
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), np.float32)
    for (cx, cy) in person_centers:
        cv2.circle(heatmap, (cx, cy), 50, 1, -1)

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), 25)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    # Overlay heatmap
    overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

    # Display crowd count
    count = len(person_centers)
    color = (0, 255, 0) if count < ALERT_THRESHOLD else (0, 0, 255)
    cv2.putText(overlay, f"Crowd Count: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Alert for overcrowding
    if count >= ALERT_THRESHOLD:
        cv2.putText(overlay, "⚠️ OVERCROWDING ALERT!", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        print("🚨 ALERT: Overcrowding detected!")

    # Initialize video writer (once)
    if out is None:
        h, w = overlay.shape[:2]
        out = cv2.VideoWriter(OUTPUT_FILE, fourcc, 20.0, (w, h))

    out.write(overlay)
    cv2.imshow("Crowd Monitor", overlay)

    # ✅ Press "Q" to stop safely
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Stopping crowd monitoring...")
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
