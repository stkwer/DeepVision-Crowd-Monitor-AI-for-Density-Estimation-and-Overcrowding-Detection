import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt, cm as c
import torch
from torchvision import transforms
from model import CSRNet

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model and weights
model = CSRNet()
checkpoint = torch.load('weights.pth', map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

# Load image
img_path = "C:/22-7359/Deep Vision-2/Root/img_to_predict_2.jpg"
img = Image.open(img_path).convert('RGB')
plt.imshow(img)
plt.title("Original Image")
plt.show()

# Predict
img_tensor = transform(img).unsqueeze(0)
with torch.no_grad():
    output = model(img_tensor)
    count = int(output.sum().item())
    print("Predicted Count:", count)
    density_map = output.squeeze().numpy()

plt.imshow(density_map, cmap=c.jet)
plt.title("Predicted Density Map")
plt.show()

# ============================================================
# run.py — Test CSRNet and visualize density map
# ============================================================
# import os
# import cv2
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from model import CSRNet
# from scipy.io import loadmat

# # -----------------------------
# # CONFIG
# # -----------------------------
# MODEL_PATH = "weights.pth"                     # pretrained model
# TEST_IMAGE_PATH = "C:/22-7359/Deep Vision Crowd Monitor/ShanghaiTech/part_A/test_data/images/IMG_114.jpg"        # input image
# GT_FILE_PATH = "C:/22-7359/Deep Vision Crowd Monitor/ShanghaiTech/part_A/test_data/ground-truth/GT_IMG_114.mat"            # ground truth MAT file
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -----------------------------
# # IMAGE PREPROCESSING
# # -----------------------------
# def load_image(img_path):
#     img = cv2.imread(img_path)
#     if img is None:
#         raise FileNotFoundError(f"Image not found: {img_path}")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485,0.456,0.406],
#                              std=[0.229,0.224,0.225])
#     ])
#     return transform(img).unsqueeze(0), img

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# model = CSRNet().to(DEVICE)
# checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
# model.load_state_dict(checkpoint)
# model.eval()
# print("✅ Model loaded successfully!")

# # -----------------------------
# # PREDICTION
# # -----------------------------
# img_tensor, orig_img = load_image(TEST_IMAGE_PATH)
# img_tensor = img_tensor.to(DEVICE)

# with torch.no_grad():
#     output = model(img_tensor)

# pred_density = output.squeeze().cpu().numpy()
# pred_count = float(pred_density.sum())
# print(f"🔹 Predicted Count: {pred_count:.2f}")

# # -----------------------------
# # LOAD GROUND TRUTH AND COMPUTE COUNT-BASED MAE
# # -----------------------------
# if os.path.exists(GT_FILE_PATH):
#     mat = loadmat(GT_FILE_PATH)
#     print("✅ Loaded .mat file keys:", mat.keys())

#     # Extract count from MATLAB struct
#     image_info = mat['image_info'][0,0]
#     if 'number' in image_info.dtype.names:
#         gt_density = np.array(image_info['number'], dtype=np.float32)
#     else:
#         raise KeyError("Density map not found in 'image_info' struct")

#     actual_count = float(gt_density.sum())
#     count_mae = abs(pred_count - actual_count)  # Count-based MAE
#     print(f"🎯 Actual Count: {actual_count:.2f}")
#     print(f"📉 Count-based MAE: {count_mae:.2f}")

# else:
#     actual_count = None
#     count_mae = None
#     print("⚠️ Ground truth file not found, skipping MAE calculation.")

# # -----------------------------
# # VISUALIZATION (Original + Predicted side by side)
# # -----------------------------
# plt.figure(figsize=(14,6))

# # Original image
# plt.subplot(1,2,1)
# plt.imshow(orig_img)
# title_text = f"Original Image\nPredicted Count: {pred_count:.2f}"
# if actual_count is not None:
#     title_text += f" | Actual Count: {actual_count:.2f}\nCount MAE: {count_mae:.2f}"
# plt.title(title_text, fontsize=13)
# plt.axis("off")

# # Predicted density map
# plt.subplot(1,2,2)
# plt.imshow(pred_density, cmap="jet")
# plt.title("Predicted Density Map", fontsize=13)
# plt.axis("off")
# plt.colorbar(label="Density")

# plt.tight_layout()
# plt.show()



# # cd "C:\22-7359\Deep Vision-2\Root"
