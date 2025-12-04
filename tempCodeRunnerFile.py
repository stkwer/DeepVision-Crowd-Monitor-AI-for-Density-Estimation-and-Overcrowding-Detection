import os
import cv2
import numpy as np

# Get the root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input folder (tumhare images folder)
input_folder = os.path.join(ROOT_DIR, "images")

# Output folder (preprocessed images)
output_folder = os.path.join(ROOT_DIR, "preprocessed")
os.makedirs(output_folder, exist_ok=True)

# Loop through all images
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".bmp", ".jpg")):
        img_path = os.path.join(input_folder, filename)

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read {filename}, skipping.")
            continue

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 512x512
        img_resized = cv2.resize(img_rgb, (512, 512))

        # Normalize to [0,1]
        img_normalized = img_resized / 255.0

        # Save preprocessed image
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, (img_normalized * 255).astype('uint8'))

        print(f"✅ Processed {filename}")

print("🎉 All images preprocessed and saved in:", output_folder)
