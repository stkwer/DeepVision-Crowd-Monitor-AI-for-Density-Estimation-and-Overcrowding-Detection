#!/usr/bin/env python3
"""
Simplified and fixed version of process_preproc_by_image_match.py
✅ Uses relative paths and handles missing files gracefully.

Usage:
  python process_crowd.py
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

# ---------------- CONFIG ----------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

preprocessed_folder = os.path.join(ROOT_DIR, "preprocessed")
image_base_folder = os.path.join(ROOT_DIR, "images", "ShanghaiTech")
output_csv = os.path.join(ROOT_DIR, "preproc_matched_to_gt_fixed.csv")
output_npy = os.path.join(ROOT_DIR, "processed_data_from_preproc_imagematch_fixed.npy")

# ---------------- HELPERS ----------------
def list_original_images_and_gt(base_folder):
    """Collect all original images and their GT .mat paths."""
    rows = []
    for part in ["part_A", "part_B"]:
        for split in ["train_data", "test_data"]:
            img_dir = os.path.join(base_folder, part, split, "images")
            possible_gt_dirs = [
                os.path.join(base_folder, part, split, "ground-truth"),
                os.path.join(base_folder, part, split, "grround-truth"),
                os.path.join(base_folder, part, split, "ground_truth"),
                os.path.join(base_folder, part, split, "groundtruth"),
            ]
            gt_dir = next((p for p in possible_gt_dirs if os.path.isdir(p)), None)
            if not os.path.isdir(img_dir) or gt_dir is None:
                continue

            for fname in os.listdir(img_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    orig_path = os.path.join(img_dir, fname)
                    base = os.path.splitext(fname)[0]
                    gt_name = f"GT_{base}.mat"
                    gt_path = os.path.join(gt_dir, gt_name)
                    if not os.path.exists(gt_path):
                        # fallback match by number
                        nums = re.findall(r'\d+', base)
                        found = None
                        if nums:
                            for g in os.listdir(gt_dir):
                                if g.lower().endswith('.mat') and all(n in g for n in nums):
                                    found = os.path.join(gt_dir, g)
                                    break
                        gt_path = found
                    rows.append({
                        "orig_image": orig_path,
                        "gt_path": gt_path,
                        "base": base
                    })
    return rows

# ---------------- MAIN PIPELINE ----------------
print("Collecting original images and GT paths...")
orig_rows = list_original_images_and_gt(image_base_folder)
orig_rows = [r for r in orig_rows if r["gt_path"] is not None]
print(f"Found {len(orig_rows)} originals with GT.")

# Build lookup table by numeric part
orig_lookup = {}
for r in orig_rows:
    nums = re.findall(r"\d+", r["base"])
    if nums:
        orig_lookup[nums[-1]] = r
print(f"Numeric lookup table built with {len(orig_lookup)} entries.")

# Get preprocessed files
pre_files = [f for f in os.listdir(preprocessed_folder)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(pre_files)} preprocessed files.")

mappings = []
for f in pre_files:
    pre_path = os.path.join(preprocessed_folder, f)
    nums = re.findall(r"\d+", f)
    match = None
    if nums and nums[-1] in orig_lookup:
        match = orig_lookup[nums[-1]]

    if match:
        mappings.append({
            "preprocessed_path": pre_path,
            "preprocessed_filename": f,
            "matched_original_path": match["orig_image"],
            "matched_original_filename": os.path.basename(match["orig_image"]),
            "gt_path": match["gt_path"],
            "score": 1.0
        })
    else:
        mappings.append({
            "preprocessed_path": pre_path,
            "preprocessed_filename": f,
            "matched_original_path": None,
            "matched_original_filename": None,
            "gt_path": None,
            "score": 0.0
        })

# Save mapping CSV
df_map = pd.DataFrame(mappings)
df_map.to_csv(output_csv, index=False)
print(f"✅ Saved mapping CSV: {output_csv}")
print("Preview:")
print(df_map.head())

# ---------------- Density Map Creation ----------------
processed = []
failed = []

for row in df_map.dropna(subset=["gt_path"]).itertuples():
    p_img = row.preprocessed_path
    gt = row.gt_path
    matched_orig = row.matched_original_path

    # Read preprocessed image
    img_pre = cv2.imread(p_img)
    if img_pre is None:
        print("⚠️ Could not read preprocessed image:", p_img)
        continue
    img_rgb = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    img_norm = img_resized.astype(np.float32) / 255.0

    # Original image for scaling
    if matched_orig and os.path.exists(matched_orig):
        orig_img = cv2.imread(matched_orig)
        if orig_img is None:
            print("⚠️ Could not read matched original image:", matched_orig)
            h_orig, w_orig = img_pre.shape[:2]
        else:
            h_orig, w_orig = orig_img.shape[:2]
    else:
        h_orig, w_orig = img_pre.shape[:2]

    # --- Load and extract ground truth points ---
    try:
        mat = loadmat(gt)
    except Exception as e:
        print("❌ Could not load mat:", gt, e)
        failed.append((gt, "load_error"))
        continue

    pts_arr = None
    try:
        for key in ["annPoints", "location", "points", "image_info", "dot", "GT", "gt"]:
            if key in mat:
                data = mat[key]
                break
        else:
            data = list(mat.values())[-1]

        if isinstance(data, np.ndarray) and data.dtype == object:
            try:
                pts_arr = np.array(data[0][0][0][0][0])
            except:
                try:
                    pts_arr = np.array(data[0][0][0][0])
                except:
                    try:
                        pts_arr = np.array(data[0][0][0])
                    except:
                        pass
        elif isinstance(data, np.ndarray):
            pts_arr = data.reshape(-1, 2)

    except Exception as e:
        print(f"⚠️ Could not extract GT from {gt}: {e}")
        failed.append((gt, "extract_error"))
        continue

    if pts_arr is None or pts_arr.size == 0:
        print("⚠️ No points found in", gt)
        failed.append((gt, "no_points"))
        continue

    # Scale points to 512x512 based on ORIGINAL image
    w_ratio = 512.0 / float(max(w_orig, 1))
    h_ratio = 512.0 / float(max(h_orig, 1))
    pts_res = pts_arr.astype(np.float32)
    pts_res[:, 0] *= w_ratio
    pts_res[:, 1] *= h_ratio
    pts_int = np.round(pts_res).astype(int)
    pts_int[:, 0] = np.clip(pts_int[:, 0], 0, 511)
    pts_int[:, 1] = np.clip(pts_int[:, 1], 0, 511)

    # Density map
    density = np.zeros((512, 512), dtype=np.float32)
    for x, y in pts_int:
        density[y, x] += 1.0
    density = gaussian_filter(density, sigma=15)
    cnt = float(density.sum())

    processed.append({
        "preprocessed_image": p_img,
        "matched_original": matched_orig,
        "gt_path": gt,
        "image": img_norm,
        "density_map": density,
        "count": cnt
    })
    print(f"✅ Processed: {os.path.basename(p_img)} → Count={cnt:.2f}")

# Save dataset
np.save(output_npy, processed, allow_pickle=True)
print(f"\n✅ Saved processed dataset: {output_npy} (items: {len(processed)})")

if failed:
    print("\n⚠️ Some GT files failed parsing:")
    for f in failed[:10]:
        print("   ", f)

print("\n🎯 Done successfully.")
