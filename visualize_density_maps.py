import os
import numpy as np
import matplotlib.pyplot as plt
import random

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(ROOT_DIR, "processed_data_from_preproc_imagematch_fixed.npy")

# --- Load, with clear error messages ---
if not os.path.exists(data_path):
    raise SystemExit(f"❌ File not found: {data_path}\nRun process_crowd.py first.")

data = np.load(data_path, allow_pickle=True)
n = len(data)
print(f"Loaded {n} samples from {data_path}")

if n == 0:
    raise SystemExit("❌ Dataset is empty. Re-run process_crowd.py and check for warnings above.")

# Show keys and shapes of first item to help debug
first = data[0]
print("\nFirst item type:", type(first))
if isinstance(first, dict):
    print("Keys:", list(first.keys()))
    if "image" in first:
        print(" image shape:", np.array(first["image"]).shape, "dtype:", np.array(first["image"]).dtype)
    if "density_map" in first:
        print(" density_map shape:", np.array(first["density_map"]).shape, "sum:", float(np.array(first["density_map"]).sum()))
    if "count" in first:
        print(" count value:", first.get("count"))

# --- Choose how many to show (safe) ---
num_to_show = min(5, n)
print(f"\nShowing {num_to_show} sample(s).")

samples = random.sample(list(data), num_to_show)

for i, sample in enumerate(samples, 1):
    img = sample.get("image")
    density = sample.get("density_map")
    count = sample.get("count", None)

    if img is None or density is None:
        print(f"⚠️ Sample {i} missing image or density_map keys, skipping.")
        continue

    # ensure numpy arrays
    img = np.array(img)
    density = np.array(density)

    plt.figure(figsize=(11, 5))

    # Left: Image
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img)
    title_left = "Original Image"
    if count is not None:
        title_left += f"\nGT Count: {count:.0f}"
    ax1.set_title(title_left)
    ax1.axis("off")

    # Right: Density Map (heatmap + contour)
    ax2 = plt.subplot(1, 2, 2)
    im = ax2.imshow(density, cmap='jet')
    ax2.set_title("Density Map (Heatmap + Contours)")
    ax2.axis("off")
    # contours
    try:
        cs = ax2.contour(density, levels=6, linewidths=0.8, alpha=0.8)
    except Exception:
        pass
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
