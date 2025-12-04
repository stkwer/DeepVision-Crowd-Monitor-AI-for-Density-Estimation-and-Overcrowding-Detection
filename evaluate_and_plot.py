import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# ---------- Dataset ----------
class CrowdDataset(Dataset):
    def __init__(self, npy_path):
        self.data = np.load(npy_path, allow_pickle=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img = torch.tensor(item["image"].transpose(2,0,1), dtype=torch.float32)
        density = torch.tensor(item["density_map"][None, :, :], dtype=torch.float32)
        return img, density, item

# ---------- CSRNet ----------
class CSRNet(torch.nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        vgg_feat = models.vgg16(pretrained=False).features
        self.frontend = torch.nn.Sequential(*list(vgg_feat.children())[:23])
        self.backend = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 256, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 128, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

# ---------- Evaluate ----------
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_gts = [], []

    with torch.no_grad():
        for imgs, dens, _ in tqdm(dataloader, desc="Evaluating"):
            imgs, dens = imgs.to(device), dens.to(device)
            outputs = model(imgs)
            outputs = F.interpolate(outputs, size=dens.shape[2:], mode='bilinear', align_corners=False)
            
            all_preds.extend(outputs.sum(dim=(1,2,3)).cpu().numpy())
            all_gts.extend(dens.sum(dim=(1,2,3)).cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    mae = np.mean(np.abs(all_preds - all_gts))
    mse = np.sqrt(np.mean((all_preds - all_gts)**2))
    return mae, mse

# ---------- Visualization ----------
def visualize_random(data, model, device, n=5):
    model.eval()
    samples = random.sample(list(data), n)

    with torch.no_grad():
        for item in samples:
            img_tensor = torch.tensor(item["image"].transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(device)
            density_true = item["density_map"]
            output = model(img_tensor)
            output = F.interpolate(output, size=density_true.shape, mode='bilinear', align_corners=False)
            density_pred = output.squeeze().cpu().numpy()
            pred_count = density_pred.sum()
            img = item["image"]

            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.imshow(img)
            plt.title(f"Original Image\nPredicted Count: {pred_count:.0f}")
            plt.axis("off")

            plt.subplot(1,2,2)
            plt.imshow(density_pred, cmap='jet')
            plt.title("Predicted Density Map")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

# ---------- Main ----------
# ---------- Main ----------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use relative paths
    npy_path = "processed_data_from_preproc_imagematch_fixed.npy"
    ckpt_path = "csrnet_epoch_27.pth"  # latest checkpoint

    data = np.load(npy_path, allow_pickle=True)
    dataset = CrowdDataset(npy_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    model = CSRNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    mae, mse = evaluate(model, dataloader, device)
    print(f"Evaluation Results — MAE: {mae:.2f}, MSE: {mse:.2f}")

    visualize_random(data, model, device)
