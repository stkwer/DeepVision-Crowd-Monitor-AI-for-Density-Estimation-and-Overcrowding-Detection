#!/usr/bin/env python3
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# ----------------- Dataset -----------------
class CrowdDataset(Dataset):
    def __init__(self, npy_path):
        self.data = np.load(npy_path, allow_pickle=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img = torch.tensor(item["image"].transpose(2, 0, 1), dtype=torch.float32)
        density = torch.tensor(item["density_map"][None, :, :], dtype=torch.float32)
        return img, density

# ----------------- CSRNet Model -----------------
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        vgg_feat = models.vgg16(pretrained=True).features
        self.frontend = nn.Sequential(*list(vgg_feat.children())[:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

# ----------------- Training -----------------
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, dens in tqdm(dataloader, desc="Training"):
        imgs = imgs.to(device)
        dens = dens.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)

        outputs = F.interpolate(outputs, size=dens.shape[2:], mode='bilinear', align_corners=False)
        
        loss = criterion(outputs, dens)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(dataloader.dataset)

# ----------------- Main -----------------
def main():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    npy_path = os.path.join(ROOT, "processed_data_from_preproc_imagematch_fixed.npy")

    batch_size = 4
    lr = 1e-6
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CrowdDataset(npy_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = CSRNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # ---------------------------
    # LOAD LATEST CHECKPOINT
    # ---------------------------
    ckpt_dir = os.path.join(ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_files = [
        f for f in os.listdir(ckpt_dir) if f.startswith("csrnet_epoch_") and f.endswith(".pth")
    ]

    start_epoch = 1

    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
        latest_ckpt = checkpoint_files[-1]
        epoch_num = int(latest_ckpt.split("_")[2].split(".")[0])

        print(f"🔄 Loading checkpoint: {latest_ckpt}")
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, latest_ckpt), map_location=device))
        start_epoch = epoch_num + 1
        print(f"➡️ Resuming from Epoch {start_epoch}")

    else:
        print("🆕 No checkpoints found. Starting training from scratch...")

    # ---------------------------
    # TRAINING LOOP
    # ---------------------------
    for epoch in range(start_epoch, num_epochs + 1):
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{num_epochs} - Loss: {loss:.6f}")
        
        checkpoint_path = os.path.join(ckpt_dir, f"csrnet_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Saved: {checkpoint_path}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
