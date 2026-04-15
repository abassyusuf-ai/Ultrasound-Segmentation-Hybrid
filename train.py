import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataloader import BUSIDataset
from src.model_unet import UNet

# 1. Hyperparameters
LEARNING_RATE = 1e-4
# Keep as "cpu" to avoid the NVIDIA driver error shown in your logs
DEVICE = "cpu" 
BATCH_SIZE = 8
EPOCHS = 20

# Use the verified WSL path to your Windows drive
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Dataset_BUSI_with_GT")
def train():
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # Load Dataset using the corrected path
    dataset = BUSIDataset(root_dir=DATA_DIR)
    
    if len(dataset) == 0:
        return # The dataloader's internal print will explain why

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Starting training on {DEVICE}...")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)
            loss = criterion(predictions, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "unet_busi.pth")
    print("Training complete. Model saved as unet_busi.pth")

if __name__ == "__main__":
    train()
