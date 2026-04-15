import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob

class BUSIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        
        # 1. More robust search: catch .png and .PNG (Windows case sensitivity)
        self.all_files = glob(os.path.join(root_dir, "**/*.png"), recursive=True) + \
                         glob(os.path.join(root_dir, "**/*.PNG"), recursive=True)
        
        # 2. Filter for images only (excluding masks)
        self.image_paths = sorted([f for f in self.all_files if "_mask" not in f])
        
        # 3. DEBUG: This will print in your terminal so you can see if it worked
        if len(self.image_paths) == 0:
            print(f"ERROR: No images found at: {root_dir}")
        else:
            print(f"SUCCESS: Loaded {len(self.image_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 4. Handle different extension cases for mask replacement
        if img_path.endswith(".png"):
            mask_path = img_path.replace(".png", "_mask.png")
        else:
            mask_path = img_path.replace(".PNG", "_mask.png")
            
        # Check if mask exists to avoid cv2 errors
        if not os.path.exists(mask_path):
            # Try lowercase mask extension if above fails
            mask_path = mask_path.replace("_mask.png", "_mask.PNG")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # If mask is missing, create a blank one to prevent crash
        if mask is None:
            mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Resize to standard U-Net input size
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        
        # Normalize: Image to [0,1], Mask to binary 0/1
        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        mask = np.where(mask > 127, 1.0, 0.0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask
