import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.model_unet import UNet
from sklearn.metrics import f1_score, accuracy_score, jaccard_score, roc_curve, auc
from skimage.segmentation import chan_vese # The Active Contour Model

DEVICE = "cpu"
MODEL_PATH = "unet_busi.pth"
TEST_IMG = "/mnt/c/Users/hp/unnes_lecture_env/Project 2/Dataset_BUSI_with_GT/benign/benign (10).png"
TEST_MASK = "/mnt/c/Users/hp/unnes_lecture_env/Project 2/Dataset_BUSI_with_GT/benign/benign (10)_mask.png"

def visualize_and_evaluate():
    # 1. Load Model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Prepare Image
    img = cv2.imread(TEST_IMG, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (256, 256))
    img_norm = img_resized.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)

    # 3. Step 1: U-Net Prediction (Global Localization)
    with torch.no_grad():
        raw_output = model(input_tensor)
        prob_map = raw_output.squeeze().cpu().numpy()
        binary_pred = (prob_map > 0.5).astype(np.float32)

    # 4. Step 2: Active Contour Refinement (Local Precision)
    # The U-Net mask acts as the 'starting' boundary for the snake model
    refined_mask = chan_vese(img_norm, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, 
                             max_num_iter=100, dt=0.5, init_level_set=binary_pred)
    refined_binary = refined_mask.astype(np.float32)

    # 5. Load Ground Truth
    gt_mask = cv2.imread(TEST_MASK, cv2.IMREAD_GRAYSCALE)
    gt_binary = (cv2.resize(gt_mask, (256, 256)) > 127).astype(np.float32)

    # 6. Compare Metrics (U-Net vs Hybrid)
    f1_unet = f1_score(gt_binary.flatten(), binary_pred.flatten())
    f1_hybrid = f1_score(gt_binary.flatten(), refined_binary.flatten())

    print(f"U-Net Dice Score:    {f1_unet:.4f}")
    print(f"Hybrid Dice Score:   {f1_hybrid:.4f}")

    # 7. Final Plotting
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 4, 1); plt.imshow(img_resized, cmap='gray'); plt.title("Original Ultrasound")
    plt.subplot(1, 4, 2); plt.imshow(gt_binary, cmap='gray'); plt.title("Ground Truth (Doctor)")
    plt.subplot(1, 4, 3); plt.imshow(binary_pred, cmap='gray'); plt.title("U-Net Prediction")
    plt.subplot(1, 4, 4); plt.imshow(refined_binary, cmap='gray'); plt.title("Hybrid (U-Net + ACM)")
    
    plt.tight_layout()
    plt.savefig("hybrid_comparison.png")
    plt.show()

if __name__ == "__main__":
    visualize_and_evaluate()
