# evaluate.py
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from torch.utils.data import DataLoader
import argparse

import config
from dataset import GoProDataset
from models import UNetGenerator

def evaluate(model_path):
    # Use full images for testing, no random crop
    test_dataset = GoProDataset(config.ROOT_DIR, phase='test', crop_size=0, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = UNetGenerator().to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    
    psnr_list, ssim_list = [], []
    
    def denormalize(tensor):
        return (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2

    for blur, sharp in tqdm(test_loader, desc="Evaluating on Test Set"):
        with torch.no_grad():
            fake = model(blur.to(config.DEVICE)).cpu()
        
        fake_np = denormalize(fake)
        sharp_np = denormalize(sharp)
        
        p = psnr(sharp_np, fake_np, data_range=1.0)
        s = ssim(sharp_np, fake_np, channel_axis=-1, data_range=1.0)
        psnr_list.append(p)
        ssim_list.append(s)
        
    print(f"\n--- Evaluation Results ---")
    print(f"Average PSNR: {np.mean(psnr_list):.4f}")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained ClearGAN model.")
    parser.add_argument("model_path", type=str, help="Path to the trained generator .pth file.")
    args = parser.parse_args()
    evaluate(args.model_path)