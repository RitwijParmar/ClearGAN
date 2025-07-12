# inference.py
import cv2
import torch
import numpy as np
import argparse
from models import UNetGenerator
import config

def deblur_image(model, image_path):
    """Deblurs a single image file."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    
    # Normalize from [0, 1] to [-1, 1]
    img_tensor = (torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) * 2) - 1
    
    with torch.no_grad():
        output_tensor = model(img_tensor.to(config.DEVICE)).cpu()
    
    # Denormalize from [-1, 1] back to [0, 1]
    output_img = (output_tensor.squeeze(0).numpy().transpose(1, 2, 0) + 1) / 2
    
    # Convert back to BGR for OpenCV
    output_bgr = cv2.cvtColor((output_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return output_bgr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deblur a single image using a trained ClearGAN model.")
    parser.add_argument("model_path", type=str, help="Path to the trained generator .pth file.")
    parser.add_argument("input_image", type=str, help="Path to the blurry input image.")
    parser.add.argument("output_image", type=str, help="Path to save the deblurred output image.")
    args = parser.parse_args()

    model = UNetGenerator().to(config.DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=config.DEVICE))
    model.eval()

    deblurred_image = deblur_image(model, args.input_image)
    cv2.imwrite(args.output_image, deblurred_image)
    print(f"Deblurred image saved to {args.output_image}")