# dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import random

class GoProDataset(Dataset):
    def __init__(self, root_dir, phase='train', crop_size=256, augment=True):
        self.root_dir = os.path.join(root_dir, phase)
        self.blur_dir = os.path.join(self.root_dir, 'blur')
        self.sharp_dir = os.path.join(self.root_dir, 'sharp')
        self.blur_files = sorted(os.listdir(self.blur_dir))
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.blur_files)

    def __getitem__(self, idx):
        blur_file_path = os.path.join(self.blur_dir, self.blur_files[idx])
        sharp_file_path = os.path.join(self.sharp_dir, self.blur_files[idx])

        blur_img = cv2.imread(blur_file_path)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB) / 255.0
        sharp_img = cv2.imread(sharp_file_path)
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB) / 255.0
        
        if self.crop_size > 0:
            h, w = blur_img.shape[:2]
            x = random.randint(0, w - self.crop_size)
            y = random.randint(0, h - self.crop_size)
            blur_img = blur_img[y:y + self.crop_size, x:x + self.crop_size]
            sharp_img = sharp_img[y:y + self.crop_size, x:x + self.crop_size]

        if self.augment:
            if random.random() > 0.5:
                blur_img = np.fliplr(blur_img)
                sharp_img = np.fliplr(sharp_img)
        
        blur_tensor = torch.from_numpy(blur_img.transpose(2, 0, 1).copy()).float()
        sharp_tensor = torch.from_numpy(sharp_img.transpose(2, 0, 1).copy()).float()
        
        # Normalize from [0, 1] to [-1, 1]
        return (blur_tensor * 2 - 1), (sharp_tensor * 2 - 1)