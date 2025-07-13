
# ClearGAN: Adaptive Adversarial Network for High-Fidelity Image Deblurring

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

An advanced Generative Adversarial Network (GAN) built with PyTorch, designed to remove motion blur from images with high fidelity.

![Deblurring Example](blur_sharpen.jpg)

---

## 📋 Project Overview

Image deblurring is a challenging inverse problem in computer vision that aims to restore a sharp, clean image from a blurry input, which often suffers from information loss and noise. This project, **ClearGAN**, tackles this challenge by implementing a sophisticated conditional GAN that learns to map blurry images to their sharp counterparts.

The architecture is built around a U-Net-based generator enhanced with modern deep learning techniques and a multi-scale discriminator for robust adversarial training. The entire system is optimized using a hybrid loss function and features a custom adaptive training engine, allowing it to produce perceptually realistic and high-quality results.

---

## ✨ Key Features

* **Advanced U-Net Generator**: A deep U-Net architecture enhanced with **Residual Blocks** to facilitate gradient flow and **Self-Attention** layers to capture long-range spatial dependencies, preserving fine details.
* **Multi-Scale Discriminator**: Utilizes a multi-scale discriminator with **Spectral Normalization** to evaluate the realism of the generated images at different resolutions, providing more robust feedback to the generator.
* **Stable Adversarial Training**: Implements a **WGAN-GP (Wasserstein GAN with Gradient Penalty)** framework to mitigate mode collapse and ensure stable training dynamics.
* **Hybrid Loss Function**: Combines three distinct loss functions for comprehensive optimization:
    * **L1 Loss**: Ensures pixel-level accuracy.
    * **Adversarial Loss (WGAN-GP)**: Pushes the generator to produce images on the manifold of real sharp images.
    * **VGG Perceptual Loss**: Improves the perceptual quality of the output by minimizing feature-space differences in a pre-trained VGG19 network.
* **Adaptive Training Engine**: A custom training loop that intelligently adjusts the weights of the L1 and adversarial losses, as well as the discriminator's training frequency, based on model performance.
* **Command-Line Interface**: Includes separate, easy-to-use scripts for evaluation and single-image inference, built with Python's `argparse`.

---

## 🛠️ Technology Stack

* **Core Framework**:
    * **PyTorch**: For building, training, and deploying the deep learning models.
    * **Torchvision**: Used for accessing the pre-trained VGG19 model for perceptual loss.
* **GAN Architecture & Techniques**:
    * **U-Net**: The foundational encoder-decoder architecture with skip connections.
    * **Residual Blocks**: Inspired by ResNet to prevent vanishing gradients in the deep generator.
    * **Self-Attention**: SAGAN-style attention mechanism to model long-range dependencies.
    * **Multi-Scale Discriminator**: Provides a robust loss signal by evaluating images at multiple scales.
    * **Spectral Normalization**: A weight normalization technique to stabilize discriminator training.
* **Training & Optimization**:
    * **WGAN-GP**: The chosen adversarial training framework for its stability.
    * **Adaptive Hyperparameter Tuning**: A custom engine to dynamically balance losses and training frequency.
    * **Optimizers**: `Adam` for the generator and `RMSprop` for the discriminator.
    * **Learning Rate Schedulers**: `ExponentialLR` to decay learning rates over time.
* **Evaluation & Data Handling**:
    * **Scikit-image**: For calculating standard image quality metrics (PSNR & SSIM).
    * **OpenCV**: For efficient image loading, color space conversion, and saving.
    * **NumPy**: For numerical operations and data manipulation.
    * **Matplotlib**: For visualizing results during training and evaluation.

---

## 📂 Project Structure

/ClearGAN/
├── train.py                # Main script to run the training process  
├── evaluate.py             # Script to evaluate a trained model on the test set  
├── inference.py            # Script to deblur a single image  
├── config.py               # Hyperparameters and global settings  
├── dataset.py              # GoProDataset class for data loading  
├── models.py               # Generator and Discriminator model architectures  
├── losses.py               # VGGPerceptualLoss custom loss function  
├── trainer.py              # The main training and validation loop logic  
└── requirements.txt        # Project dependencies

---

## 🚀 Usage Guide

### 1. Setup

```bash
git clone <your-repo-url>
cd ClearGAN
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset

Download the GoPro dataset and structure it as follows:

/path/to/your/data/  
├── train/  
│   ├── blur/  
│   └── sharp/  
└── test/  
    ├── blur/  
    └── sharp/  

Set `ROOT_DIR` in `config.py` to this path.

### 3. Training

```bash
python train.py
```

Training progress, validation results, and sample output images will be saved to disk.

### 4. Evaluation

```bash
python evaluate.py path/to/your/trained_model.pth
```

Calculates PSNR and SSIM metrics on the test dataset.

### 5. Inference

```bash
python inference.py path/to/your/trained_model.pth path/to/blurry_image.jpg path/to/output_image.jpg
```

---


### 🏆 Results and Metrics

The performance of **ClearGAN** is evaluated on a held-out validation set using two standard image quality metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**: Evaluates pixel-level accuracy. Higher values indicate better reconstruction quality.
- **SSIM (Structural Similarity Index Measure)**: Assesses perceptual similarity between the original and generated images. Closer to 1.0 is better.

#### 📊 Quantitative Results

| Metric | Score     |
|--------|-----------|
| PSNR   | 25.93 dB  |
| SSIM   | 0.770     |

These results are based on the model's performance after 29 epochs of training and provide a solid baseline for image deblurring tasks. The scores highlight ClearGAN's ability to effectively restore sharp images from blurry inputs.

## Results
