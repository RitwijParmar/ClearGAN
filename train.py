# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from dataset import GoProDataset
from models import UNetGenerator, MultiScaleDiscriminator
from losses import VGGPerceptualLoss
from trainer import train_one_epoch, validate_one_epoch, AdaptiveHyperparams

def main():
    # Datasets and Dataloaders
    train_dataset = GoProDataset(config.ROOT_DIR, phase='train', crop_size=config.CROP_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    
    val_dataset = GoProDataset(config.ROOT_DIR, phase='test', crop_size=config.CROP_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    # Models
    gen = UNetGenerator().to(config.DEVICE)
    disc = MultiScaleDiscriminator(ndf=16, n_layers=2).to(config.DEVICE)
    
    # Losses
    losses = {
        'l1': torch.nn.L1Loss(),
        'perceptual': VGGPerceptualLoss().to(config.DEVICE)
    }
    
    # Optimizers
    optimizers = {
        'gen': optim.Adam(gen.parameters(), lr=config.GEN_LR, betas=config.GEN_BETAS),
        'disc': optim.RMSprop(disc.parameters(), lr=config.DISC_LR)
    }
    
    # Schedulers
    schedulers = {
        'gen': optim.lr_scheduler.ExponentialLR(optimizers['gen'], gamma=config.GEN_SCHEDULER_GAMMA),
        'disc': optim.lr_scheduler.ExponentialLR(optimizers['disc'], gamma=config.DISC_SCHEDULER_GAMMA)
    }

    # Adaptive Hyperparameters
    adaptive_hp = AdaptiveHyperparams(
        initial_l1=config.INITIAL_LAMBDA_L1,
        initial_gan=config.INITIAL_LAMBDA_GAN,
        initial_freq=config.INITIAL_DISC_TRAIN_FREQ
    )
    
    # --- Main Training Loop ---
    for epoch in range(config.NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        
        gen_loss, disc_loss, disc_acc, adaptive_hp = train_one_epoch(
            gen, disc, train_loader, optimizers, losses, schedulers, adaptive_hp, config
        )
        
        val_psnr, val_ssim = validate_one_epoch(
            gen, disc, val_loader, losses, adaptive_hp, config, epoch
        )

        print(f"Train Gen Loss: {gen_loss:.4f}, Train Disc Loss: {disc_loss:.4f}")
        print(f"Val PSNR: {val_psnr:.4f}, Val SSIM: {val_ssim:.4f}")
        print(f"Adaptive Params: λ_L1={adaptive_hp.lambda_l1:.3f}, λ_GAN={adaptive_hp.lambda_gan:.3f}, freq={adaptive_hp.disc_train_freq}, disc_acc={disc_acc:.3f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(gen.state_dict(), config.MODEL_SAVE_PATH.format(epoch))
            print(f"Model saved at epoch {epoch}")

if __name__ == '__main__':
    main()