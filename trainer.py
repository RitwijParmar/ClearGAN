# trainer.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

class AdaptiveHyperparams:
    """A class to adaptively tune loss weights and training frequencies."""
    def __init__(self, initial_l1, initial_gan, initial_freq):
        self.lambda_l1 = initial_l1
        self.lambda_gan = initial_gan
        self.disc_train_freq = initial_freq
        self.loss_history = {'gen': [], 'disc': []}

    def update_weights(self, gen_loss, disc_loss):
        self.loss_history['gen'].append(gen_loss)
        self.loss_history['disc'].append(disc_loss)
        if len(self.loss_history['gen']) > 10:
            self.loss_history['gen'].pop(0)
            self.loss_history['disc'].pop(0)
        
        if len(self.loss_history['gen']) >= 3:
            gen_trend = np.mean(self.loss_history['gen'][-3:])
            disc_trend = np.mean(self.loss_history['disc'][-3:])
            loss_ratio = gen_trend / (disc_trend + 1e-8)
            if loss_ratio > 2.5:
                self.lambda_gan = min(self.lambda_gan * 1.02, 3.0)
                self.lambda_l1 = max(self.lambda_l1 * 0.98, 2.0)
            elif loss_ratio < 0.5:
                self.lambda_gan = max(self.lambda_gan * 0.95, 0.2)
                self.lambda_l1 = min(self.lambda_l1 * 1.05, 15.0)
    
    def update_training_freq(self, disc_accuracy):
        if disc_accuracy > 0.75:
            self.disc_train_freq = min(self.disc_train_freq + 1, 5)
        elif disc_accuracy < 0.45:
            self.disc_train_freq = max(self.disc_train_freq - 1, 2)

def _denormalize(tensor):
    return (tensor.permute(1, 2, 0).cpu().numpy() + 1) / 2

def compute_gradient_penalty(disc, blur, sharp, fake, device):
    alpha = torch.rand(sharp.size(0), 1, 1, 1).to(device).expand_as(sharp)
    interpolates = (alpha * sharp + (1 - alpha) * fake).requires_grad_(True)
    disc_interpolates = disc(blur, interpolates)
    gp = 0
    for pred in disc_interpolates:
        gradients = torch.autograd.grad(
            outputs=pred, inputs=interpolates,
            grad_outputs=torch.ones_like(pred),
            create_graph=True, retain_graph=True
        )[0]
        gp += ((gradients.norm(2, dim=[1, 2, 3]) - 1) ** 2).mean()
    return gp / len(disc_interpolates)

def calculate_disc_accuracy(real_preds, fake_preds):
    with torch.no_grad():
        real_correct = sum([(pred > 0).float().mean() for pred in real_preds])
        fake_correct = sum([(pred < 0).float().mean() for pred in fake_preds])
        return ((real_correct + fake_correct) / (len(real_preds) * 2)).item()

def train_one_epoch(gen, disc, train_loader, optimizers, losses, schedulers, adaptive_hp, config):
    gen.train()
    disc.train()
    epoch_gen_loss, epoch_disc_loss, epoch_disc_acc = 0.0, 0.0, 0.0
    step = 0
    
    lambda_l1, lambda_gan, disc_train_freq = adaptive_hp.lambda_l1, adaptive_hp.lambda_gan, adaptive_hp.disc_train_freq

    for blur, sharp in tqdm(train_loader, desc="Training"):
        blur, sharp = blur.to(config.DEVICE), sharp.to(config.DEVICE)
        fake = gen(blur)
        
        # Train Discriminator
        if step % disc_train_freq == 0:
            optimizers['disc'].zero_grad()
            d_real_preds = disc(blur, sharp)
            d_fake_preds = disc(blur, fake.detach())
            loss_d_real = sum([-torch.mean(pred) for pred in d_real_preds])
            loss_d_fake = sum([torch.mean(pred) for pred in d_fake_preds])
            gp = compute_gradient_penalty(disc, blur, sharp, fake.detach(), config.DEVICE)
            loss_d = (loss_d_real + loss_d_fake) / len(d_real_preds) + config.LAMBDA_GP * gp
            loss_d.backward()
            optimizers['disc'].step()
            epoch_disc_loss += loss_d.item()
        
        disc_acc = calculate_disc_accuracy(disc(blur, sharp), disc(blur, fake.detach()))
        epoch_disc_acc += disc_acc

        # Train Generator
        optimizers['gen'].zero_grad()
        d_fake_gen_preds = disc(blur, fake)
        loss_g_gan = sum([-torch.mean(pred) for pred in d_fake_gen_preds]) / len(d_fake_gen_preds)
        loss_g_l1 = losses['l1'](fake, sharp)
        loss_g_perceptual = losses['perceptual'](fake, sharp)
        loss_g = (lambda_gan * loss_g_gan) + (lambda_l1 * loss_g_l1) + (config.LAMBDA_PERCEPTUAL * loss_g_perceptual)
        loss_g.backward()
        optimizers['gen'].step()
        epoch_gen_loss += loss_g.item()
        step += 1
    
    schedulers['gen'].step()
    schedulers['disc'].step()
    
    avg_gen_loss = epoch_gen_loss / len(train_loader)
    avg_disc_loss = epoch_disc_loss / max(1, len(train_loader) // disc_train_freq)
    avg_disc_acc = epoch_disc_acc / len(train_loader)
    
    adaptive_hp.update_weights(avg_gen_loss, avg_disc_loss)
    adaptive_hp.update_training_freq(avg_disc_acc)
    
    return avg_gen_loss, avg_disc_loss, avg_disc_acc, adaptive_hp

def validate_one_epoch(gen, disc, val_loader, losses, adaptive_hp, config, epoch):
    gen.eval()
    disc.eval()
    val_psnr, val_ssim = 0.0, 0.0
    with torch.no_grad():
        for i, (blur_val, sharp_val) in enumerate(tqdm(val_loader, desc="Validating")):
            blur_val, sharp_val = blur_val.to(config.DEVICE), sharp_val.to(config.DEVICE)
            fake_val = gen(blur_val)
            
            fake_np = _denormalize(fake_val[0])
            sharp_np = _denormalize(sharp_val[0])
            val_psnr += psnr(sharp_np, fake_np, data_range=1.0)
            val_ssim += ssim(sharp_np, fake_np, channel_axis=-1, data_range=1.0)
            
            if i == 0: # Visualize first batch
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(_denormalize(blur_val[0])); axs[0].set_title('Blur')
                axs[1].imshow(fake_np); axs[1].set_title('Deblurred')
                axs[2].imshow(sharp_np); axs[2].set_title('Sharp')
                plt.savefig(config.VISUALIZATION_SAVE_PATH.format(epoch))
                plt.close()
                
    return val_psnr / len(val_loader), val_ssim / len(val_loader)