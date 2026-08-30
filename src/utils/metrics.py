"""
Evaluation metrics for Construction Site AI Reconstruction and Generation.
"""
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def compute_reconstruction_metrics(real_imgs_np, recon_imgs_np):
    """
    Computes MSE, SSIM, and PSNR across a batch of images.
    
    Args:
        real_imgs_np: numpy array of shape [B, H, W, C] in range [0, 1]
        recon_imgs_np: numpy array of shape [B, H, W, C] in range [0, 1]
        
    Returns:
        dict containing mean and std for MSE, SSIM, and PSNR.
    """
    B = real_imgs_np.shape[0]
    mse_list, ssim_list, psnr_list = [], [], []

    for i in range(B):
        real = real_imgs_np[i]
        recon = recon_imgs_np[i]
        
        mse = np.mean((real - recon) ** 2)
        ssim_val = ssim_fn(real, recon, multichannel=True, channel_axis=2, data_range=1.0)
        psnr_val = psnr_fn(real, recon, data_range=1.0)
        
        mse_list.append(mse)
        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)

    return {
        'mse_mean': float(np.mean(mse_list)),
        'mse_std': float(np.std(mse_list)),
        'ssim_mean': float(np.mean(ssim_list)),
        'ssim_std': float(np.std(ssim_list)),
        'psnr_mean': float(np.mean(psnr_list)),
        'psnr_std': float(np.std(psnr_list)),
    }
