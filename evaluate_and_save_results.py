import os
import csv
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_msssim import ssim
from lw_srcnn_model import LWSRCNNModel
from dataset import get_sr_datasets

def calculate_psnr(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) between two images"""
    # img1 and img2 have range [0, 1]
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_batch_psnr(sr, hr):
    """Calculate average PSNR for a batch of images"""
    batch_psnr = 0
    for i in range(sr.shape[0]):
        batch_psnr += calculate_psnr(sr[i], hr[i])
    return batch_psnr / sr.shape[0]

def save_evaluation_results(model, dataloader_dict, device, save_path='results.csv'):
    """
    Evaluate model on datasets and save results to CSV
    Columns: Dataset | Bicubic PSNR | Bicubic SSIM | LW-SRCNN PSNR | LW-SRCNN SSIM
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for dataset_name, dataloader in dataloader_dict.items():
            bicubic_psnr_total = 0
            bicubic_ssim_total = 0
            model_psnr_total = 0
            model_ssim_total = 0
            num_samples = 0
            
            for batch in tqdm(dataloader, desc=f'Evaluating {dataset_name}', ncols=100):
                lr_imgs = batch['lr'].to(device)
                hr_imgs = batch['hr'].to(device)
                
                # Generate bicubic interpolation
                bicubic_imgs = F.interpolate(
                    lr_imgs, size=hr_imgs.shape[2:], 
                    mode='bicubic', align_corners=False)
                
                # Model prediction
                sr_imgs = model(lr_imgs)
                
                # Calculate PSNR
                bicubic_psnr = calculate_batch_psnr(bicubic_imgs, hr_imgs)
                model_psnr = calculate_batch_psnr(sr_imgs, hr_imgs)
                
                # Calculate SSIM
                bicubic_ssim = ssim(bicubic_imgs, hr_imgs, data_range=1.0)
                model_ssim = ssim(sr_imgs, hr_imgs, data_range=1.0)
                
                # Accumulate results
                batch_size = lr_imgs.size(0)
                bicubic_psnr_total += bicubic_psnr * batch_size
                bicubic_ssim_total += bicubic_ssim.item() * batch_size
                model_psnr_total += model_psnr * batch_size
                model_ssim_total += model_ssim.item() * batch_size
                num_samples += batch_size
            
            # Calculate averages
            bicubic_psnr_avg = bicubic_psnr_total / num_samples
            bicubic_ssim_avg = bicubic_ssim_total / num_samples
            model_psnr_avg = model_psnr_total / num_samples
            model_ssim_avg = model_ssim_total / num_samples
            
            # Store results
            results.append({
                'Dataset': dataset_name,
                'Bicubic PSNR': f"{bicubic_psnr_avg:.2f}",
                'Bicubic SSIM': f"{bicubic_ssim_avg:.4f}",
                'LW-SRCNN PSNR': f"{model_psnr_avg:.2f}",
                'LW-SRCNN SSIM': f"{model_ssim_avg:.4f}"
            })
    
    # Write results to CSV
    with open(save_path, 'w', newline='') as csvfile:
        fieldnames = ['Dataset', 'Bicubic PSNR', 'Bicubic SSIM', 'LW-SRCNN PSNR', 'LW-SRCNN SSIM']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"Results saved to {save_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate LW-SRCNN and save results to CSV')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--data-dir', default='datasets', type=str, help='Dataset directory')
    parser.add_argument('--scale', default=4, type=int, help='Super resolution scale factor')
    parser.add_argument('--output', default='results.csv', type=str, help='Output CSV file path')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get datasets
    data_loaders = get_sr_datasets(
        data_dir=args.data_dir,
        patch_size=0,  # Use full images for evaluation
        scale_factor=args.scale,
        batch_size=1,  # Process one image at a time for evaluation
        train=False    # Only get validation datasets
    )
    val_loaders = data_loaders['val_loaders']
    
    # Initialize model
    model = LWSRCNNModel(upscale_factor=args.scale)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Evaluate and save results
    save_evaluation_results(model, val_loaders, device, args.output)

if __name__ == '__main__':
    main()
