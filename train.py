import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import our custom modules
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

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_psnr = 0
    
    with tqdm(total=len(dataloader), desc='Training', ncols=100) as pbar:
        for batch in dataloader:
            # Get LR and HR images
            lr_imgs = batch['lr'].to(device)
            hr_imgs = batch['hr'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            sr_imgs = model(lr_imgs)
            
            # Calculate loss
            loss = criterion(sr_imgs, hr_imgs)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate PSNR for monitoring
            with torch.no_grad():
                batch_psnr = calculate_batch_psnr(sr_imgs, hr_imgs)
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_psnr += batch_psnr
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{batch_psnr:.2f}")
            
    return epoch_loss / len(dataloader), epoch_psnr / len(dataloader)

def validate(model, dataloader_dict, device):
    """Validate model on validation sets"""
    model.eval()
    results = {}
    
    with torch.no_grad():
        for dataset_name, dataloader in dataloader_dict.items():
            total_psnr = 0
            for batch in tqdm(dataloader, desc=f'Validating on {dataset_name}', ncols=100):
                lr_imgs = batch['lr'].to(device)
                hr_imgs = batch['hr'].to(device)
                
                # Forward pass
                sr_imgs = model(lr_imgs)
                
                # Calculate PSNR
                psnr = calculate_psnr(sr_imgs, hr_imgs)
                total_psnr += psnr.item()
            
            avg_psnr = total_psnr / len(dataloader)
            results[dataset_name] = avg_psnr
            
    return results

def main():
    parser = argparse.ArgumentParser(description='Train LW-SRCNN')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train')
    parser.add_argument('--batch-size', default=16, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--scale', default=4, type=int, help='Super resolution scale factor')
    parser.add_argument('--patch-size', default=96, type=int, help='HR patch size')
    parser.add_argument('--data-dir', default='datasets', type=str, help='Dataset directory')
    parser.add_argument('--save-dir', default='checkpoints', type=str, help='Checkpoint directory')
    parser.add_argument('--loss', default='mse', type=str, choices=['mse', 'l1'], help='Loss function')
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get datasets
    data_loaders = get_sr_datasets(
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        scale_factor=args.scale,
        batch_size=args.batch_size
    )
    train_loader = data_loaders['train_loader']
    val_loaders = data_loaders['val_loaders']
    
    # Initialize model
    model = LWSRCNNModel(upscale_factor=args.scale)
    model = model.to(device)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    # Define loss function
    if args.loss == 'mse':
        criterion = nn.MSELoss()
        print("Using MSE Loss")
    else:
        criterion = nn.L1Loss()
        print("Using L1 Loss")
    
    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    best_psnr = 0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch: {epoch}/{args.epochs}")
        
        # Train for one epoch
        start_time = time.time()
        train_loss, train_psnr = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate on all validation datasets
        val_results = validate(model, val_loaders, device)
        
        # Calculate average PSNR across all validation datasets
        avg_psnr = sum(val_results.values()) / len(val_results)
        end_time = time.time()
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f} dB")
        for dataset_name, psnr in val_results.items():
            print(f"{dataset_name} PSNR: {psnr:.2f} dB")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Time: {end_time - start_time:.2f}s")
        
        # Update learning rate
        scheduler.step(avg_psnr)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
            }, os.path.join(args.save_dir, f'lw_srcnn_best_x{args.scale}.pth'))
            print(f"Saved best model with PSNR: {best_psnr:.2f} dB")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': avg_psnr,
            }, os.path.join(args.save_dir, f'lw_srcnn_epoch_{epoch}_x{args.scale}.pth'))
    
    print(f"\nTraining completed. Best PSNR: {best_psnr:.2f} dB at epoch {best_epoch}")

if __name__ == '__main__':
    main()
