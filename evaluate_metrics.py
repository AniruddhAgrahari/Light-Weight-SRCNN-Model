import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchmetrics.image
from tqdm import tqdm
from tabulate import tabulate

# Import our custom modules
from model_loader import load_model
from dataset import SRDatasetDownloader, SuperResolutionDataset

def evaluate_model(model, datasets_dir, scale_factor, batch_size=1, device=None):
    """
    Evaluate model on benchmark datasets using PSNR and SSIM metrics
    
    Args:
        model: Trained LW-SRCNN model
        datasets_dir: Directory containing benchmark datasets
        scale_factor: Super resolution scale factor
        batch_size: Batch size for evaluation (usually 1)
        device: Device to run evaluation on
        
    Returns:
        dict: Dictionary with metrics for each dataset
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize metrics
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    
    # Ensure dataset directory exists
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Download datasets if needed
    downloader = SRDatasetDownloader(datasets_dir)
    benchmark_datasets = ['Set5', 'Set14', 'BSD100', 'Urban100']
    dataset_paths = {}
    
    for dataset_name in benchmark_datasets:
        dataset_path = downloader.download_dataset(dataset_name)
        dataset_paths[dataset_name] = dataset_path
    
    # Evaluation results
    results = {}
    
    # Evaluate on each dataset
    for dataset_name, dataset_path in dataset_paths.items():
        print(f"\nEvaluating on {dataset_name}...")
        
        # Create dataset and dataloader
        test_dataset = SuperResolutionDataset(
            dataset_path,
            patch_size=0,  # No cropping for evaluation
            scale_factor=scale_factor,
            augment=False,
            split='val'
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Initialize metrics for this dataset
        total_psnr = 0.0
        total_ssim = 0.0
        count = 0
        
        # Process each image
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Processing {dataset_name}"):
                lr_imgs = batch['lr'].to(device)
                hr_imgs = batch['hr'].to(device)
                
                # Forward pass
                sr_imgs = model(lr_imgs)
                
                # Compute metrics
                psnr = psnr_metric(sr_imgs, hr_imgs)
                ssim = ssim_metric(sr_imgs, hr_imgs)
                
                total_psnr += psnr.item()
                total_ssim += ssim.item()
                count += 1
        
        # Reset metrics for next dataset
        psnr_metric.reset()
        ssim_metric.reset()
        
        # Calculate averages
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        
        results[dataset_name] = {
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }
        
        print(f"{dataset_name}: PSNR = {avg_psnr:.2f} dB, SSIM = {avg_ssim:.4f}")
    
    return results

def evaluate_with_bicubic(datasets_dir, scale_factor, batch_size=1, device=None):
    """
    Evaluate bicubic upsampling on benchmark datasets for comparison
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize metrics
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    
    # Initialize downloader
    downloader = SRDatasetDownloader(datasets_dir)
    benchmark_datasets = ['Set5', 'Set14', 'BSD100', 'Urban100']
    
    # Results for bicubic upsampling
    bicubic_results = {}
    
    for dataset_name in benchmark_datasets:
        print(f"\nEvaluating bicubic upsampling on {dataset_name}...")
        dataset_path = downloader.download_dataset(dataset_name)
        
        # Create dataset and dataloader
        test_dataset = SuperResolutionDataset(
            dataset_path,
            patch_size=0,  # No cropping for evaluation
            scale_factor=scale_factor,
            augment=False,
            split='val'
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Initialize metrics for this dataset
        total_psnr = 0.0
        total_ssim = 0.0
        count = 0
        
        # Process each image - compare bicubic upscaled LR with HR
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Processing {dataset_name}"):
                # Get bicubic upscaled images and HR images
                lr_imgs = batch['lr'].to(device)
                hr_imgs = batch['hr'].to(device)
                
                # Bicubic upscaling (using interpolate)
                bicubic_imgs = torch.nn.functional.interpolate(
                    lr_imgs,
                    scale_factor=scale_factor,
                    mode='bicubic',
                    align_corners=False
                ).clamp(0, 1)
                
                # Compute metrics
                psnr = psnr_metric(bicubic_imgs, hr_imgs)
                ssim = ssim_metric(bicubic_imgs, hr_imgs)
                
                total_psnr += psnr.item()
                total_ssim += ssim.item()
                count += 1
        
        # Reset metrics for next dataset
        psnr_metric.reset()
        ssim_metric.reset()
        
        # Calculate averages
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        
        bicubic_results[dataset_name] = {
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }
        
        print(f"{dataset_name} (Bicubic): PSNR = {avg_psnr:.2f} dB, SSIM = {avg_ssim:.4f}")
    
    return bicubic_results

def print_results_table(model_results, bicubic_results=None):
    """
    Print evaluation results in a clean table format
    """
    headers = ["Dataset", "PSNR (dB)", "SSIM"]
    
    if bicubic_results:
        headers = ["Dataset", "Bicubic PSNR", "Bicubic SSIM", "LW-SRCNN PSNR", "LW-SRCNN SSIM", "PSNR Gain", "SSIM Gain"]
    
    table_data = []
    
    for dataset_name in model_results.keys():
        if bicubic_results and dataset_name in bicubic_results:
            bicubic_psnr = bicubic_results[dataset_name]['psnr']
            bicubic_ssim = bicubic_results[dataset_name]['ssim']
            model_psnr = model_results[dataset_name]['psnr']
            model_ssim = model_results[dataset_name]['ssim']
            
            psnr_gain = model_psnr - bicubic_psnr
            ssim_gain = model_ssim - bicubic_ssim
            
            table_data.append([
                dataset_name,
                f"{bicubic_psnr:.2f}",
                f"{bicubic_ssim:.4f}",
                f"{model_psnr:.2f}",
                f"{model_ssim:.4f}",
                f"{psnr_gain:.2f}",
                f"{ssim_gain:.4f}"
            ])
        else:
            model_psnr = model_results[dataset_name]['psnr']
            model_ssim = model_results[dataset_name]['ssim']
            
            table_data.append([
                dataset_name,
                f"{model_psnr:.2f}",
                f"{model_ssim:.4f}"
            ])
    
    # Add average row if we have multiple datasets
    if len(model_results) > 1:
        if bicubic_results:
            avg_bicubic_psnr = sum(item['psnr'] for item in bicubic_results.values()) / len(bicubic_results)
            avg_bicubic_ssim = sum(item['ssim'] for item in bicubic_results.values()) / len(bicubic_results)
            avg_model_psnr = sum(item['psnr'] for item in model_results.values()) / len(model_results)
            avg_model_ssim = sum(item['ssim'] for item in model_results.values()) / len(model_results)
            
            avg_psnr_gain = avg_model_psnr - avg_bicubic_psnr
            avg_ssim_gain = avg_model_ssim - avg_bicubic_ssim
            
            table_data.append([
                "Average",
                f"{avg_bicubic_psnr:.2f}",
                f"{avg_bicubic_ssim:.4f}",
                f"{avg_model_psnr:.2f}",
                f"{avg_model_ssim:.4f}",
                f"{avg_psnr_gain:.2f}",
                f"{avg_ssim_gain:.4f}"
            ])
        else:
            avg_model_psnr = sum(item['psnr'] for item in model_results.values()) / len(model_results)
            avg_model_ssim = sum(item['ssim'] for item in model_results.values()) / len(model_results)
            
            table_data.append([
                "Average",
                f"{avg_model_psnr:.2f}",
                f"{avg_model_ssim:.4f}"
            ])
    
    # Print table
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    parser = argparse.ArgumentParser(description='Evaluate LW-SRCNN on benchmark datasets')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--scale', default=4, type=int, help='Super resolution scale factor')
    parser.add_argument('--data-dir', default='datasets', type=str, help='Dataset directory')
    parser.add_argument('--compare-bicubic', action='store_true', help='Compare with bicubic upsampling')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = load_model(args.checkpoint, scale_factor=args.scale, device=device)
    
    # Evaluate model
    model_results = evaluate_model(model, args.data_dir, args.scale, device=device)
    
    # Evaluate bicubic upsampling if requested
    bicubic_results = None
    if args.compare_bicubic:
        bicubic_results = evaluate_with_bicubic(args.data_dir, args.scale, device=device)
    
    # Print results
    print_results_table(model_results, bicubic_results)

if __name__ == '__main__':
    main()
