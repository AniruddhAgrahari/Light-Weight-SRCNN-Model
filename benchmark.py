import os
import time
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from pytorch_msssim import ssim
from lw_srcnn_model import LWSRCNNModel
from dataset import get_sr_datasets

class SRCNN(torch.nn.Module):
    """Original SRCNN model for comparison"""
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def benchmark_model(model, model_name, device, dataloader, num_runs=10):
    """Benchmark model inference speed and calculate metrics"""
    model.to(device)
    model.eval()
    
    total_time = 0
    total_psnr = 0
    total_ssim = 0
    total_images = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Benchmarking {model_name} on {device}'):
            lr_imgs = batch['lr'].to(device)
            hr_imgs = batch['hr'].to(device)
            
            # Warm-up run
            _ = model(lr_imgs)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            
            # Time inference (multiple runs for more accurate timing)
            start_time = time.time()
            for _ in range(num_runs):
                sr_imgs = model(lr_imgs)
                torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            # Average time per run
            inference_time = (end_time - start_time) / num_runs
            total_time += inference_time
            
            # Calculate metrics
            psnr_val = calculate_psnr(sr_imgs, hr_imgs)
            ssim_val = ssim(sr_imgs, hr_imgs, data_range=1.0)
            
            total_psnr += psnr_val.item()
            total_ssim += ssim_val.item()
            total_images += 1
    
    # Calculate averages
    avg_time = total_time / total_images * 1000  # convert to milliseconds
    avg_psnr = total_psnr / total_images
    avg_ssim = total_ssim / total_images
    
    return {
        'Inference Time (ms)': avg_time,
        'PSNR': avg_psnr,
        'SSIM': avg_ssim
    }

def bicubic_benchmark(device, dataloader, num_runs=10):
    """Benchmark bicubic interpolation"""
    total_time = 0
    total_psnr = 0
    total_ssim = 0
    total_images = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Benchmarking Bicubic on {device}'):
            lr_imgs = batch['lr'].to(device)
            hr_imgs = batch['hr'].to(device)
            
            # Warm-up run
            _ = F.interpolate(lr_imgs, size=hr_imgs.shape[2:], mode='bicubic', align_corners=False)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            
            # Time inference
            start_time = time.time()
            for _ in range(num_runs):
                bicubic_imgs = F.interpolate(lr_imgs, size=hr_imgs.shape[2:], 
                                            mode='bicubic', align_corners=False)
                torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            # Average time per run
            inference_time = (end_time - start_time) / num_runs
            total_time += inference_time
            
            # Calculate metrics
            psnr_val = calculate_psnr(bicubic_imgs, hr_imgs)
            ssim_val = ssim(bicubic_imgs, hr_imgs, data_range=1.0)
            
            total_psnr += psnr_val.item()
            total_ssim += ssim_val.item()
            total_images += 1
    
    # Calculate averages
    avg_time = total_time / total_images * 1000  # convert to milliseconds
    avg_psnr = total_psnr / total_images
    avg_ssim = total_ssim / total_images
    
    return {
        'Inference Time (ms)': avg_time,
        'PSNR': avg_psnr,
        'SSIM': avg_ssim
    }

def run_benchmarks(dataloaders, lwsrcnn_path, srcnn_path, output_path):
    """Run all benchmarks and save results"""
    results = {}
    
    # Initialize models
    lwsrcnn = LWSRCNNModel(upscale_factor=4)
    srcnn = SRCNN()
    
    # Load model weights
    lwsrcnn.load_state_dict(torch.load(lwsrcnn_path, map_location='cpu')['model_state_dict'])
    
    if os.path.exists(srcnn_path):
        srcnn.load_state_dict(torch.load(srcnn_path, map_location='cpu')['model_state_dict'])
    else:
        print(f"Warning: SRCNN checkpoint not found at {srcnn_path}")
    
    # Create dummy model wrapper for bicubic
    class BicubicWrapper(torch.nn.Module):
        def forward(self, x):
            return F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
    
    bicubic = BicubicWrapper()
    
    # Test devices
    devices = []
    devices.append(torch.device('cpu'))
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))
    
    # Run benchmarks for each dataset
    for dataset_name, dataloader in dataloaders.items():
        print(f"\n=== Benchmarking on {dataset_name} ===")
        results[dataset_name] = {}
        
        for device in devices:
            device_name = device.type
            print(f"\nRunning on {device_name.upper()}")
            
            # Benchmark Bicubic
            bicubic_results = bicubic_benchmark(device, dataloader)
            results[dataset_name][f'Bicubic ({device_name})'] = bicubic_results
            
            # Benchmark SRCNN
            srcnn_results = benchmark_model(srcnn, 'SRCNN', device, dataloader)
            results[dataset_name][f'SRCNN ({device_name})'] = srcnn_results
            
            # Benchmark LW-SRCNN
            lwsrcnn_results = benchmark_model(lwsrcnn, 'LW-SRCNN', device, dataloader)
            results[dataset_name][f'LW-SRCNN ({device_name})'] = lwsrcnn_results
            
            # Print results
            print(f"Bicubic - Time: {bicubic_results['Inference Time (ms)']:.2f}ms, PSNR: {bicubic_results['PSNR']:.2f}, SSIM: {bicubic_results['SSIM']:.4f}")
            print(f"SRCNN - Time: {srcnn_results['Inference Time (ms)']:.2f}ms, PSNR: {srcnn_results['PSNR']:.2f}, SSIM: {srcnn_results['SSIM']:.4f}")
            print(f"LW-SRCNN - Time: {lwsrcnn_results['Inference Time (ms)']:.2f}ms, PSNR: {lwsrcnn_results['PSNR']:.2f}, SSIM: {lwsrcnn_results['SSIM']:.4f}")
    
    # Format and save results
    formatted_results = []
    
    for dataset_name in results.keys():
        for method_device in results[dataset_name].keys():
            method, device = method_device.split(' (')
            device = device.rstrip(')')
            result = results[dataset_name][method_device]
            
            formatted_results.append({
                'Dataset': dataset_name,
                'Method': method,
                'Device': device,
                'PSNR': f"{result['PSNR']:.2f}",
                'SSIM': f"{result['SSIM']:.4f}",
                'Inference Time (ms)': f"{result['Inference Time (ms)']:.2f}"
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(formatted_results)
    df.to_csv(output_path, index=False)
    print(f"\nBenchmark results saved to {output_path}")
    
    # Also save as a more readable format
    html_path = output_path.replace('.csv', '.html')
    pivot_table = pd.pivot_table(
        df, 
        values=['PSNR', 'SSIM', 'Inference Time (ms)'],
        index=['Dataset', 'Method'],
        columns='Device'
    )
    
    with open(html_path, 'w') as f:
        f.write(pivot_table.to_html())
    print(f"Formatted benchmark results saved to {html_path}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark SR models')
    parser.add_argument('--lwsrcnn', required=True, type=str, help='Path to LW-SRCNN checkpoint')
    parser.add_argument('--srcnn', required=False, type=str, default='checkpoints/srcnn.pth', help='Path to SRCNN checkpoint')
    parser.add_argument('--data-dir', default='datasets', type=str, help='Dataset directory')
    parser.add_argument('--output', default='benchmark_results.csv', type=str, help='Output file path')
    args = parser.parse_args()
    
    # Get datasets
    data_loaders = get_sr_datasets(
        data_dir=args.data_dir,
        patch_size=0,  # Use full images for evaluation
        scale_factor=4,
        batch_size=1,  # Process one image at a time for evaluation
        train=False    # Only get validation datasets
    )
    val_loaders = data_loaders['val_loaders']
    
    # Run benchmarks
    run_benchmarks(val_loaders, args.lwsrcnn, args.srcnn, args.output)

if __name__ == '__main__':
    main()
