import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torchmetrics.image

# Import custom modules
from lw_srcnn_model import LWSRCNNModel
from model_loader import load_model, preprocess_image

class SRCNN(nn.Module):
    """Original SRCNN model architecture"""
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def load_srcnn_model(model_path, device=None):
    """Load a pre-trained SRCNN model"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SRCNN()
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load SRCNN model: {e}")
        return None

def bicubic_upscale(img_path, scale_factor):
    """Upscale image using OpenCV bicubic interpolation"""
    # Read image using OpenCV
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not open image: {img_path}")
    
    # Get dimensions
    h, w = img.shape[:2]
    
    # Resize with bicubic interpolation
    upscaled = cv2.resize(img, (w * scale_factor, h * scale_factor), 
                          interpolation=cv2.INTER_CUBIC)
    
    # Convert BGR to RGB
    upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
    return upscaled_rgb

def calculate_metrics(img1, img2):
    """Calculate PSNR and SSIM between two images"""
    # Ensure images are numpy arrays
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Calculate PSNR
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if np.max(img1) > 1 else 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Calculate SSIM
    ssim_value = ssim(img1, img2, multichannel=True, 
                      data_range=255.0 if np.max(img1) > 1 else 1.0)
    
    return psnr, ssim_value

def prepare_for_metrics(img):
    """Prepare image for metric calculation"""
    if isinstance(img, torch.Tensor):
        # Convert tensor to numpy
        img = img.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Ensure image is in RGB format
    if img.shape[-1] == 4:  # RGBA
        img = img[:, :, :3]
    
    # Ensure values are in correct range
    if np.max(img) <= 1.0:
        img = img * 255.0
        
    return img

def compare_methods(
    image_path, 
    lw_srcnn_model_path, 
    srcnn_model_path=None, 
    scale_factor=4, 
    output_dir=None
):
    """Compare bicubic interpolation, SRCNN, and LW-SRCNN on a single image"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load LW-SRCNN model
    lw_srcnn = load_model(lw_srcnn_model_path, scale_factor=scale_factor, device=device)
    
    # Load SRCNN model if path provided
    srcnn = None
    if srcnn_model_path:
        srcnn = load_srcnn_model(srcnn_model_path, device=device)
    
    # Check if image exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read original image for reference
    original = Image.open(image_path).convert('RGB')
    ori_w, ori_h = original.size
    
    # Create low-resolution version (for reference, not for processing)
    low_res = original.resize((ori_w // scale_factor, ori_h // scale_factor), Image.BICUBIC)
    low_res_tensor = transforms.ToTensor()(low_res).unsqueeze(0).to(device)
    
    # Method 1: Bicubic Interpolation
    bicubic_np = bicubic_upscale(image_path, scale_factor)
    bicubic_pil = Image.fromarray(bicubic_np)
    bicubic_tensor = transforms.ToTensor()(bicubic_pil).to(device)
    
    # Method 2: SRCNN (if available)
    srcnn_result = None
    if srcnn:
        # SRCNN typically uses bicubic upsampled input
        with torch.no_grad():
            srcnn_input = transforms.ToTensor()(bicubic_pil).unsqueeze(0).to(device)
            srcnn_output = srcnn(srcnn_input).squeeze(0).clamp(0, 1)
            srcnn_result = srcnn_output
    
    # Method 3: LW-SRCNN
    with torch.no_grad():
        lw_srcnn_output = lw_srcnn(low_res_tensor).squeeze(0).clamp(0, 1)
    
    # Original high-resolution image as tensor
    original_tensor = transforms.ToTensor()(original).to(device)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = {}
    
    # Prepare images for metric calculation (ensure consistent formats)
    hr_np = prepare_for_metrics(original_tensor)
    bicubic_np = prepare_for_metrics(bicubic_tensor)
    lw_srcnn_np = prepare_for_metrics(lw_srcnn_output)
    
    # Calculate metrics for bicubic
    bicubic_psnr, bicubic_ssim = calculate_metrics(hr_np, bicubic_np)
    metrics['Bicubic'] = {'PSNR': bicubic_psnr, 'SSIM': bicubic_ssim}
    
    # Calculate metrics for LW-SRCNN
    lw_srcnn_psnr, lw_srcnn_ssim = calculate_metrics(hr_np, lw_srcnn_np)
    metrics['LW-SRCNN'] = {'PSNR': lw_srcnn_psnr, 'SSIM': lw_srcnn_ssim}
    
    # Calculate metrics for SRCNN if available
    if srcnn_result is not None:
        srcnn_np = prepare_for_metrics(srcnn_result)
        srcnn_psnr, srcnn_ssim = calculate_metrics(hr_np, srcnn_np)
        metrics['SRCNN'] = {'PSNR': srcnn_psnr, 'SSIM': srcnn_ssim}
    
    # Print results table
    print("\nResults comparison:")
    headers = ['Method', 'PSNR (dB)', 'SSIM']
    rows = []
    for method, values in metrics.items():
        rows.append([method, f"{values['PSNR']:.2f}", f"{values['SSIM']:.4f}"])
    
    # Print table using formatted strings
    row_format = "{:<10} {:<12} {:<12}"
    print(row_format.format(*headers))
    print("-" * 34)
    for row in rows:
        print(row_format.format(*row))
    
    # Visualize results
    num_methods = 2 + (1 if srcnn_result is not None else 0)  # bicubic, LW-SRCNN, and optionally SRCNN
    fig, axes = plt.subplots(1, num_methods + 2, figsize=(4*(num_methods + 2), 6))
    
    # Original and low-resolution
    axes[0].imshow(low_res)
    axes[0].set_title('Low Resolution')
    axes[0].axis('off')
    
    axes[1].imshow(bicubic_pil)
    axes[1].set_title(f'Bicubic\nPSNR: {bicubic_psnr:.2f}dB\nSSIM: {bicubic_ssim:.4f}')
    axes[1].axis('off')
    
    col_idx = 2
    
    if srcnn_result is not None:
        srcnn_img = transforms.ToPILImage()(srcnn_result.cpu())
        axes[col_idx].imshow(srcnn_img)
        axes[col_idx].set_title(f'SRCNN\nPSNR: {metrics["SRCNN"]["PSNR"]:.2f}dB\nSSIM: {metrics["SRCNN"]["SSIM"]:.4f}')
        axes[col_idx].axis('off')
        col_idx += 1
    
    # LW-SRCNN
    lw_srcnn_img = transforms.ToPILImage()(lw_srcnn_output.cpu())
    axes[col_idx].imshow(lw_srcnn_img)
    axes[col_idx].set_title(f'LW-SRCNN\nPSNR: {lw_srcnn_psnr:.2f}dB\nSSIM: {lw_srcnn_ssim:.4f}')
    axes[col_idx].axis('off')
    col_idx += 1
    
    # Original high-resolution
    axes[col_idx].imshow(original)
    axes[col_idx].set_title('Original HR')
    axes[col_idx].axis('off')
    
    plt.tight_layout()
    
    # Save or display
    if output_dir:
        fig_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_comparison.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison saved to: {fig_path}")
        
        # Save individual results
        low_res.save(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_lowres.png"))
        bicubic_pil.save(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_bicubic.png"))
        lw_srcnn_img.save(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_lw_srcnn.png"))
        
        if srcnn_result is not None:
            srcnn_img.save(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_srcnn.png"))
    else:
        plt.show()
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Compare Super-Resolution Methods')
    parser.add_argument('--image', required=True, type=str, help='Path to image')
    parser.add_argument('--lw-srcnn', required=True, type=str, help='Path to LW-SRCNN model')
    parser.add_argument('--srcnn', type=str, help='Path to SRCNN model (optional)')
    parser.add_argument('--scale', default=4, type=int, help='Scale factor')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    args = parser.parse_args()
    
    compare_methods(
        args.image, 
        args.lw_srcnn, 
        args.srcnn, 
        args.scale, 
        args.output_dir
    )

if __name__ == '__main__':
    main()
