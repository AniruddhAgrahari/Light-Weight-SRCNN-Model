import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Import from our custom modules
from model_loader import load_model, preprocess_image

def visualize_super_resolution(image_path, model_path, scale_factor=4, output_path='comparison.png'):
    """
    Visualize super-resolution output from LW-SRCNN compared to bicubic interpolation
    
    Args:
        image_path: Path to low-resolution input image
        model_path: Path to trained LW-SRCNN model
        scale_factor: Super-resolution scale factor
        output_path: Path to save comparison image
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, scale_factor=scale_factor, device=device)
    
    # Open the input image
    original = Image.open(image_path).convert('RGB')
    print(f"Input image size: {original.width}x{original.height}")
    
    # Create input tensor for the model
    lr_tensor = preprocess_image(original, device)
    
    # Generate bicubic upscaled image
    bicubic_img = original.resize(
        (original.width * scale_factor, original.height * scale_factor), 
        Image.BICUBIC
    )
    print(f"Bicubic upscaled size: {bicubic_img.width}x{bicubic_img.height}")
    
    # Generate LW-SRCNN super-resolved image
    with torch.no_grad():
        sr_output = model(lr_tensor)
        sr_output = sr_output.clamp(0, 1)
        sr_img = transforms.ToPILImage()(sr_output.squeeze(0).cpu())
    print(f"Super-resolved size: {sr_img.width}x{sr_img.height}")
    
    # Create figure for comparison
    plt.figure(figsize=(15, 5))
    
    # Plot original low-res image
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title(f'Original Low-Res\n{original.width}x{original.height}')
    plt.axis('off')
    
    # Plot bicubic upscaled image
    plt.subplot(1, 3, 2)
    plt.imshow(bicubic_img)
    plt.title(f'Bicubic Interpolation\n{bicubic_img.width}x{bicubic_img.height}')
    plt.axis('off')
    
    # Plot super-resolved image
    plt.subplot(1, 3, 3)
    plt.imshow(sr_img)
    plt.title(f'LW-SRCNN Output\n{sr_img.width}x{sr_img.height}')
    plt.axis('off')
    
    # Save the comparison
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison to {output_path}")
    
    # Show plot if running interactively
    plt.show()
    
    # Save individual images
    base_name = os.path.splitext(output_path)[0]
    original.save(f"{base_name}_lowres.png")
    bicubic_img.save(f"{base_name}_bicubic.png")
    sr_img.save(f"{base_name}_lwsrcnn.png")
    print(f"Saved individual images with prefix: {base_name}_")
    
    return original, bicubic_img, sr_img

def main():
    parser = argparse.ArgumentParser(description='Visualize LW-SRCNN Super Resolution')
    parser.add_argument('--image', required=True, type=str, help='Path to input low-resolution image')
    parser.add_argument('--model', required=True, type=str, help='Path to trained LW-SRCNN model')
    parser.add_argument('--scale', default=4, type=int, help='Super-resolution scale factor')
    parser.add_argument('--output', default='comparison.png', type=str, help='Path to save comparison image')
    args = parser.parse_args()
    
    visualize_super_resolution(args.image, args.model, args.scale, args.output)

if __name__ == '__main__':
    main()
