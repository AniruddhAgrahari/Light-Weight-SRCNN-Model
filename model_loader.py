import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from lw_srcnn_model import LWSRCNNModel

def load_model(model_path, scale_factor=4, device=None):
    """
    Load a trained LW-SRCNN model and prepare it for evaluation
    
    Args:
        model_path: Path to the saved model checkpoint
        scale_factor: The super resolution scale factor the model was trained for
        device: Device to run the model on (None for auto-detection)
        
    Returns:
        model: The loaded model ready for inference
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model on {device}")
    
    # Initialize model architecture
    model = LWSRCNNModel(upscale_factor=scale_factor)
    
    # Load weights from checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Loaded model from epoch {epoch}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model weights (no epoch information)")
            
        # Move model to device
        model = model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        print("Model set to evaluation mode")
        
        # Calculate and print number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")
        
        return model
    
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

def preprocess_image(image, device=None):
    """
    Preprocess an image for inference with LW-SRCNN model
    
    Args:
        image: PIL image or path to image
        device: Device to run inference on (None for auto-detection)
        
    Returns:
        tensor: Preprocessed image tensor ready for model input
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    # Convert to tensor and normalize (0-1 range)
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(image)
    
    # Add batch dimension and move to device
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    return img_tensor

def get_inference_function(model, device=None):
    """
    Create a simple inference function that handles preprocessing
    
    Args:
        model: The loaded LW-SRCNN model
        device: Device to run inference on
        
    Returns:
        function: A function that takes a PIL image and returns a super-resolved image
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def infer(image):
        """
        Run inference on an image
        
        Args:
            image: PIL image or path to image
            
        Returns:
            PIL Image: Super-resolved image
        """
        # Preprocess
        input_tensor = preprocess_image(image, device)
        
        # Run inference
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Convert back to PIL image
        output_tensor = output_tensor.clamp(0, 1)
        output_tensor = output_tensor.squeeze(0).cpu()
        output_image = transforms.ToPILImage()(output_tensor)
        
        return output_image
    
    return infer

if __name__ == "__main__":
    # Example usage
    model_path = "checkpoints/best_lwsrcnn.pth"
    
    try:
        # Load the model
        model = load_model(model_path)
        
        print("\nModel loaded successfully. Ready for evaluation.")
        
        # Create inference function
        sr_func = get_inference_function(model)
        print("Created inference function. Use sr_func(image) for super-resolution.")
        
        # Example preprocessing
        print("\nTo preprocess an image:")
        print("input_tensor = preprocess_image('path/to/image.jpg')")
        print("With torch.no_grad():")
        print("    output_tensor = model(input_tensor)")
        
    except Exception as e:
        print(f"Error in example: {e}")
