import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution layer that reduces parameters by factorizing
    standard convolution into a depthwise and pointwise convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LWSRCNNModel(nn.Module):
    """
    Lightweight Super-Resolution CNN model optimized for speed and efficiency
    
    Args:
        upscale_factor (int): Factor to upscale the image (default: 4)
        num_channels (int): Number of input image channels (default: 3 for RGB)
    """
    def __init__(self, upscale_factor=4, num_channels=3):
        super(LWSRCNNModel, self).__init__()
        
        # Feature extraction block
        self.feature_extraction = nn.Sequential(
            DepthwiseSeparableConv(num_channels, 28, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(28)
        )
        
        # Non-linear mapping block - compact design with few layers
        self.mapping = nn.Sequential(
            DepthwiseSeparableConv(28, 28, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(28),
            DepthwiseSeparableConv(28, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        
        # Efficient upscaling block using pixel shuffle (more efficient than transposed convs)
        self.upscaling = nn.Sequential(
            nn.Conv2d(16, num_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )
    
    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x: Input low-resolution image
            
        Returns:
            Super-resolved high-resolution image
        """
        features = self.feature_extraction(x)
        mapped = self.mapping(features)
        upscaled = self.upscaling(mapped)
        return upscaled
