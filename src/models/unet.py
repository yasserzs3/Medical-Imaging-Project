"""UNet model implementation for medical image segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and dropout."""
    
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.0):
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + out_channels, out_channels, mid_channels=out_channels, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels, mid_channels=out_channels, dropout_rate=dropout_rate)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad x1 to match x2 dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution."""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net model for image segmentation."""
    
    def __init__(self, in_channels=1, n_classes=1, bilinear=True, features=[32, 64, 128, 256], dropout_rate=0.2):
        """
        Initialize U-Net model.
        
        Args:
            in_channels (int): Number of input channels
            n_classes (int): Number of output channels/classes
            bilinear (bool): Whether to use bilinear upsampling
            features (list): Feature sizes for each level
            dropout_rate (float): Dropout rate
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self._printed_debug = False
        
        # Initial double convolution
        self.inc = DoubleConv(in_channels, features[0])
        
        # Downsampling path
        self.down1 = Down(features[0], features[1], dropout_rate)
        self.down2 = Down(features[1], features[2], dropout_rate)
        self.down3 = Down(features[2], features[3], dropout_rate)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor, dropout_rate)
        
        # Upsampling path with skip connections
        bottleneck_channels = features[3] * 2 // factor
        self.up1 = Up(bottleneck_channels, features[3], bilinear, dropout_rate)
        self.up2 = Up(features[3], features[2], bilinear, dropout_rate)
        self.up3 = Up(features[2], features[1], bilinear, dropout_rate)
        self.up4 = Up(features[1], features[0], bilinear, dropout_rate)
        
        # Output convolution
        self.outc = OutConv(features[0], n_classes)
    
    def forward(self, x):
        """Forward pass of the U-Net model."""
        # Print input tensor shape and stats for debugging
        if not hasattr(self, '_printed_debug') or not self._printed_debug:
            print(f"Model input shape: {x.shape}, dtype: {x.dtype}, "
                  f"range: [{x.min().item()}, {x.max().item()}]")
            self._printed_debug = True
            
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final convolution
        logits = self.outc(x)
        
        return logits


def build_unet(in_channels=1, n_classes=1, bilinear=True):
    """
    Build a U-Net model with default parameters.
    
    Args:
        in_channels (int): Number of input channels
        n_classes (int): Number of output classes
        bilinear (bool): Whether to use bilinear upsampling
    
    Returns:
        UNet: The constructed U-Net model
    """
    # Default feature sizes for each level
    features = [32, 64, 128, 256]
    # Moderate dropout rate
    dropout_rate = 0.2
    
    print(f"Building UNet with in_channels={in_channels}, n_classes={n_classes}, "
          f"features={features}, dropout_rate={dropout_rate}")
    
    model = UNet(
        in_channels=in_channels, 
        n_classes=n_classes, 
        bilinear=bilinear,
        features=features,
        dropout_rate=dropout_rate
    )
    return model 