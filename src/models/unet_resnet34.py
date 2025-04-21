import torch
import torch.nn as nn
import torchvision.models as models


class UNetResNet34(nn.Module):
    """U-Net with pre-trained ResNet-34 encoder"""
    
    def __init__(self, in_channels=3, n_classes=1, pretrained=True):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        # Load pre-trained ResNet-34 as encoder
        resnet = models.resnet34(pretrained=pretrained)
        
        # Encoder (ResNet-34 layers)
        if in_channels != 3:
            # Replace first conv if input channels != 3
            self.firstconv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.firstconv = resnet.conv1
        
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1  # 64 channels
        self.encoder2 = resnet.layer2  # 128 channels
        self.encoder3 = resnet.layer3  # 256 channels
        self.encoder4 = resnet.layer4  # 512 channels
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self._decoder_block(512, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self._decoder_block(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self._decoder_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = self._decoder_block(96, 32)  # 64 + 32 = 96
        
        # Final upsampling and output
        self.upconv0 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder0 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(16, n_classes, kernel_size=1)
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_init = x  # Save for later skip connection
        
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)  # 64 channels
        e2 = self.encoder2(e1)  # 128 channels
        e3 = self.encoder3(e2)  # 256 channels
        e4 = self.encoder4(e3)  # 512 channels
        
        # Decoder with skip connections
        d4 = self.upconv4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, x_init], dim=1)
        d1 = self.decoder1(d1)
        
        d0 = self.upconv0(d1)
        d0 = self.decoder0(d0)
        
        output = self.final(d0)
        
        return output
    
    def freeze_encoder(self):
        """Freeze encoder parameters to prevent updating during training."""
        for param in self.firstconv.parameters():
            param.requires_grad = False
        for param in self.firstbn.parameters():
            param.requires_grad = False
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False
        for param in self.encoder3.parameters():
            param.requires_grad = False
        for param in self.encoder4.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for fine-tuning."""
        for param in self.firstconv.parameters():
            param.requires_grad = True
        for param in self.firstbn.parameters():
            param.requires_grad = True
        for param in self.encoder1.parameters():
            param.requires_grad = True
        for param in self.encoder2.parameters():
            param.requires_grad = True
        for param in self.encoder3.parameters():
            param.requires_grad = True
        for param in self.encoder4.parameters():
            param.requires_grad = True


def build(in_channels=3, n_classes=1, pretrained=True, freeze_epochs=0):
    """
    Build a U-Net model with pre-trained ResNet-34 encoder.
    
    Args:
        in_channels (int): Number of input channels
        n_classes (int): Number of output classes
        pretrained (bool): Whether to use pre-trained weights
        freeze_epochs (int): Number of epochs to freeze encoder
    
    Returns:
        nn.Module: UNetResNet34 model
        int: Number of epochs to freeze encoder
    """
    model = UNetResNet34(in_channels=in_channels, n_classes=n_classes, pretrained=pretrained)
    
    # Freeze encoder if required
    if freeze_epochs > 0:
        model.freeze_encoder()
    
    return model, freeze_epochs 