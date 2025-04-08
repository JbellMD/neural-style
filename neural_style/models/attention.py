"""
Attention-based neural style transfer model.

Based on the paper:
"Attention-aware Multi-stroke Style Transfer"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SelfAttention(nn.Module):
    """Self-attention module for style transfer."""
    
    def __init__(self, in_channels):
        """
        Initialize self-attention module.
        
        Args:
            in_channels (int): Number of input channels
        """
        super(SelfAttention, self).__init__()
        
        # Convolutions for query, key, and value
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Gamma parameter (learnable)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Output tensor with self-attention applied
        """
        batch_size, channels, height, width = x.size()
        
        # Calculate query, key, and value
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x (H*W) x C'
        key = self.key_conv(x).view(batch_size, -1, height * width)  # B x C' x (H*W)
        value = self.value_conv(x).view(batch_size, -1, height * width)  # B x C x (H*W)
        
        # Calculate attention map
        attention = torch.bmm(query, key)  # B x (H*W) x (H*W)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, channels, height, width)
        
        # Apply residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


class AttentionStyleTransfer(nn.Module):
    """
    Attention-based style transfer model.
    
    This model uses self-attention mechanisms to better capture
    and transfer style patterns across the image.
    """
    
    def __init__(self):
        """Initialize the attention-based style transfer model."""
        super(AttentionStyleTransfer, self).__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # Encoder - first part of VGG19
        self.encoder1 = nn.Sequential()
        self.encoder2 = nn.Sequential()
        self.encoder3 = nn.Sequential()
        self.encoder4 = nn.Sequential()
        self.encoder5 = nn.Sequential()
        
        # Split VGG into blocks
        for i in range(4):  # conv1_1 to relu1_2
            self.encoder1.add_module(str(i), vgg[i])
        
        for i in range(4, 9):  # conv2_1 to relu2_2
            self.encoder2.add_module(str(i), vgg[i])
        
        for i in range(9, 18):  # conv3_1 to relu3_4
            self.encoder3.add_module(str(i), vgg[i])
        
        for i in range(18, 27):  # conv4_1 to relu4_4
            self.encoder4.add_module(str(i), vgg[i])
        
        for i in range(27, 36):  # conv5_1 to relu5_4
            self.encoder5.add_module(str(i), vgg[i])
        
        # Freeze encoder parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Attention modules
        self.attention3 = SelfAttention(256)  # After encoder3
        self.attention4 = SelfAttention(512)  # After encoder4
        self.attention5 = SelfAttention(512)  # After encoder5
        
        # Decoder - mirrored structure of encoder with upsampling
        self.decoder5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        """
        Encode an image to extract hierarchical features.
        
        Args:
            x (torch.Tensor): Input image tensor
            
        Returns:
            tuple: Feature tensors at different levels
        """
        feat1 = self.encoder1(x)
        feat2 = self.encoder2(feat1)
        feat3 = self.encoder3(feat2)
        feat4 = self.encoder4(feat3)
        feat5 = self.encoder5(feat4)
        
        return feat1, feat2, feat3, feat4, feat5
    
    def decode(self, feat3, feat4, feat5):
        """
        Decode features to generate an image.
        
        Args:
            feat3 (torch.Tensor): Feature tensor from encoder3
            feat4 (torch.Tensor): Feature tensor from encoder4
            feat5 (torch.Tensor): Feature tensor from encoder5
            
        Returns:
            torch.Tensor: Output image tensor
        """
        # Apply attention to features
        att5 = self.attention5(feat5)
        d5 = self.decoder5(att5)
        
        att4 = self.attention4(feat4)
        d4 = self.decoder4(d5 + att4)
        
        att3 = self.attention3(feat3)
        d3 = self.decoder3(d4 + att3)
        
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)
        
        return d1
    
    def style_transfer(self, content, style, alpha=1.0):
        """
        Apply style transfer.
        
        Args:
            content (torch.Tensor): Content image tensor
            style (torch.Tensor): Style image tensor
            alpha (float): Style interpolation weight
            
        Returns:
            torch.Tensor: Styled image tensor
        """
        # Extract features
        content_feat1, content_feat2, content_feat3, content_feat4, content_feat5 = self.encode(content)
        style_feat1, style_feat2, style_feat3, style_feat4, style_feat5 = self.encode(style)
        
        # Blend content and style features
        if alpha < 1.0:
            # Interpolate between content and style features
            blend_feat3 = alpha * style_feat3 + (1 - alpha) * content_feat3
            blend_feat4 = alpha * style_feat4 + (1 - alpha) * content_feat4
            blend_feat5 = alpha * style_feat5 + (1 - alpha) * content_feat5
        else:
            # Use style features directly
            blend_feat3 = style_feat3
            blend_feat4 = style_feat4
            blend_feat5 = style_feat5
        
        # Decode to generate styled image
        styled_image = self.decode(content_feat3, content_feat4, blend_feat5)
        
        return styled_image
    
    def forward(self, content, style, alpha=1.0):
        """
        Forward pass for style transfer.
        
        Args:
            content (torch.Tensor): Content image tensor
            style (torch.Tensor): Style image tensor
            alpha (float): Style interpolation weight
            
        Returns:
            torch.Tensor: Styled image tensor
        """
        return self.style_transfer(content, style, alpha)


def train_attention_model(model, content_dataset, style_dataset, output_path, 
                         epochs=10, batch_size=4, learning_rate=1e-4, 
                         content_weight=1.0, style_weight=10.0, device=None):
    """
    Train an attention-based style transfer model.
    
    Args:
        model (AttentionStyleTransfer): Model to train
        content_dataset: Dataset of content images
        style_dataset: Dataset of style images
        output_path (str): Path to save the trained model
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        content_weight (float): Weight of content loss
        style_weight (float): Weight of style loss
        device (str): Device to use for training
        
    Returns:
        AttentionStyleTransfer: Trained model
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else 
                             "cpu")
    
    # Move model to device
    model = model.to(device)
    
    # Create data loaders
    content_loader = torch.utils.data.DataLoader(
        content_dataset, batch_size=batch_size, shuffle=True)
    style_loader = torch.utils.data.DataLoader(
        style_dataset, batch_size=batch_size, shuffle=True)
    
    # Create optimizer (only train decoder and attention modules)
    params = list(model.decoder1.parameters()) + \
             list(model.decoder2.parameters()) + \
             list(model.decoder3.parameters()) + \
             list(model.decoder4.parameters()) + \
             list(model.decoder5.parameters()) + \
             list(model.attention3.parameters()) + \
             list(model.attention4.parameters()) + \
             list(model.attention5.parameters())
    
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        
        # Iterate over batches
        for content_batch, style_batch in zip(content_loader, style_loader):
            # Get content and style images
            content_images = content_batch.to(device)
            style_images = style_batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Generate styled image
            styled_images = model(content_images, style_images)
            
            # Extract features for loss calculation
            content_feat1, content_feat2, content_feat3, content_feat4, content_feat5 = model.encode(content_images)
            style_feat1, style_feat2, style_feat3, style_feat4, style_feat5 = model.encode(style_images)
            styled_feat1, styled_feat2, styled_feat3, styled_feat4, styled_feat5 = model.encode(styled_images)
            
            # Content loss (using higher-level features)
            content_loss = F.mse_loss(styled_feat4, content_feat4)
            
            # Style loss (using gram matrices across multiple levels)
            style_loss = 0
            for sf, stf in zip([styled_feat1, styled_feat2, styled_feat3, styled_feat4, styled_feat5],
                              [style_feat1, style_feat2, style_feat3, style_feat4, style_feat5]):
                # Calculate gram matrices
                sf_gram = gram_matrix(sf)
                stf_gram = gram_matrix(stf)
                style_loss += F.mse_loss(sf_gram, stf_gram)
            
            # Total loss
            loss = content_weight * content_loss + style_weight * style_loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            count += 1
        
        # Print progress
        avg_loss = total_loss / count
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), output_path)
    
    return model


def gram_matrix(x):
    """
    Calculate gram matrix for style representation.
    
    Args:
        x (torch.Tensor): Feature tensor of shape [B, C, H, W]
        
    Returns:
        torch.Tensor: Gram matrix
    """
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram.div(c * h * w)
