"""
AdaIN (Adaptive Instance Normalization) model for neural style transfer.

Based on the paper:
"Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"
by Xun Huang and Serge Belongie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AdaINModel(nn.Module):
    """
    AdaIN model for fast and flexible style transfer.
    
    This model uses adaptive instance normalization to transfer style from
    a style image to a content image in a single forward pass.
    """
    
    def __init__(self, encoder=None, decoder=None):
        """
        Initialize the AdaIN model.
        
        Args:
            encoder (nn.Module): Encoder network (VGG by default)
            decoder (nn.Module): Decoder network
        """
        super(AdaINModel, self).__init__()
        
        # Initialize encoder (VGG-based)
        self.encoder = encoder or self._build_encoder()
        
        # Initialize decoder
        self.decoder = decoder or self._build_decoder()
    
    def _build_encoder(self):
        """
        Build the encoder network based on VGG19.
        
        Returns:
            nn.Sequential: Encoder network
        """
        # Load pretrained VGG19 and extract first few layers
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # Create encoder with layers up to relu4_1
        encoder = nn.Sequential()
        for i in range(21):  # Up to relu4_1
            encoder.add_module(str(i), vgg[i])
        
        # Freeze encoder parameters
        for param in encoder.parameters():
            param.requires_grad = False
        
        return encoder
    
    def _build_decoder(self):
        """
        Build the decoder network, which is a mirrored version of the encoder.
        
        Returns:
            nn.Sequential: Decoder network
        """
        # Define decoder architecture (mirror of encoder)
        decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )
        
        return decoder
    
    def encode(self, x):
        """
        Encode an image to extract features.
        
        Args:
            x (torch.Tensor): Input image tensor
            
        Returns:
            torch.Tensor: Encoded features
        """
        return self.encoder(x)
    
    def decode(self, x):
        """
        Decode features to generate an image.
        
        Args:
            x (torch.Tensor): Feature tensor
            
        Returns:
            torch.Tensor: Output image tensor
        """
        return self.decoder(x)
    
    def adain(self, content_feat, style_feat):
        """
        Apply Adaptive Instance Normalization.
        
        Args:
            content_feat (torch.Tensor): Content feature tensor
            style_feat (torch.Tensor): Style feature tensor
            
        Returns:
            torch.Tensor: AdaIN normalized feature tensor
        """
        size = content_feat.size()
        
        # Calculate mean and std of content features
        content_mean = content_feat.mean(dim=[2, 3], keepdim=True)
        content_std = content_feat.std(dim=[2, 3], keepdim=True) + 1e-5
        
        # Calculate mean and std of style features
        style_mean = style_feat.mean(dim=[2, 3], keepdim=True)
        style_std = style_feat.std(dim=[2, 3], keepdim=True) + 1e-5
        
        # Normalize content features
        normalized_feat = (content_feat - content_mean) / content_std
        
        # Scale and shift with style statistics
        return normalized_feat * style_std + style_mean
    
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
        # Extract features
        content_feat = self.encode(content)
        style_feat = self.encode(style)
        
        # Apply AdaIN
        if alpha == 1.0:
            t = self.adain(content_feat, style_feat)
        else:
            # Interpolate between content and style
            t = alpha * self.adain(content_feat, style_feat) + (1 - alpha) * content_feat
        
        # Decode to generate styled image
        styled_image = self.decode(t)
        
        return styled_image
    
    def calculate_style_loss(self, target_features, style_features):
        """
        Calculate style loss for training.
        
        Args:
            target_features (torch.Tensor): Target feature tensor
            style_features (torch.Tensor): Style feature tensor
            
        Returns:
            torch.Tensor: Style loss
        """
        # Calculate mean and std
        target_mean = target_features.mean(dim=[2, 3])
        target_std = target_features.std(dim=[2, 3]) + 1e-5
        
        style_mean = style_features.mean(dim=[2, 3])
        style_std = style_features.std(dim=[2, 3]) + 1e-5
        
        # Calculate loss
        mean_loss = F.mse_loss(target_mean, style_mean)
        std_loss = F.mse_loss(target_std, style_std)
        
        return mean_loss + std_loss
    
    def calculate_content_loss(self, target_features, content_features):
        """
        Calculate content loss for training.
        
        Args:
            target_features (torch.Tensor): Target feature tensor
            content_features (torch.Tensor): Content feature tensor
            
        Returns:
            torch.Tensor: Content loss
        """
        return F.mse_loss(target_features, content_features)


def train_adain_model(model, content_dataset, style_dataset, output_path, 
                     epochs=10, batch_size=8, learning_rate=1e-4, 
                     content_weight=1.0, style_weight=10.0, device=None):
    """
    Train an AdaIN model.
    
    Args:
        model (AdaINModel): AdaIN model to train
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
        AdaINModel: Trained model
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
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=learning_rate)
    
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
            
            # Extract features
            content_features = model.encode(content_images)
            style_features = model.encode(style_images)
            
            # Apply AdaIN
            t = model.adain(content_features, style_features)
            
            # Generate styled image
            styled_images = model.decode(t)
            
            # Extract features of styled image
            styled_features = model.encode(styled_images)
            
            # Calculate losses
            content_loss = model.calculate_content_loss(styled_features, t)
            style_loss = model.calculate_style_loss(styled_features, style_features)
            
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
    torch.save(model.decoder.state_dict(), output_path)
    
    return model
