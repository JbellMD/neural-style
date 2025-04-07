"""
Core style transfer implementation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from tqdm import tqdm
import copy

from .models.vgg import VGG
from .models.transformer_net import TransformerNet
from .utils.image_utils import load_image, save_image, preprocess_image, deprocess_image
from .utils.logging import get_logger

# Setup logger
logger = get_logger(__name__)


class StyleTransfer:
    """
    Neural Style Transfer implementation with multiple methods.
    """
    
    METHODS = ['vgg', 'fast', 'adain', 'attention']
    
    def __init__(
        self, 
        method='vgg', 
        device=None, 
        model_path=None,
        content_layers=None,
        style_layers=None
    ):
        """
        Initialize the style transfer model.
        
        Args:
            method (str): Style transfer method to use ('vgg', 'fast', 'adain', 'attention')
            device (str): Device to use for computation ('cpu', 'cuda', 'mps')
            model_path (str): Path to pre-trained model (for fast transfer methods)
            content_layers (list): Content layer names for VGG method
            style_layers (list): Style layer names for VGG method
        """
        self.method = method.lower()
        if self.method not in self.METHODS:
            raise ValueError(f"Method must be one of {self.METHODS}")
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                      "mps" if torch.backends.mps.is_available() else 
                                      "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Set content and style layers
        self.content_layers = content_layers or ['conv_4']
        self.style_layers = style_layers or ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        # Initialize model based on method
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path):
        """Initialize the appropriate model based on the selected method."""
        if self.method == 'vgg':
            # VGG-based method (Gatys et al.)
            self.model = VGG(self.content_layers, self.style_layers).to(self.device)
            logger.info("Initialized VGG-based style transfer model")
            
        elif self.method == 'fast':
            # Fast neural style transfer (Johnson et al.)
            self.model = TransformerNet().to(self.device)
            
            # Load pre-trained model if provided
            if model_path and os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded pre-trained fast style transfer model from {model_path}")
            else:
                logger.warning("No pre-trained model provided for fast style transfer")
                
        elif self.method == 'adain':
            # Adaptive Instance Normalization (AdaIN)
            # Will be implemented in future versions
            raise NotImplementedError("AdaIN method is not yet implemented")
            
        elif self.method == 'attention':
            # Style-Attentional Networks
            # Will be implemented in future versions
            raise NotImplementedError("Attention-based method is not yet implemented")
    
    def transfer(
        self,
        content_image,
        style_image=None,
        output_path=None,
        image_size=512,
        style_weight=1e6,
        content_weight=1,
        tv_weight=0,
        iterations=300,
        optimizer='lbfgs',
        lr=1.0,
        show_progress=True
    ):
        """
        Apply style transfer to the content image.
        
        Args:
            content_image (str or PIL.Image): Content image path or PIL image
            style_image (str or PIL.Image): Style image path or PIL image
            output_path (str): Path to save the output image
            image_size (int): Size of the output image (assumes square image)
            style_weight (float): Weight of style loss
            content_weight (float): Weight of content loss
            tv_weight (float): Weight of total variation loss
            iterations (int): Number of optimization iterations
            optimizer (str): Optimizer to use ('lbfgs', 'adam')
            lr (float): Learning rate for optimizer
            show_progress (bool): Whether to show progress bar
            
        Returns:
            PIL.Image: Styled image
        """
        # Load and preprocess images
        if isinstance(content_image, str):
            content_image = load_image(content_image, image_size)
        content_tensor = preprocess_image(content_image).to(self.device)
        
        if self.method == 'vgg':
            if style_image is None:
                raise ValueError("Style image is required for VGG method")
                
            if isinstance(style_image, str):
                style_image = load_image(style_image, image_size)
            style_tensor = preprocess_image(style_image).to(self.device)
            
            # Apply VGG-based style transfer
            output_tensor = self._vgg_style_transfer(
                content_tensor, 
                style_tensor,
                style_weight=style_weight,
                content_weight=content_weight,
                tv_weight=tv_weight,
                iterations=iterations,
                optimizer=optimizer,
                lr=lr,
                show_progress=show_progress
            )
            
        elif self.method == 'fast':
            # Apply fast style transfer
            with torch.no_grad():
                output_tensor = self.model(content_tensor).cpu()
        
        # Convert output tensor to PIL image
        output_image = deprocess_image(output_tensor)
        
        # Save output image if path is provided
        if output_path:
            save_image(output_image, output_path)
            logger.info(f"Saved output image to {output_path}")
        
        return output_image
    
    def _vgg_style_transfer(
        self,
        content_tensor,
        style_tensor,
        style_weight=1e6,
        content_weight=1,
        tv_weight=0,
        iterations=300,
        optimizer='lbfgs',
        lr=1.0,
        show_progress=True
    ):
        """
        Apply VGG-based style transfer (Gatys et al.).
        
        Args:
            content_tensor (torch.Tensor): Content image tensor
            style_tensor (torch.Tensor): Style image tensor
            style_weight (float): Weight of style loss
            content_weight (float): Weight of content loss
            tv_weight (float): Weight of total variation loss
            iterations (int): Number of optimization iterations
            optimizer (str): Optimizer to use ('lbfgs', 'adam')
            lr (float): Learning rate for optimizer
            show_progress (bool): Whether to show progress bar
            
        Returns:
            torch.Tensor: Styled image tensor
        """
        # Extract features
        content_features = self.model(content_tensor)
        style_features = self.model(style_tensor)
        
        # Compute gram matrices for style features
        style_grams = {layer: self._gram_matrix(style_features[layer]) for layer in self.style_layers}
        
        # Initialize input image (use content image as starting point)
        input_tensor = content_tensor.clone().requires_grad_(True)
        
        # Setup optimizer
        if optimizer.lower() == 'lbfgs':
            optimizer = optim.LBFGS([input_tensor], lr=lr)
        elif optimizer.lower() == 'adam':
            optimizer = optim.Adam([input_tensor], lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Optimization loop
        progress_bar = tqdm(range(iterations)) if show_progress else range(iterations)
        
        for i in progress_bar:
            def closure():
                optimizer.zero_grad()
                
                # Forward pass
                input_features = self.model(input_tensor)
                
                # Content loss
                content_loss = 0
                for layer in self.content_layers:
                    content_loss += F.mse_loss(input_features[layer], content_features[layer])
                content_loss *= content_weight
                
                # Style loss
                style_loss = 0
                for layer in self.style_layers:
                    input_gram = self._gram_matrix(input_features[layer])
                    style_gram = style_grams[layer]
                    style_loss += F.mse_loss(input_gram, style_gram)
                style_loss *= style_weight / len(self.style_layers)
                
                # Total variation loss (optional)
                tv_loss = 0
                if tv_weight > 0:
                    tv_loss = tv_weight * self._total_variation_loss(input_tensor)
                
                # Total loss
                total_loss = content_loss + style_loss + tv_loss
                
                # Backward pass
                total_loss.backward()
                
                # Update progress bar
                if show_progress:
                    progress_bar.set_description(f"Loss: {total_loss.item():.4f}")
                
                return total_loss
            
            optimizer.step(closure)
        
        # Return final image
        return input_tensor.detach().cpu()
    
    def _gram_matrix(self, tensor):
        """
        Calculate Gram matrix for style representation.
        
        Args:
            tensor (torch.Tensor): Feature tensor
            
        Returns:
            torch.Tensor: Gram matrix
        """
        b, c, h, w = tensor.size()
        features = tensor.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)
    
    def _total_variation_loss(self, image):
        """
        Calculate total variation loss for smoothing.
        
        Args:
            image (torch.Tensor): Image tensor
            
        Returns:
            torch.Tensor: Total variation loss
        """
        h_tv = torch.mean((image[:, :, 1:, :] - image[:, :, :-1, :]).abs())
        w_tv = torch.mean((image[:, :, :, 1:] - image[:, :, :, :-1]).abs())
        return h_tv + w_tv
    
    def train_fast_model(
        self,
        style_image,
        dataset_path,
        output_model_path,
        epochs=2,
        batch_size=4,
        image_size=256,
        style_weight=1e6,
        content_weight=1,
        tv_weight=1e-6,
        lr=1e-3,
        log_interval=100,
        checkpoint_interval=1000
    ):
        """
        Train a fast style transfer model on a dataset of content images.
        
        Args:
            style_image (str): Path to style image
            dataset_path (str): Path to dataset of content images
            output_model_path (str): Path to save trained model
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            image_size (int): Size of training images
            style_weight (float): Weight of style loss
            content_weight (float): Weight of content loss
            tv_weight (float): Weight of total variation loss
            lr (float): Learning rate
            log_interval (int): Interval for logging training progress
            checkpoint_interval (int): Interval for saving model checkpoints
            
        Returns:
            None
        """
        if self.method != 'fast':
            raise ValueError("This method is only available for fast style transfer")
        
        # Will be implemented in a future version
        raise NotImplementedError("Training fast style transfer models is not yet implemented")
    
    def interpolate_styles(
        self,
        content_image,
        style_images,
        weights=None,
        output_path=None,
        image_size=512,
        **kwargs
    ):
        """
        Interpolate between multiple styles.
        
        Args:
            content_image (str or PIL.Image): Content image
            style_images (list): List of style image paths or PIL images
            weights (list): List of weights for each style
            output_path (str): Path to save the output image
            image_size (int): Size of the output image
            **kwargs: Additional arguments for style transfer
            
        Returns:
            PIL.Image: Styled image
        """
        # Will be implemented in a future version
        raise NotImplementedError("Style interpolation is not yet implemented")
