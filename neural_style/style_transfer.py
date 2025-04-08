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
from .models.adain import AdaINModel
from .models.attention import AttentionStyleTransfer
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
            self.model = AdaINModel().to(self.device)
            
            # Load pre-trained decoder if provided
            if model_path and os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.decoder.load_state_dict(state_dict)
                logger.info(f"Loaded pre-trained AdaIN decoder from {model_path}")
            else:
                logger.info("Using default AdaIN model (no pre-trained decoder)")
            
        elif self.method == 'attention':
            # Style-Attentional Networks
            self.model = AttentionStyleTransfer().to(self.device)
            
            # Load pre-trained model if provided
            if model_path and os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded pre-trained attention model from {model_path}")
            else:
                logger.info("Using default attention model (no pre-trained weights)")
    
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
        alpha=1.0,
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
            alpha (float): Style interpolation weight (for AdaIN and Attention methods)
            show_progress (bool): Whether to show progress bar
            
        Returns:
            PIL.Image: Styled image
        """
        # Load and preprocess images
        if isinstance(content_image, str):
            content_image = load_image(content_image, image_size)
        content_tensor = preprocess_image(content_image).to(self.device)
        
        if self.method in ['vgg', 'adain', 'attention']:
            if style_image is None:
                raise ValueError(f"Style image is required for {self.method} method")
                
            if isinstance(style_image, str):
                style_image = load_image(style_image, image_size)
            style_tensor = preprocess_image(style_image).to(self.device)
        
        # Apply style transfer based on method
        if self.method == 'vgg':
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
        elif self.method == 'adain':
            # Apply AdaIN style transfer
            with torch.no_grad():
                output_tensor = self.model(content_tensor, style_tensor, alpha=alpha).cpu()
        elif self.method == 'attention':
            # Apply attention-based style transfer
            with torch.no_grad():
                output_tensor = self.model(content_tensor, style_tensor, alpha=alpha).cpu()
        
        # Convert tensor to image
        output_image = deprocess_image(output_tensor.clone())
        
        # Save output image if path is provided
        if output_path:
            save_image(output_image, output_path)
            logger.info(f"Saved styled image to {output_path}")
        
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
    
    def batch_transfer(
        self,
        content_dir,
        style_image=None,
        output_dir=None,
        image_size=512,
        recursive=False,
        **kwargs
    ):
        """
        Apply style transfer to a batch of content images.
        
        Args:
            content_dir (str): Directory containing content images
            style_image (str or PIL.Image): Style image path or PIL image
            output_dir (str): Directory to save output images
            image_size (int): Size of the output images
            recursive (bool): Whether to search for images recursively
            **kwargs: Additional arguments for style transfer
            
        Returns:
            list: List of output image paths
        """
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get list of image files
        if recursive:
            image_files = []
            for root, _, files in os.walk(content_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(root, file))
        else:
            image_files = [
                os.path.join(content_dir, f) for f in os.listdir(content_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
        
        # Load style image once
        if self.method != 'fast' and style_image is None:
            raise ValueError(f"Style image is required for {self.method} method")
        
        if isinstance(style_image, str) and self.method != 'fast':
            style_image = load_image(style_image, image_size)
        
        # Process each content image
        output_paths = []
        
        for i, content_path in enumerate(tqdm(image_files, desc="Processing images")):
            # Generate output path
            if output_dir:
                filename = os.path.basename(content_path)
                output_path = os.path.join(output_dir, f"styled_{filename}")
            else:
                output_path = None
            
            # Apply style transfer
            try:
                self.transfer(
                    content_image=content_path,
                    style_image=style_image,
                    output_path=output_path,
                    image_size=image_size,
                    **kwargs
                )
                
                if output_path:
                    output_paths.append(output_path)
                    
            except Exception as e:
                logger.error(f"Error processing {content_path}: {str(e)}")
        
        return output_paths
    
    def train_adain_model(
        self,
        content_dataset,
        style_dataset,
        output_model_path,
        epochs=10,
        batch_size=8,
        learning_rate=1e-4,
        content_weight=1.0,
        style_weight=10.0
    ):
        """
        Train an AdaIN model on datasets of content and style images.
        
        Args:
            content_dataset: Dataset of content images
            style_dataset: Dataset of style images
            output_model_path (str): Path to save trained model
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            content_weight (float): Weight of content loss
            style_weight (float): Weight of style loss
            
        Returns:
            None
        """
        from .models.adain import train_adain_model
        
        if self.method != 'adain':
            raise ValueError("Method must be 'adain' to train an AdaIN model")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        
        # Train model
        train_adain_model(
            self.model,
            content_dataset,
            style_dataset,
            output_model_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            content_weight=content_weight,
            style_weight=style_weight,
            device=self.device
        )
        
        logger.info(f"Saved trained AdaIN model to {output_model_path}")
    
    def train_attention_model(
        self,
        content_dataset,
        style_dataset,
        output_model_path,
        epochs=10,
        batch_size=4,
        learning_rate=1e-4,
        content_weight=1.0,
        style_weight=10.0
    ):
        """
        Train an attention-based style transfer model.
        
        Args:
            content_dataset: Dataset of content images
            style_dataset: Dataset of style images
            output_model_path (str): Path to save trained model
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            content_weight (float): Weight of content loss
            style_weight (float): Weight of style loss
            
        Returns:
            None
        """
        from .models.attention import train_attention_model
        
        if self.method != 'attention':
            raise ValueError("Method must be 'attention' to train an attention model")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        
        # Train model
        train_attention_model(
            self.model,
            content_dataset,
            style_dataset,
            output_model_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            content_weight=content_weight,
            style_weight=style_weight,
            device=self.device
        )
        
        logger.info(f"Saved trained attention model to {output_model_path}")
