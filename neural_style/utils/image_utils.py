"""
Image utilities for neural style transfer.
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, to_pil_image

# Mean and std for ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def load_image(path, size=None, scale=None):
    """
    Load an image from path and optionally resize it.
    
    Args:
        path (str): Path to the image
        size (int or tuple): Size to resize the image to
        scale (float): Scale factor for resizing
        
    Returns:
        PIL.Image: Loaded image
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    img = Image.open(path).convert('RGB')
    
    if size is not None:
        if isinstance(size, int):
            # Preserve aspect ratio
            w, h = img.size
            if w < h:
                new_w = size
                new_h = int(h * size / w)
            else:
                new_h = size
                new_w = int(w * size / h)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            # Resize to specific dimensions
            img = img.resize(size, Image.LANCZOS)
    elif scale is not None:
        w, h = img.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    return img


def save_image(img, path):
    """
    Save an image to path.
    
    Args:
        img (PIL.Image or torch.Tensor): Image to save
        path (str): Path to save the image to
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Convert tensor to PIL image if necessary
    if isinstance(img, torch.Tensor):
        img = to_pil_image(img.squeeze(0))
    
    # Save image
    img.save(path)


def preprocess_image(img, normalize=True):
    """
    Preprocess an image for neural style transfer.
    
    Args:
        img (PIL.Image): Image to preprocess
        normalize (bool): Whether to normalize the image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Convert PIL image to tensor
    if isinstance(img, Image.Image):
        tensor = to_tensor(img)
    else:
        tensor = img
    
    # Add batch dimension if needed
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    # Normalize
    if normalize:
        tensor = normalize_tensor(tensor)
    
    return tensor


def deprocess_image(tensor, denormalize=True):
    """
    Deprocess an image tensor to a PIL image.
    
    Args:
        tensor (torch.Tensor): Image tensor to deprocess
        denormalize (bool): Whether to denormalize the image
        
    Returns:
        PIL.Image: Deprocessed image
    """
    # Clone tensor to avoid modifying the original
    tensor = tensor.clone().detach()
    
    # Denormalize
    if denormalize:
        tensor = denormalize_tensor(tensor)
    
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL image
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    return to_pil_image(tensor)


def normalize_tensor(tensor):
    """
    Normalize a tensor using ImageNet mean and std.
    
    Args:
        tensor (torch.Tensor): Tensor to normalize
        
    Returns:
        torch.Tensor: Normalized tensor
    """
    device = tensor.device
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    
    if tensor.dim() == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    
    return (tensor - mean) / std


def denormalize_tensor(tensor):
    """
    Denormalize a tensor using ImageNet mean and std.
    
    Args:
        tensor (torch.Tensor): Tensor to denormalize
        
    Returns:
        torch.Tensor: Denormalized tensor
    """
    device = tensor.device
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    
    if tensor.dim() == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    
    return tensor * std + mean


def resize_image(img, size):
    """
    Resize an image while preserving aspect ratio.
    
    Args:
        img (PIL.Image): Image to resize
        size (int): Target size (for the smaller dimension)
        
    Returns:
        PIL.Image: Resized image
    """
    w, h = img.size
    if w < h:
        new_w = size
        new_h = int(h * size / w)
    else:
        new_h = size
        new_w = int(w * size / h)
    
    return img.resize((new_w, new_h), Image.LANCZOS)


def crop_to_size(img, size):
    """
    Crop an image to a square of the specified size.
    
    Args:
        img (PIL.Image): Image to crop
        size (int): Target size
        
    Returns:
        PIL.Image: Cropped image
    """
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    right = left + size
    bottom = top + size
    
    return img.crop((left, top, right, bottom))


def create_image_grid(images, rows, cols):
    """
    Create a grid of images.
    
    Args:
        images (list): List of PIL images
        rows (int): Number of rows
        cols (int): Number of columns
        
    Returns:
        PIL.Image: Grid image
    """
    w, h = images[0].size
    grid = Image.new('RGB', (cols * w, rows * h))
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid.paste(img, (col * w, row * h))
    
    return grid
