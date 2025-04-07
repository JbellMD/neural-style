#!/usr/bin/env python
"""
Script to download pre-trained models for neural style transfer.
"""

import os
import sys
import argparse
import requests
import zipfile
import io
from tqdm import tqdm

# Add parent directory to path to import neural_style
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_style.utils.logging import get_logger

# Setup logger
logger = get_logger(__name__)

# Model URLs
MODEL_URLS = {
    'fast': {
        'starry_night': 'https://github.com/pytorch/examples/raw/main/fast_neural_style/saved_models/starry_night.pth',
        'mosaic': 'https://github.com/pytorch/examples/raw/main/fast_neural_style/saved_models/mosaic.pth',
        'candy': 'https://github.com/pytorch/examples/raw/main/fast_neural_style/saved_models/candy.pth',
        'rain_princess': 'https://github.com/pytorch/examples/raw/main/fast_neural_style/saved_models/rain_princess.pth',
        'udnie': 'https://github.com/pytorch/examples/raw/main/fast_neural_style/saved_models/udnie.pth',
    }
}

# Example image URLs
EXAMPLE_URLS = {
    'content': {
        'landscape': 'https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/content-images/amber.jpg',
        'portrait': 'https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/content-images/dancing.jpg',
        'building': 'https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/content-images/hoovertowernight.jpg',
    },
    'styles': {
        'starry_night': 'https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/starry_night.jpg',
        'mosaic': 'https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/mosaic.jpg',
        'candy': 'https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/candy.jpg',
        'rain_princess': 'https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/rain_princess.jpg',
        'udnie': 'https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/udnie.jpg',
    }
}


def download_file(url, destination):
    """
    Download a file from URL to destination with progress bar.
    
    Args:
        url (str): URL to download
        destination (str): Destination path
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Show progress bar
        progress_bar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=os.path.basename(destination)
        )
        
        # Write file
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False


def download_models(model_type, output_dir, models=None):
    """
    Download pre-trained models.
    
    Args:
        model_type (str): Model type (fast, adain, attention)
        output_dir (str): Output directory
        models (list): List of model names to download (None for all)
        
    Returns:
        int: Number of models downloaded
    """
    if model_type not in MODEL_URLS:
        logger.error(f"Unknown model type: {model_type}")
        return 0
    
    # Get model URLs
    model_urls = MODEL_URLS[model_type]
    
    # Filter models if specified
    if models:
        model_urls = {name: url for name, url in model_urls.items() if name in models}
    
    if not model_urls:
        logger.error(f"No models found for type: {model_type}")
        return 0
    
    # Create output directory
    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # Download models
    count = 0
    for name, url in model_urls.items():
        logger.info(f"Downloading {name} model...")
        destination = os.path.join(model_dir, f"{name}.pth")
        
        if os.path.exists(destination):
            logger.info(f"Model already exists: {destination}")
            count += 1
            continue
        
        if download_file(url, destination):
            logger.info(f"Downloaded {name} model to {destination}")
            count += 1
    
    return count


def download_examples(output_dir):
    """
    Download example images.
    
    Args:
        output_dir (str): Output directory
        
    Returns:
        int: Number of images downloaded
    """
    count = 0
    
    # Download content images
    content_dir = os.path.join(output_dir, 'content')
    os.makedirs(content_dir, exist_ok=True)
    
    for name, url in EXAMPLE_URLS['content'].items():
        logger.info(f"Downloading content image: {name}...")
        destination = os.path.join(content_dir, f"{name}.jpg")
        
        if os.path.exists(destination):
            logger.info(f"Image already exists: {destination}")
            count += 1
            continue
        
        if download_file(url, destination):
            logger.info(f"Downloaded content image to {destination}")
            count += 1
    
    # Download style images
    styles_dir = os.path.join(output_dir, 'styles')
    os.makedirs(styles_dir, exist_ok=True)
    
    for name, url in EXAMPLE_URLS['styles'].items():
        logger.info(f"Downloading style image: {name}...")
        destination = os.path.join(styles_dir, f"{name}.jpg")
        
        if os.path.exists(destination):
            logger.info(f"Image already exists: {destination}")
            count += 1
            continue
        
        if download_file(url, destination):
            logger.info(f"Downloaded style image to {destination}")
            count += 1
    
    return count


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Download pre-trained models for neural style transfer')
    parser.add_argument('--model-type', type=str, default='fast', choices=MODEL_URLS.keys(),
                        help='Model type to download')
    parser.add_argument('--models', type=str, nargs='+',
                        help='Specific models to download (default: all)')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for models')
    parser.add_argument('--examples', action='store_true',
                        help='Download example images')
    parser.add_argument('--examples-dir', type=str, default='examples/images',
                        help='Output directory for example images')
    
    args = parser.parse_args()
    
    # Download models
    logger.info(f"Downloading {args.model_type} models...")
    count = download_models(args.model_type, args.output_dir, args.models)
    logger.info(f"Downloaded {count} models")
    
    # Download examples if requested
    if args.examples:
        logger.info("Downloading example images...")
        count = download_examples(args.examples_dir)
        logger.info(f"Downloaded {count} example images")
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
