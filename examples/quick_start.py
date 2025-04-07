"""
Quick start example for neural style transfer.

This script demonstrates how to use the neural style transfer application
with minimal setup. It downloads example images and a pre-trained model,
then applies style transfer to create a stylized image.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import neural_style
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_style import StyleTransfer
from neural_style.utils.image_utils import load_image, save_image
from scripts.download_models import download_file, EXAMPLE_URLS, MODEL_URLS


def ensure_example_images():
    """
    Ensure example images are available.
    
    Returns:
        tuple: Paths to content and style images
    """
    # Create directories
    examples_dir = Path('examples/images')
    content_dir = examples_dir / 'content'
    styles_dir = examples_dir / 'styles'
    
    os.makedirs(content_dir, exist_ok=True)
    os.makedirs(styles_dir, exist_ok=True)
    
    # Download content image if needed
    content_name = 'landscape'
    content_path = content_dir / f"{content_name}.jpg"
    
    if not content_path.exists():
        print(f"Downloading content image: {content_name}...")
        download_file(EXAMPLE_URLS['content'][content_name], str(content_path))
    
    # Download style image if needed
    style_name = 'starry_night'
    style_path = styles_dir / f"{style_name}.jpg"
    
    if not style_path.exists():
        print(f"Downloading style image: {style_name}...")
        download_file(EXAMPLE_URLS['styles'][style_name], str(style_path))
    
    return str(content_path), str(style_path)


def ensure_model():
    """
    Ensure a pre-trained model is available.
    
    Returns:
        str: Path to model
    """
    # Create directory
    models_dir = Path('models/fast')
    os.makedirs(models_dir, exist_ok=True)
    
    # Download model if needed
    model_name = 'starry_night'
    model_path = models_dir / f"{model_name}.pth"
    
    if not model_path.exists():
        print(f"Downloading pre-trained model: {model_name}...")
        download_file(MODEL_URLS['fast'][model_name], str(model_path))
    
    return str(model_path)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Quick start example for neural style transfer')
    parser.add_argument('--method', type=str, default='vgg', choices=['vgg', 'fast'],
                        help='Style transfer method to use')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Size of the output image')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations for VGG method')
    parser.add_argument('--output', type=str, default='examples/output/quick_start_result.jpg',
                        help='Output image path')
    
    args = parser.parse_args()
    
    # Ensure example images are available
    content_path, style_path = ensure_example_images()
    print(f"Using content image: {content_path}")
    print(f"Using style image: {style_path}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize style transfer model
    model_path = None
    if args.method == 'fast':
        model_path = ensure_model()
        print(f"Using pre-trained model: {model_path}")
    
    model = StyleTransfer(method=args.method, model_path=model_path)
    
    # Apply style transfer
    print(f"Applying style transfer using {args.method} method...")
    result = model.transfer(
        content_image=content_path,
        style_image=style_path,
        output_path=args.output,
        image_size=args.image_size,
        iterations=args.iterations,
        show_progress=True
    )
    
    print(f"Style transfer complete!")
    print(f"Result saved to: {args.output}")
    
    # Display result
    try:
        result.show()
    except Exception:
        print("Unable to display result image. Please open it manually.")


if __name__ == '__main__':
    main()
