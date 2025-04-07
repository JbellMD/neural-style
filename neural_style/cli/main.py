"""
Command-line interface for neural style transfer.
"""

import os
import sys
import time
import click
import torch
from PIL import Image
from pathlib import Path

from ..style_transfer import StyleTransfer
from ..utils.image_utils import load_image, save_image
from ..utils.logging import get_logger

# Setup logger
logger = get_logger(__name__)


@click.group()
def cli():
    """Neural Style Transfer command-line interface."""
    pass


@cli.command()
@click.option('--content', required=True, type=click.Path(exists=True), help='Content image path')
@click.option('--style', required=True, type=click.Path(exists=True), help='Style image path')
@click.option('--output', required=True, type=click.Path(), help='Output image path')
@click.option('--method', type=click.Choice(['vgg', 'fast', 'adain', 'attention']), default='vgg', 
              help='Style transfer method')
@click.option('--model-path', type=click.Path(exists=True), help='Path to pre-trained model (for fast method)')
@click.option('--image-size', type=int, default=512, help='Output image size')
@click.option('--style-weight', type=float, default=1e6, help='Weight of style loss')
@click.option('--content-weight', type=float, default=1.0, help='Weight of content loss')
@click.option('--tv-weight', type=float, default=0.0, help='Weight of total variation loss')
@click.option('--iterations', type=int, default=300, help='Number of optimization iterations')
@click.option('--optimizer', type=click.Choice(['lbfgs', 'adam']), default='lbfgs', help='Optimizer to use')
@click.option('--lr', type=float, default=1.0, help='Learning rate')
@click.option('--device', type=str, help='Device to use (cpu, cuda, mps)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def transfer(content, style, output, method, model_path, image_size, style_weight, content_weight, 
             tv_weight, iterations, optimizer, lr, device, quiet):
    """Apply style transfer to a content image."""
    start_time = time.time()
    
    # Initialize style transfer model
    try:
        style_transfer = StyleTransfer(method=method, device=device, model_path=model_path)
        
        # Apply style transfer
        result = style_transfer.transfer(
            content_image=content,
            style_image=style,
            output_path=output,
            image_size=image_size,
            style_weight=style_weight,
            content_weight=content_weight,
            tv_weight=tv_weight,
            iterations=iterations,
            optimizer=optimizer,
            lr=lr,
            show_progress=not quiet
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Style transfer completed in {elapsed_time:.2f} seconds")
        logger.info(f"Output saved to {output}")
        
    except Exception as e:
        logger.error(f"Error during style transfer: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--content-dir', required=True, type=click.Path(exists=True), help='Content images directory')
@click.option('--style', required=True, type=click.Path(exists=True), help='Style image path')
@click.option('--output-dir', required=True, type=click.Path(), help='Output directory')
@click.option('--method', type=click.Choice(['vgg', 'fast', 'adain', 'attention']), default='vgg', 
              help='Style transfer method')
@click.option('--model-path', type=click.Path(exists=True), help='Path to pre-trained model (for fast method)')
@click.option('--image-size', type=int, default=512, help='Output image size')
@click.option('--style-weight', type=float, default=1e6, help='Weight of style loss')
@click.option('--content-weight', type=float, default=1.0, help='Weight of content loss')
@click.option('--device', type=str, help='Device to use (cpu, cuda, mps)')
@click.option('--recursive', is_flag=True, help='Recursively process subdirectories')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def batch_transfer(content_dir, style, output_dir, method, model_path, image_size, style_weight, 
                   content_weight, device, recursive, quiet):
    """Apply style transfer to multiple content images."""
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get content image paths
    content_path = Path(content_dir)
    if recursive:
        content_files = list(content_path.glob('**/*.jpg')) + list(content_path.glob('**/*.jpeg')) + \
                       list(content_path.glob('**/*.png'))
    else:
        content_files = list(content_path.glob('*.jpg')) + list(content_path.glob('*.jpeg')) + \
                       list(content_path.glob('*.png'))
    
    if not content_files:
        logger.error(f"No image files found in {content_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(content_files)} content images")
    
    # Initialize style transfer model
    try:
        style_transfer = StyleTransfer(method=method, device=device, model_path=model_path)
        
        # Load style image once
        style_img = load_image(style, image_size)
        
        # Process each content image
        for i, content_file in enumerate(content_files):
            if not quiet:
                logger.info(f"Processing {i+1}/{len(content_files)}: {content_file}")
            
            # Determine output path
            rel_path = content_file.relative_to(content_path) if content_file.is_relative_to(content_path) else content_file.name
            output_path = Path(output_dir) / rel_path
            os.makedirs(output_path.parent, exist_ok=True)
            
            # Apply style transfer
            style_transfer.transfer(
                content_image=str(content_file),
                style_image=style_img,
                output_path=str(output_path),
                image_size=image_size,
                style_weight=style_weight,
                content_weight=content_weight,
                show_progress=False
            )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Batch style transfer completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {len(content_files)} images")
        
    except Exception as e:
        logger.error(f"Error during batch style transfer: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--style', required=True, type=click.Path(exists=True), help='Style image path')
@click.option('--dataset', required=True, type=click.Path(exists=True), help='Training dataset directory')
@click.option('--output-model', required=True, type=click.Path(), help='Output model path')
@click.option('--epochs', type=int, default=2, help='Number of training epochs')
@click.option('--batch-size', type=int, default=4, help='Batch size')
@click.option('--image-size', type=int, default=256, help='Training image size')
@click.option('--style-weight', type=float, default=1e6, help='Weight of style loss')
@click.option('--content-weight', type=float, default=1.0, help='Weight of content loss')
@click.option('--tv-weight', type=float, default=1e-6, help='Weight of total variation loss')
@click.option('--lr', type=float, default=1e-3, help='Learning rate')
@click.option('--device', type=str, help='Device to use (cpu, cuda, mps)')
def train(style, dataset, output_model, epochs, batch_size, image_size, style_weight, 
          content_weight, tv_weight, lr, device):
    """Train a fast style transfer model."""
    try:
        # Initialize style transfer model
        style_transfer = StyleTransfer(method='fast', device=device)
        
        # Train model
        style_transfer.train_fast_model(
            style_image=style,
            dataset_path=dataset,
            output_model_path=output_model,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            style_weight=style_weight,
            content_weight=content_weight,
            tv_weight=tv_weight,
            lr=lr
        )
        
        logger.info(f"Model training completed")
        logger.info(f"Model saved to {output_model}")
        
    except NotImplementedError:
        logger.error("Training fast style transfer models is not yet implemented")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        sys.exit(1)


@cli.command()
def list_devices():
    """List available devices for computation."""
    devices = []
    
    # Check CPU
    devices.append("cpu")
    
    # Check CUDA
    if torch.cuda.is_available():
        devices.append("cuda")
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append("mps")
    
    # Print available devices
    click.echo("Available devices:")
    for device in devices:
        click.echo(f"- {device}")


if __name__ == '__main__':
    cli()
