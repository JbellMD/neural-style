#!/usr/bin/env python
"""
Benchmark script for neural style transfer methods.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path to import neural_style
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_style.style_transfer import StyleTransfer
from neural_style.utils.image_utils import load_image, save_image, create_image_grid
from neural_style.utils.logging import get_logger

# Setup logger
logger = get_logger(__name__)


def benchmark_methods(content_image, style_image, output_dir, image_size=512, iterations=300, device=None):
    """
    Benchmark different style transfer methods.
    
    Args:
        content_image (str): Path to content image
        style_image (str): Path to style image
        output_dir (str): Output directory
        image_size (int): Image size
        iterations (int): Number of iterations for optimization-based methods
        device (str): Device to use
        
    Returns:
        dict: Benchmark results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    content_img = load_image(content_image, size=image_size)
    style_img = load_image(style_image, size=image_size)
    
    # Save input images
    content_img.save(os.path.join(output_dir, 'content.jpg'))
    style_img.save(os.path.join(output_dir, 'style.jpg'))
    
    # Methods to benchmark
    methods = ['vgg']
    
    # Check if fast method models are available
    fast_model_path = os.path.join('models', 'fast', 'starry_night.pth')
    if os.path.exists(fast_model_path):
        methods.append('fast')
    
    # Results
    results = {}
    result_images = [content_img, style_img]
    
    # Benchmark each method
    for method in methods:
        logger.info(f"Benchmarking {method} method...")
        
        # Initialize model
        model_path = fast_model_path if method == 'fast' else None
        model = StyleTransfer(method=method, device=device, model_path=model_path)
        
        # Apply style transfer
        start_time = time.time()
        
        if method == 'vgg':
            # Benchmark different iteration counts
            iteration_counts = [50, 100, 200, 300]
            for iters in iteration_counts:
                iter_start_time = time.time()
                result = model.transfer(
                    content_image=content_img,
                    style_image=style_img,
                    iterations=iters,
                    show_progress=True
                )
                iter_elapsed = time.time() - iter_start_time
                
                # Save result
                result.save(os.path.join(output_dir, f'{method}_{iters}.jpg'))
                
                # Record result
                results[f'{method}_{iters}'] = {
                    'method': method,
                    'iterations': iters,
                    'time': iter_elapsed,
                    'time_per_iteration': iter_elapsed / iters
                }
                
                # Add to result images
                if iters == iteration_counts[-1]:
                    result_images.append(result)
        else:
            # Benchmark fast method
            result = model.transfer(
                content_image=content_img,
                style_image=style_img
            )
            elapsed_time = time.time() - start_time
            
            # Save result
            result.save(os.path.join(output_dir, f'{method}.jpg'))
            
            # Record result
            results[method] = {
                'method': method,
                'iterations': 1,
                'time': elapsed_time,
                'time_per_iteration': elapsed_time
            }
            
            # Add to result images
            result_images.append(result)
    
    # Create comparison grid
    grid = create_image_grid(result_images, rows=1, cols=len(result_images))
    grid.save(os.path.join(output_dir, 'comparison.jpg'))
    
    # Print results
    logger.info("Benchmark results:")
    logger.info(f"{'Method':<15} {'Iterations':<10} {'Time (s)':<10} {'Time/Iter (s)':<15}")
    logger.info("-" * 50)
    
    for name, result in results.items():
        logger.info(f"{result['method']:<15} {result['iterations']:<10} {result['time']:.2f}s{'':<10} {result['time_per_iteration']:.4f}s")
    
    return results


def benchmark_style_weights(content_image, style_image, output_dir, image_size=512, iterations=100, device=None):
    """
    Benchmark different style weights.
    
    Args:
        content_image (str): Path to content image
        style_image (str): Path to style image
        output_dir (str): Output directory
        image_size (int): Image size
        iterations (int): Number of iterations
        device (str): Device to use
        
    Returns:
        dict: Benchmark results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    content_img = load_image(content_image, size=image_size)
    style_img = load_image(style_image, size=image_size)
    
    # Save input images
    content_img.save(os.path.join(output_dir, 'content.jpg'))
    style_img.save(os.path.join(output_dir, 'style.jpg'))
    
    # Initialize model
    model = StyleTransfer(method='vgg', device=device)
    
    # Style weights to benchmark
    style_weights = [1e3, 1e4, 1e5, 1e6, 1e7]
    
    # Results
    results = {}
    result_images = [content_img, style_img]
    
    # Benchmark each style weight
    for weight in style_weights:
        logger.info(f"Benchmarking style weight: {weight}")
        
        # Apply style transfer
        start_time = time.time()
        result = model.transfer(
            content_image=content_img,
            style_image=style_img,
            style_weight=weight,
            iterations=iterations,
            show_progress=True
        )
        elapsed_time = time.time() - start_time
        
        # Save result
        result.save(os.path.join(output_dir, f'weight_{weight:.0e}.jpg'))
        
        # Record result
        results[f'weight_{weight:.0e}'] = {
            'style_weight': weight,
            'time': elapsed_time
        }
        
        # Add to result images
        result_images.append(result)
    
    # Create comparison grid
    grid = create_image_grid(result_images, rows=1, cols=len(result_images))
    grid.save(os.path.join(output_dir, 'comparison.jpg'))
    
    # Print results
    logger.info("Style weight benchmark results:")
    logger.info(f"{'Style Weight':<15} {'Time (s)':<10}")
    logger.info("-" * 30)
    
    for name, result in results.items():
        logger.info(f"{result['style_weight']:<15.0e} {result['time']:.2f}s")
    
    return results


def benchmark_image_sizes(content_image, style_image, output_dir, iterations=100, device=None):
    """
    Benchmark different image sizes.
    
    Args:
        content_image (str): Path to content image
        style_image (str): Path to style image
        output_dir (str): Output directory
        iterations (int): Number of iterations
        device (str): Device to use
        
    Returns:
        dict: Benchmark results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Image sizes to benchmark
    image_sizes = [128, 256, 384, 512, 640, 768]
    
    # Initialize model
    model = StyleTransfer(method='vgg', device=device)
    
    # Results
    results = {}
    result_images = []
    
    # Benchmark each image size
    for size in image_sizes:
        logger.info(f"Benchmarking image size: {size}")
        
        # Load images at current size
        content_img = load_image(content_image, size=size)
        style_img = load_image(style_image, size=size)
        
        # Apply style transfer
        start_time = time.time()
        result = model.transfer(
            content_image=content_img,
            style_image=style_img,
            iterations=iterations,
            show_progress=True
        )
        elapsed_time = time.time() - start_time
        
        # Save result
        result.save(os.path.join(output_dir, f'size_{size}.jpg'))
        
        # Record result
        results[f'size_{size}'] = {
            'image_size': size,
            'time': elapsed_time,
            'pixels': size * size
        }
        
        # Resize result for comparison grid
        result_resized = result.resize((256, 256), Image.LANCZOS)
        result_images.append(result_resized)
    
    # Create comparison grid
    grid = create_image_grid(result_images, rows=2, cols=3)
    grid.save(os.path.join(output_dir, 'comparison.jpg'))
    
    # Print results
    logger.info("Image size benchmark results:")
    logger.info(f"{'Image Size':<15} {'Pixels':<15} {'Time (s)':<10} {'Time/MPixel (s)':<15}")
    logger.info("-" * 60)
    
    for name, result in results.items():
        mpixels = result['pixels'] / 1e6
        time_per_mpixel = result['time'] / mpixels
        logger.info(f"{result['image_size']:<15} {result['pixels']:<15} {result['time']:.2f}s{'':<10} {time_per_mpixel:.2f}s")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Benchmark neural style transfer methods')
    parser.add_argument('--content', type=str, required=True, help='Content image path')
    parser.add_argument('--style', type=str, required=True, help='Style image path')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Output directory')
    parser.add_argument('--benchmark-type', type=str, default='methods',
                        choices=['methods', 'style_weights', 'image_sizes'],
                        help='Type of benchmark to run')
    parser.add_argument('--image-size', type=int, default=512, help='Image size')
    parser.add_argument('--iterations', type=int, default=300, help='Number of iterations')
    parser.add_argument('--device', type=str, help='Device to use (cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.content):
        logger.error(f"Content image not found: {args.content}")
        return
    
    if not os.path.exists(args.style):
        logger.error(f"Style image not found: {args.style}")
        return
    
    # Run benchmark
    if args.benchmark_type == 'methods':
        benchmark_methods(
            args.content,
            args.style,
            args.output_dir,
            args.image_size,
            args.iterations,
            args.device
        )
    elif args.benchmark_type == 'style_weights':
        benchmark_style_weights(
            args.content,
            args.style,
            args.output_dir,
            args.image_size,
            args.iterations,
            args.device
        )
    elif args.benchmark_type == 'image_sizes':
        benchmark_image_sizes(
            args.content,
            args.style,
            args.output_dir,
            args.iterations,
            args.device
        )


if __name__ == '__main__':
    main()
