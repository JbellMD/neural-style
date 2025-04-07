"""
Basic usage examples for neural style transfer.
"""

import os
import sys
import time
from PIL import Image

# Add parent directory to path to import neural_style
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_style import StyleTransfer
from neural_style.utils.image_utils import load_image, save_image


def basic_style_transfer():
    """Basic style transfer example."""
    print("Running basic style transfer example...")
    
    # Initialize style transfer model
    model = StyleTransfer(method='vgg')
    
    # Load content and style images
    content_image = load_image('examples/images/content/landscape.jpg', size=512)
    style_image = load_image('examples/images/styles/starry_night.jpg', size=512)
    
    # Apply style transfer
    start_time = time.time()
    result = model.transfer(
        content_image=content_image,
        style_image=style_image,
        style_weight=1e6,
        content_weight=1.0,
        iterations=300,
        show_progress=True
    )
    elapsed_time = time.time() - start_time
    
    # Save result
    os.makedirs('examples/output', exist_ok=True)
    save_image(result, 'examples/output/basic_result.jpg')
    
    print(f"Style transfer completed in {elapsed_time:.2f} seconds")
    print(f"Result saved to examples/output/basic_result.jpg")
    
    # Display result
    result.show()


def fast_style_transfer():
    """Fast style transfer example using a pre-trained model."""
    print("Running fast style transfer example...")
    
    # Check if model exists
    model_path = 'models/fast/starry_night.pth'
    if not os.path.exists(model_path):
        print(f"Pre-trained model not found: {model_path}")
        print("Please train a model first or download a pre-trained model")
        return
    
    # Initialize style transfer model
    model = StyleTransfer(method='fast', model_path=model_path)
    
    # Load content image
    content_image = load_image('examples/images/content/portrait.jpg', size=512)
    
    # Apply style transfer
    start_time = time.time()
    result = model.transfer(content_image=content_image)
    elapsed_time = time.time() - start_time
    
    # Save result
    os.makedirs('examples/output', exist_ok=True)
    save_image(result, 'examples/output/fast_result.jpg')
    
    print(f"Fast style transfer completed in {elapsed_time:.2f} seconds")
    print(f"Result saved to examples/output/fast_result.jpg")
    
    # Display result
    result.show()


def batch_style_transfer():
    """Batch style transfer example."""
    print("Running batch style transfer example...")
    
    # Initialize style transfer model
    model = StyleTransfer(method='vgg')
    
    # Load style image
    style_image = load_image('examples/images/styles/wave.jpg', size=512)
    
    # Get content images
    content_dir = 'examples/images/content'
    content_files = [os.path.join(content_dir, f) for f in os.listdir(content_dir) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not content_files:
        print(f"No content images found in {content_dir}")
        return
    
    # Create output directory
    output_dir = 'examples/output/batch'
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply style transfer to each content image
    start_time = time.time()
    for i, content_file in enumerate(content_files):
        print(f"Processing {i+1}/{len(content_files)}: {content_file}")
        
        # Load content image
        content_image = load_image(content_file, size=512)
        
        # Apply style transfer
        result = model.transfer(
            content_image=content_image,
            style_image=style_image,
            style_weight=1e6,
            content_weight=1.0,
            iterations=100,  # Fewer iterations for batch processing
            show_progress=False
        )
        
        # Save result
        output_file = os.path.join(output_dir, os.path.basename(content_file))
        save_image(result, output_file)
    
    elapsed_time = time.time() - start_time
    print(f"Batch style transfer completed in {elapsed_time:.2f} seconds")
    print(f"Results saved to {output_dir}")


def style_weight_comparison():
    """Compare different style weights."""
    print("Running style weight comparison example...")
    
    # Initialize style transfer model
    model = StyleTransfer(method='vgg')
    
    # Load content and style images
    content_image = load_image('examples/images/content/landscape.jpg', size=512)
    style_image = load_image('examples/images/styles/starry_night.jpg', size=512)
    
    # Create output directory
    output_dir = 'examples/output/style_weights'
    os.makedirs(output_dir, exist_ok=True)
    
    # Try different style weights
    style_weights = [1e3, 1e4, 1e5, 1e6, 1e7]
    
    for style_weight in style_weights:
        print(f"Processing style weight: {style_weight}")
        
        # Apply style transfer
        result = model.transfer(
            content_image=content_image,
            style_image=style_image,
            style_weight=style_weight,
            content_weight=1.0,
            iterations=100,  # Fewer iterations for comparison
            show_progress=False
        )
        
        # Save result
        output_file = os.path.join(output_dir, f"weight_{style_weight:.0e}.jpg")
        save_image(result, output_file)
    
    print(f"Style weight comparison completed")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    # Create example directories if they don't exist
    os.makedirs('examples/images/content', exist_ok=True)
    os.makedirs('examples/images/styles', exist_ok=True)
    
    # Check if example images exist
    if not os.path.exists('examples/images/content/landscape.jpg'):
        print("Example images not found. Please add some images to examples/images/content and examples/images/styles")
        sys.exit(1)
    
    # Run examples
    basic_style_transfer()
    # Uncomment to run other examples
    # fast_style_transfer()
    # batch_style_transfer()
    # style_weight_comparison()
