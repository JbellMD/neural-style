# Neural Style Transfer Application

A powerful application that applies artistic styles to images using deep learning techniques.

## Overview

This project implements neural style transfer algorithms to transform ordinary photos into artistic masterpieces by applying the style of famous artworks or custom style images. It provides both a command-line interface and a web-based UI for easy interaction.

## Features

- **Multiple Style Transfer Methods**:
  - VGG-based Neural Style Transfer (Gatys et al.)
  - Fast Neural Style Transfer (Johnson et al.)
  - Adaptive Instance Normalization (AdaIN)
  - Style-Attentional Networks

- **User-Friendly Interfaces**:
  - Web UI with real-time preview
  - Command-line interface for batch processing
  - Python API for integration with other applications

- **Advanced Options**:
  - Style weight adjustment
  - Content preservation control
  - Resolution control
  - Style interpolation between multiple styles

- **Pre-trained Models**:
  - Collection of pre-trained models for popular art styles
  - Support for custom style images

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-style.git
cd neural-style

# Install dependencies
pip install -r requirements.txt

# Optional: Install GPU dependencies
pip install -r requirements-gpu.txt
```

## Quick Start

### Command Line

```bash
# Basic style transfer
python -m neural_style.cli --content path/to/content.jpg --style path/to/style.jpg --output output.jpg

# Advanced options
python -m neural_style.cli --content path/to/content.jpg --style path/to/style.jpg --output output.jpg --style-weight 10.0 --content-weight 1.0 --image-size 512
```

### Web UI

```bash
# Start the web server
python -m neural_style.web

# Then open your browser at http://localhost:8000
```

### Python API

```python
from neural_style import StyleTransfer

# Create a style transfer model
model = StyleTransfer(method='vgg')

# Apply style transfer
output = model.transfer(
    content_image='path/to/content.jpg',
    style_image='path/to/style.jpg',
    style_weight=10.0,
    content_weight=1.0
)

# Save the result
output.save('output.jpg')
```

## Examples

| Content Image | Style Image | Result |
|---------------|-------------|--------|
| <img src="examples/content/landscape.jpg" width="200"> | <img src="examples/styles/starry_night.jpg" width="200"> | <img src="examples/results/landscape_starry_night.jpg" width="200"> |
| <img src="examples/content/portrait.jpg" width="200"> | <img src="examples/styles/picasso.jpg" width="200"> | <img src="examples/results/portrait_picasso.jpg" width="200"> |

## License

MIT
