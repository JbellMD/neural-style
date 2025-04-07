"""
Unit tests for the StyleTransfer class.
"""

import os
import unittest
import torch
import numpy as np
from PIL import Image

from neural_style.style_transfer import StyleTransfer
from neural_style.models.vgg import VGG
from neural_style.models.transformer_net import TransformerNet
from neural_style.utils.image_utils import load_image, preprocess_image, deprocess_image


class TestStyleTransfer(unittest.TestCase):
    """Test cases for StyleTransfer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small test image
        self.test_image_size = 64
        self.content_image = Image.new('RGB', (self.test_image_size, self.test_image_size), color='blue')
        self.style_image = Image.new('RGB', (self.test_image_size, self.test_image_size), color='red')
        
        # Preprocess images
        self.content_tensor = preprocess_image(self.content_image)
        self.style_tensor = preprocess_image(self.style_image)
        
        # Set device
        self.device = torch.device('cpu')
    
    def test_init(self):
        """Test StyleTransfer initialization."""
        # Test with default parameters
        model = StyleTransfer()
        self.assertEqual(model.method, 'vgg')
        self.assertEqual(model.device, self.device)
        
        # Test with custom parameters
        model = StyleTransfer(method='vgg', device='cpu', 
                             content_layers=['conv_1'], style_layers=['conv_2'])
        self.assertEqual(model.method, 'vgg')
        self.assertEqual(model.content_layers, ['conv_1'])
        self.assertEqual(model.style_layers, ['conv_2'])
        
        # Test with invalid method
        with self.assertRaises(ValueError):
            StyleTransfer(method='invalid')
    
    def test_vgg_model(self):
        """Test VGG model initialization."""
        # Initialize VGG model
        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        vgg_model = VGG(content_layers, style_layers)
        
        # Test forward pass
        with torch.no_grad():
            features = vgg_model(self.content_tensor)
        
        # Check that features contain the expected layers
        self.assertSetEqual(set(features.keys()), set(content_layers + style_layers))
        
        # Check feature shapes
        for layer in content_layers + style_layers:
            self.assertEqual(len(features[layer].shape), 4)  # [batch, channels, height, width]
    
    def test_transformer_net(self):
        """Test TransformerNet model initialization."""
        # Initialize TransformerNet model
        transformer = TransformerNet()
        
        # Test forward pass
        with torch.no_grad():
            output = transformer(self.content_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, self.content_tensor.shape)
        
        # Check output values are in range [0, 1]
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
    
    def test_gram_matrix(self):
        """Test gram matrix calculation."""
        # Initialize StyleTransfer model
        model = StyleTransfer()
        
        # Create a test tensor
        test_tensor = torch.randn(1, 3, 4, 4)
        
        # Calculate gram matrix
        gram = model._gram_matrix(test_tensor)
        
        # Check shape
        self.assertEqual(gram.shape, (1, 3, 3))
        
        # Check that gram matrix is symmetric
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(gram[0, i, j].item(), gram[0, j, i].item(), places=5)
    
    def test_total_variation_loss(self):
        """Test total variation loss calculation."""
        # Initialize StyleTransfer model
        model = StyleTransfer()
        
        # Create a test tensor with constant values
        constant_tensor = torch.ones(1, 3, 4, 4)
        
        # Calculate total variation loss
        tv_loss = model._total_variation_loss(constant_tensor)
        
        # For a constant tensor, TV loss should be 0
        self.assertAlmostEqual(tv_loss.item(), 0, places=5)
        
        # Create a test tensor with high variation
        varying_tensor = torch.zeros(1, 3, 4, 4)
        varying_tensor[:, :, 0::2, 0::2] = 1  # Checkerboard pattern
        
        # Calculate total variation loss
        tv_loss = model._total_variation_loss(varying_tensor)
        
        # For a checkerboard pattern, TV loss should be high
        self.assertGreater(tv_loss.item(), 0.5)
    
    def test_transfer_vgg(self):
        """Test VGG-based style transfer."""
        # Initialize StyleTransfer model
        model = StyleTransfer(method='vgg')
        
        # Apply style transfer with minimal iterations for testing
        result = model.transfer(
            content_image=self.content_image,
            style_image=self.style_image,
            iterations=2,  # Use minimal iterations for testing
            show_progress=False
        )
        
        # Check that result is a PIL Image
        self.assertIsInstance(result, Image.Image)
        
        # Check result size
        self.assertEqual(result.size, (self.test_image_size, self.test_image_size))
    
    def test_transfer_fast(self):
        """Test fast style transfer."""
        # Skip test if no model is available
        try:
            # Initialize StyleTransfer model
            model = StyleTransfer(method='fast')
            
            # Apply style transfer
            result = model.transfer(
                content_image=self.content_image,
                show_progress=False
            )
            
            # Check that result is a PIL Image
            self.assertIsInstance(result, Image.Image)
            
            # Check result size
            self.assertEqual(result.size, (self.test_image_size, self.test_image_size))
        except NotImplementedError:
            self.skipTest("Fast style transfer model not implemented or available")


if __name__ == '__main__':
    unittest.main()
