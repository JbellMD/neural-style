"""
Unit tests for image utility functions.
"""

import os
import unittest
import tempfile
import torch
import numpy as np
from PIL import Image

from neural_style.utils.image_utils import (
    load_image, save_image, preprocess_image, deprocess_image,
    normalize_tensor, denormalize_tensor, resize_image, crop_to_size,
    create_image_grid
)


class TestImageUtils(unittest.TestCase):
    """Test cases for image utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test images
        self.test_image = Image.new('RGB', (100, 100), color='blue')
        self.test_image_path = os.path.join(tempfile.gettempdir(), 'test_image.jpg')
        self.test_image.save(self.test_image_path)
        
        # Create test tensor
        self.test_tensor = torch.ones(1, 3, 100, 100) * 0.5  # Mid-gray
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove test image
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def test_load_image(self):
        """Test load_image function."""
        # Test loading with default parameters
        img = load_image(self.test_image_path)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (100, 100))
        
        # Test loading with size parameter
        img = load_image(self.test_image_path, size=50)
        self.assertEqual(img.size, (50, 50))
        
        # Test loading with scale parameter
        img = load_image(self.test_image_path, scale=0.5)
        self.assertEqual(img.size, (50, 50))
        
        # Test loading non-existent image
        with self.assertRaises(FileNotFoundError):
            load_image('non_existent_image.jpg')
    
    def test_save_image(self):
        """Test save_image function."""
        # Test saving PIL image
        output_path = os.path.join(tempfile.gettempdir(), 'output_image.jpg')
        save_image(self.test_image, output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Test saving tensor
        tensor_output_path = os.path.join(tempfile.gettempdir(), 'tensor_output.jpg')
        save_image(self.test_tensor, tensor_output_path)
        self.assertTrue(os.path.exists(tensor_output_path))
        
        # Clean up
        os.remove(output_path)
        os.remove(tensor_output_path)
    
    def test_preprocess_image(self):
        """Test preprocess_image function."""
        # Test preprocessing PIL image
        tensor = preprocess_image(self.test_image)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 3, 100, 100))
        
        # Test preprocessing without normalization
        tensor = preprocess_image(self.test_image, normalize=False)
        self.assertTrue(torch.all(tensor >= 0))
        self.assertTrue(torch.all(tensor <= 1))
        
        # Test preprocessing tensor
        tensor2 = preprocess_image(tensor)
        self.assertEqual(tensor2.shape, tensor.shape)
    
    def test_deprocess_image(self):
        """Test deprocess_image function."""
        # Test deprocessing tensor
        img = deprocess_image(self.test_tensor)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (100, 100))
        
        # Test deprocessing without denormalization
        img = deprocess_image(self.test_tensor, denormalize=False)
        self.assertIsInstance(img, Image.Image)
        
        # Test round-trip conversion
        tensor = preprocess_image(self.test_image)
        img = deprocess_image(tensor)
        # Images should be similar (not exact due to compression and conversion)
        self.assertEqual(img.size, self.test_image.size)
    
    def test_normalize_denormalize(self):
        """Test normalize_tensor and denormalize_tensor functions."""
        # Create test tensor with values in [0, 1]
        test_tensor = torch.ones(1, 3, 10, 10) * 0.5
        
        # Normalize
        normalized = normalize_tensor(test_tensor)
        
        # Denormalize
        denormalized = denormalize_tensor(normalized)
        
        # Check round-trip conversion
        self.assertTrue(torch.allclose(test_tensor, denormalized, atol=1e-5))
    
    def test_resize_image(self):
        """Test resize_image function."""
        # Create test image
        test_img = Image.new('RGB', (200, 100), color='blue')
        
        # Resize to smaller size
        resized = resize_image(test_img, 50)
        
        # Check that aspect ratio is preserved
        self.assertEqual(resized.size, (50, 25))
        
        # Create square test image
        square_img = Image.new('RGB', (100, 100), color='blue')
        
        # Resize square image
        resized = resize_image(square_img, 50)
        self.assertEqual(resized.size, (50, 50))
    
    def test_crop_to_size(self):
        """Test crop_to_size function."""
        # Create test image
        test_img = Image.new('RGB', (200, 100), color='blue')
        
        # Crop to square
        cropped = crop_to_size(test_img, 50)
        
        # Check that result is square with requested size
        self.assertEqual(cropped.size, (50, 50))
        
        # Crop to larger size than image
        with self.assertRaises(Exception):
            crop_to_size(test_img, 300)
    
    def test_create_image_grid(self):
        """Test create_image_grid function."""
        # Create test images
        images = [
            Image.new('RGB', (50, 50), color='red'),
            Image.new('RGB', (50, 50), color='green'),
            Image.new('RGB', (50, 50), color='blue'),
            Image.new('RGB', (50, 50), color='yellow')
        ]
        
        # Create 2x2 grid
        grid = create_image_grid(images, rows=2, cols=2)
        
        # Check grid size
        self.assertEqual(grid.size, (100, 100))
        
        # Create 1x4 grid
        grid = create_image_grid(images, rows=1, cols=4)
        self.assertEqual(grid.size, (200, 50))


if __name__ == '__main__':
    unittest.main()
