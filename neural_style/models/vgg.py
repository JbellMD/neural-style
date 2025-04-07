"""
VGG model for neural style transfer.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGG(nn.Module):
    """
    VGG-19 model for neural style transfer.
    
    This model extracts features from specified layers of the VGG-19 network
    for use in neural style transfer.
    """
    
    def __init__(self, content_layers, style_layers, weights='DEFAULT'):
        """
        Initialize the VGG model.
        
        Args:
            content_layers (list): List of content layer names
            style_layers (list): List of style layer names
            weights (str): Pretrained weights to use
        """
        super(VGG, self).__init__()
        
        # Load pretrained VGG19 model
        if weights == 'DEFAULT':
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        else:
            vgg = models.vgg19(weights=None).features
            vgg.load_state_dict(torch.load(weights))
        
        # Freeze all parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Create model with selected layers
        self.layers = nn.ModuleList()
        self.layer_names = []
        
        # Track all layers we need to extract
        target_layers = sorted(list(set(content_layers + style_layers)))
        
        # Map layer names to indices
        self.layer_map = {
            'conv_1': 1,
            'conv_2': 3,
            'conv_3': 6,
            'conv_4': 8,
            'conv_5': 11,
            'conv_6': 13,
            'conv_7': 15,
            'conv_8': 17,
            'conv_9': 20,
            'conv_10': 22,
            'conv_11': 24,
            'conv_12': 26,
            'conv_13': 29,
            'conv_14': 31,
            'conv_15': 33,
            'conv_16': 35
        }
        
        # Create sequential model with only the layers we need
        current_layer = 0
        block = nn.Sequential()
        
        for i, layer in enumerate(vgg.children()):
            block.add_module(str(i), layer)
            
            # If this is a target layer or we've reached the end of a block
            if i + 1 in [self.layer_map[name] for name in target_layers]:
                # Get the layer name
                for name, idx in self.layer_map.items():
                    if idx == i + 1:
                        layer_name = name
                        break
                
                # Add the block to our model
                self.layers.append(block)
                self.layer_names.append(layer_name)
                
                # Start a new block
                block = nn.Sequential()
        
        # Store which layers are content and style layers
        self.content_layers = content_layers
        self.style_layers = style_layers
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            dict: Dictionary of feature maps for each target layer
        """
        features = {}
        
        # Pass input through each block and store the outputs
        for i, block in enumerate(self.layers):
            x = block(x)
            layer_name = self.layer_names[i]
            
            # Only store features for content and style layers
            if layer_name in self.content_layers or layer_name in self.style_layers:
                features[layer_name] = x
        
        return features
