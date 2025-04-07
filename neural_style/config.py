"""
Configuration module for neural style transfer application.
"""

import os
import yaml
from pathlib import Path


class Config:
    """Configuration class for neural style transfer application."""
    
    # Default configuration
    DEFAULT_CONFIG = {
        # Model settings
        'models': {
            'vgg': {
                'content_layers': ['conv_4'],
                'style_layers': ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],
                'content_weight': 1.0,
                'style_weight': 1e6,
                'tv_weight': 0.0,
                'optimizer': 'lbfgs',
                'learning_rate': 1.0,
                'iterations': 300
            },
            'fast': {
                'model_dir': 'models/fast'
            },
            'adain': {
                'model_dir': 'models/adain'
            },
            'attention': {
                'model_dir': 'models/attention'
            }
        },
        
        # Web UI settings
        'web': {
            'host': '127.0.0.1',
            'port': 8000,
            'debug': False,
            'upload_folder': 'uploads',
            'results_folder': 'results',
            'styles_folder': 'styles',
            'max_content_length': 16 * 1024 * 1024  # 16 MB
        },
        
        # Image settings
        'image': {
            'default_size': 512,
            'max_size': 1024,
            'quality': 95
        },
        
        # Paths
        'paths': {
            'models_dir': 'models',
            'examples_dir': 'examples',
            'cache_dir': '.cache'
        }
    }
    
    def __init__(self, config_path=None):
        """
        Initialize configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        # Start with default configuration
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """
        Load configuration from file.
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Update configuration with user settings
        self._update_config(self.config, user_config)
    
    def _update_config(self, config, updates):
        """
        Recursively update configuration dictionary.
        
        Args:
            config (dict): Configuration dictionary to update
            updates (dict): Updates to apply
        """
        for key, value in updates.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                self._update_config(config[key], value)
            else:
                config[key] = value
    
    def save_config(self, config_path):
        """
        Save configuration to file.
        
        Args:
            config_path (str): Path to save configuration file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Save configuration
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, *keys, default=None):
        """
        Get configuration value.
        
        Args:
            *keys: Key path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, value, *keys):
        """
        Set configuration value.
        
        Args:
            value: Value to set
            *keys: Key path to configuration value
        """
        if not keys:
            return
        
        config = self.config
        for key in keys[:-1]:
            if key not in config or not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def get_model_config(self, method):
        """
        Get model configuration.
        
        Args:
            method (str): Model method
            
        Returns:
            dict: Model configuration
        """
        return self.get('models', method, default={})
    
    def get_web_config(self):
        """
        Get web UI configuration.
        
        Returns:
            dict: Web UI configuration
        """
        return self.get('web', default={})
    
    def get_image_config(self):
        """
        Get image configuration.
        
        Returns:
            dict: Image configuration
        """
        return self.get('image', default={})
    
    def get_path(self, path_name):
        """
        Get path from configuration.
        
        Args:
            path_name (str): Path name
            
        Returns:
            str: Path
        """
        return self.get('paths', path_name, default='')


# Global configuration instance
config = Config()
