"""
Utility functions for the federated learning project
"""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(config: Dict[str, Any]):
    """
    Create necessary directories for the project
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})
    data_config = config.get('data', {})
    
    # Create model directories
    os.makedirs(paths.get('models_dir', 'models'), exist_ok=True)
    os.makedirs(paths.get('checkpoints_dir', 'models/checkpoints'), exist_ok=True)
    os.makedirs(paths.get('saved_models_dir', 'models/saved'), exist_ok=True)
    os.makedirs(paths.get('pretrained_dir', 'models/pretrained'), exist_ok=True)
    
    # Create data directories
    os.makedirs(data_config.get('raw_data_path', 'data/raw'), exist_ok=True)
    os.makedirs(data_config.get('processed_data_path', 'data/processed'), exist_ok=True)
    
    # Create log directories
    os.makedirs(paths.get('logs_dir', 'logs'), exist_ok=True)
    os.makedirs(paths.get('tensorboard_dir', 'logs/tensorboard'), exist_ok=True)
    
    # Create member directories
    for member in config.get('client', {}).get('member_names', []):
        os.makedirs(f"{data_config.get('raw_data_path', 'data/raw')}/{member}", exist_ok=True)
        os.makedirs(f"{data_config.get('processed_data_path', 'data/processed')}/{member}", exist_ok=True)
    
    logger.info("Directories created successfully")


def save_model(model: torch.nn.Module, path: str, metadata: Dict[str, Any] = None):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        path: Path to save model
        metadata: Additional metadata to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)
    logger.info(f"Model saved to {path}")


def load_model(model: torch.nn.Module, path: str, device: str = 'cpu'):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model to load weights into
        path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded metadata
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    metadata = checkpoint.get('metadata', {})
    logger.info(f"Model loaded from {path}")
    return metadata


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

