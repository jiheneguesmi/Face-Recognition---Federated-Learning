"""
Dataset class for face recognition
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FaceDataset(Dataset):
    """
    Dataset class for face images
    """
    
    def __init__(self, data_dir: str, member_name: str, transform=None, mtcnn=None, member_label=None):
        """
        Initialize face dataset
        
        Args:
            data_dir: Directory containing processed face images
            member_name: Name of the member (e.g., 'omarmej', 'abir', 'omarbr', 'jihene')
            transform: Image transformations
            mtcnn: MTCNN face detector (optional, for on-the-fly detection)
            member_label: Label index for this member (0, 1, 2, or 3). If None, will be determined from config.
        """
        self.data_dir = Path(data_dir) / member_name
        self.member_name = member_name
        self.transform = transform
        self.mtcnn = mtcnn
        
        # Set member label
        if member_label is not None:
            self.member_label = member_label
        else:
            # Map member name to label using config
            try:
                from src.utils import load_config
                config = load_config()
                member_names = config.get('client', {}).get('member_names', [])
                if member_name in member_names:
                    self.member_label = member_names.index(member_name)
                else:
                    logger.warning(f"Member {member_name} not found in config, using label 0")
                    self.member_label = 0
            except Exception as e:
                logger.warning(f"Could not load config for member label mapping: {e}, using label 0")
                self.member_label = 0
        
        # Get all image files
        self.image_paths = []
        if self.data_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                self.image_paths.extend(list(self.data_dir.glob(ext)))
        
        if len(self.image_paths) == 0:
            logger.warning(f"No images found in {self.data_dir}")
        else:
            logger.info(f"Loaded {len(self.image_paths)} images for {member_name} (label: {self.member_label})")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get item from dataset
        
        Args:
            idx: Index
            
        Returns:
            image: Preprocessed image tensor
            label: Member label (0, 1, 2, or 3)
        """
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (160, 160), (0, 0, 0))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Use the member label
        label = torch.tensor(self.member_label, dtype=torch.long)
        
        return image, label


def get_transforms(image_size: int = 160, is_training: bool = True):
    """
    Get image transformations
    
    Args:
        image_size: Target image size
        is_training: Whether this is for training (apply augmentation)
        
    Returns:
        Transform composition
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    return transform


def create_dataloader(data_dir: str, member_name: str, batch_size: int = 16, 
                     image_size: int = 160, is_training: bool = True, 
                     shuffle: bool = True, member_label: int = None) -> DataLoader:
    """
    Create DataLoader for a member's data
    
    Args:
        data_dir: Directory containing processed images
        member_name: Name of the member (e.g., 'omarmej', 'abir', 'omarbr', 'jihene')
        batch_size: Batch size
        image_size: Image size
        is_training: Whether for training
        shuffle: Whether to shuffle data
        member_label: Label index for this member (optional, will be determined from config if not provided)
        
    Returns:
        DataLoader
    """
    transform = get_transforms(image_size, is_training)
    dataset = FaceDataset(data_dir, member_name, transform=transform, member_label=member_label)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader

