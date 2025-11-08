"""
Local training script for federated learning clients
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from pathlib import Path

from src.model import FaceRecognitionModel
from src.dataset import create_dataloader
from src.utils import save_model, load_model

logger = logging.getLogger(__name__)


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, 
               criterion: nn.Module, device: str = 'cpu') -> float:
    """
    Train model for one epoch
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        embeddings, logits = model(images)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
            device: str = 'cpu') -> tuple:
    """
    Validate model
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Average loss and accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            embeddings, logits = model(images)
            
            # Calculate loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = 100 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def train_client(model: nn.Module, data_dir: str, member_name: str, 
                config: dict, device: str = 'cpu', num_epochs: int = 5) -> tuple:
    """
    Train model on client data
    
    Args:
        model: Model to train
        data_dir: Directory containing processed data
        member_name: Name of the member
        config: Configuration dictionary
        device: Device to train on
        num_epochs: Number of training epochs
        
    Returns:
        Trained model state dictionary and number of training samples
    """
    # Get training parameters from config
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    batch_size = training_config.get('batch_size', 16)
    learning_rate = training_config.get('learning_rate', 0.001)
    image_size = model_config.get('image_size', 160)
    
    # Create data loaders
    train_loader = create_dataloader(
        data_dir, member_name, batch_size=batch_size,
        image_size=image_size, is_training=True, shuffle=True
    )
    
    # Count number of training samples
    num_samples = len(train_loader.dataset)
    
    if num_samples == 0:
        logger.warning(f"No training data found for {member_name}")
        return model.state_dict(), 0
    
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer and loss function
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=training_config.get('momentum', 0.9),
        weight_decay=training_config.get('weight_decay', 0.0001)
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    logger.info(f"Training {member_name} for {num_epochs} epochs on {num_samples} samples")
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
    
    # Return trained model state and sample count
    return model.state_dict(), num_samples

