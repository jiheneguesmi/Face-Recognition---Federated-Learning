"""
Model evaluation script
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

from src.model import initialize_model
from src.dataset import create_dataloader
from src.train import validate
from src.utils import load_config, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model_path: str, config_path: str = "config.yaml"):
    """
    Evaluate the trained model
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to configuration file
    """
    config = load_config(config_path)
    
    # Initialize model
    model = initialize_model(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Load model weights
    load_model(model, model_path, device)
    logger.info(f"Loaded model from {model_path}")
    
    # Get all predictions
    all_predictions = []
    all_labels = []
    member_names = config.get('client', {}).get('member_names', [])
    data_config = config.get('data', {})
    data_dir = data_config.get('processed_data_path', 'data/processed')
    model_config = config.get('model', {})
    image_size = model_config.get('image_size', 160)
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate on each member's data
    for member_name in member_names:
        logger.info(f"Evaluating on {member_name}'s data...")
        
        # Create dataloader
        val_loader = create_dataloader(
            data_dir, member_name, batch_size=16,
            image_size=image_size, is_training=False, shuffle=False
        )
        
        if len(val_loader.dataset) == 0:
            logger.warning(f"No data found for {member_name}")
            continue
        
        # Get predictions
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                _, logits = model(images)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    logger.info(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    logger.info("\nConfusion Matrix:")
    logger.info(cm)
    
    # Classification report
    target_names = member_names  # Use actual member names
    report = classification_report(all_labels, all_predictions, target_names=target_names)
    logger.info("\nClassification Report:")
    logger.info(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(config['paths']['saved_models_dir']) / "confusion_matrix.png"
    plt.savefig(plot_path)
    logger.info(f"Confusion matrix saved to {plot_path}")
    
    return accuracy, cm


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.config)


if __name__ == "__main__":
    main()

