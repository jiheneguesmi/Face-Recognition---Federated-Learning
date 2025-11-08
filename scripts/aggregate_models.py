"""
Script to aggregate multiple model checkpoints
Useful for sequential training approach
"""

import argparse
import torch
from pathlib import Path
import logging

from src.model import initialize_model
from src.aggregation import federated_averaging
from src.utils import load_config, save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def aggregate_checkpoints(model_paths: list, sample_counts: list, 
                         output_path: str, config_path: str = "config.yaml"):
    """
    Aggregate multiple model checkpoints
    
    Args:
        model_paths: List of paths to model checkpoints
        sample_counts: List of sample counts for each model
        output_path: Path to save aggregated model
        config_path: Path to configuration file
    """
    config = load_config(config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Validate inputs
    if len(model_paths) != len(sample_counts):
        raise ValueError("Number of model paths must match number of sample counts")
    
    # Load all models
    model_states = []
    for i, model_path in enumerate(model_paths):
        if not Path(model_path).exists():
            logger.warning(f"Model {model_path} not found, skipping...")
            continue
        
        logger.info(f"Loading model {i+1}/{len(model_paths)}: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model_state = checkpoint.get('model_state_dict', checkpoint)
        model_states.append(model_state)
    
    if len(model_states) == 0:
        raise ValueError("No valid models found to aggregate")
    
    # Adjust sample counts if some models were skipped
    sample_counts = sample_counts[:len(model_states)]
    
    logger.info(f"Aggregating {len(model_states)} models...")
    logger.info(f"Sample counts: {sample_counts}")
    
    # Aggregate using FedAvg
    aggregated_state = federated_averaging(model_states, sample_counts)
    
    # Create and save aggregated model
    aggregated_model = initialize_model(config)
    aggregated_model.load_state_dict(aggregated_state)
    
    save_model(aggregated_model, output_path, {
        'round': 'aggregated',
        'method': 'fedavg',
        'num_models': len(model_states),
        'sample_counts': sample_counts
    })
    
    logger.info(f"Aggregated model saved to {output_path}")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Aggregate model checkpoints')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='Paths to model checkpoints')
    parser.add_argument('--sample-counts', type=int, nargs='+', required=True,
                       help='Sample counts for each model')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for aggregated model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file path')
    
    args = parser.parse_args()
    
    aggregate_checkpoints(
        args.models,
        args.sample_counts,
        args.output,
        args.config
    )


if __name__ == "__main__":
    main()

