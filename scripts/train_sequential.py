"""
Sequential training script (alternative to centralized server)
Each member trains on the model and passes it to the next member
"""

import argparse
import torch
from pathlib import Path
import logging

from src.model import initialize_model
from src.train import train_client
from src.utils import load_config, save_model, load_model
from src.aggregation import federated_averaging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_sequential(member_name: str, round_num: int, input_model_path: str = None, 
                    output_model_path: str = None, config_path: str = "config.yaml"):
    """
    Train model sequentially for one member
    
    Args:
        member_name: Name of the member
        round_num: Current round number
        input_model_path: Path to input model (None for first member)
        output_model_path: Path to save output model
        config_path: Path to configuration file
    """
    config = load_config(config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize or load model
    model = initialize_model(config)
    model = model.to(device)
    
    if input_model_path and Path(input_model_path).exists():
        load_model(model, input_model_path, device)
        logger.info(f"Loaded model from {input_model_path}")
    else:
        logger.info("Starting with fresh model")
    
    # Train model
    training_config = config.get('training', {})
    num_epochs = training_config.get('epochs_per_client', 5)
    data_config = config.get('data', {})
    data_dir = data_config.get('processed_data_path', 'data/processed')
    
    model_state, sample_count = train_client(
        model=model,
        data_dir=data_dir,
        member_name=member_name,
        config=config,
        device=device,
        num_epochs=num_epochs
    )
    
    # Save model
    if output_model_path:
        save_model(model, output_model_path, {
            'member': member_name,
            'round': round_num,
            'samples': sample_count
        })
        logger.info(f"Model saved to {output_model_path}")
    
    return model_state, sample_count


def aggregate_sequential_models(model_paths: list, sample_counts: list, 
                               output_path: str, config_path: str = "config.yaml"):
    """
    Aggregate multiple sequentially trained models
    
    Args:
        model_paths: List of model checkpoint paths
        sample_counts: List of sample counts
        output_path: Path to save aggregated model
        config_path: Path to configuration file
    """
    config = load_config(config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load all models
    model_states = []
    for model_path in model_paths:
        model = initialize_model(config)
        load_model(model, model_path, device)
        model_states.append(model.state_dict())
    
    # Aggregate using FedAvg
    aggregated_state = federated_averaging(model_states, sample_counts)
    
    # Save aggregated model
    aggregated_model = initialize_model(config)
    aggregated_model.load_state_dict(aggregated_state)
    save_model(aggregated_model, output_path, {
        'round': 'aggregated',
        'method': 'fedavg'
    })
    
    logger.info(f"Aggregated model saved to {output_path}")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Sequential training')
    parser.add_argument('--member', type=str, required=True, help='Member name')
    parser.add_argument('--round', type=int, required=True, help='Round number')
    parser.add_argument('--input-model', type=str, default=None, help='Input model path')
    parser.add_argument('--output-model', type=str, required=True, help='Output model path')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    train_sequential(
        args.member,
        args.round,
        args.input_model,
        args.output_model,
        args.config
    )


if __name__ == "__main__":
    main()

