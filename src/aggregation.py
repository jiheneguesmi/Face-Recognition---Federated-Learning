"""
Federated aggregation functions
"""

import torch
import torch.nn as nn
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def federated_averaging(models: List[Dict[str, torch.Tensor]], 
                       sample_counts: List[int]) -> Dict[str, torch.Tensor]:
    """
    Perform Federated Averaging (FedAvg)
    
    Args:
        models: List of model state dictionaries from clients
        sample_counts: List of number of samples per client
        
    Returns:
        Averaged model state dictionary
    """
    if len(models) == 0:
        raise ValueError("No models to aggregate")
    
    if len(models) != len(sample_counts):
        raise ValueError("Number of models and sample counts must match")
    
    # Calculate total number of samples
    total_samples = sum(sample_counts)
    
    if total_samples == 0:
        raise ValueError("Total sample count is zero")
    
    # Initialize aggregated model with first model's structure
    aggregated_model = {}
    
    # Get all parameter names from first model
    param_names = models[0].keys()
    
    # Weighted average of parameters
    for param_name in param_names:
        # Initialize aggregated parameter
        aggregated_param = None
        
        for model, n_samples in zip(models, sample_counts):
            param = model[param_name]
            
            # Calculate weight for this client
            weight = n_samples / total_samples
            
            # Weighted sum
            weighted_param = param * weight
            
            if aggregated_param is None:
                aggregated_param = weighted_param.clone()
            else:
                aggregated_param += weighted_param
        
        aggregated_model[param_name] = aggregated_param
    
    logger.info(f"Aggregated {len(models)} models with sample counts: {sample_counts}")
    
    return aggregated_model


def aggregate_models(model_dicts: List[Dict], sample_counts: List[int]) -> Dict[str, torch.Tensor]:
    """
    Aggregate multiple models using FedAvg
    
    Args:
        model_dicts: List of model state dictionaries
        sample_counts: List of sample counts for each model
        
    Returns:
        Aggregated model state dictionary
    """
    return federated_averaging(model_dicts, sample_counts)


def update_global_model(global_model: nn.Module, client_models: List[Dict[str, torch.Tensor]], 
                       sample_counts: List[int]):
    """
    Update global model with aggregated client models
    
    Args:
        global_model: Global model to update
        client_models: List of client model state dictionaries
        sample_counts: List of sample counts for each client
    """
    # Aggregate client models
    aggregated_state = federated_averaging(client_models, sample_counts)
    
    # Update global model
    global_model.load_state_dict(aggregated_state)
    
    logger.info("Global model updated with aggregated client models")

