"""
Face recognition model based on FaceNet
"""

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
import logging

logger = logging.getLogger(__name__)


class FaceRecognitionModel(nn.Module):
    """
    Face recognition model for federated learning
    Uses pretrained FaceNet (InceptionResnetV1) as backbone
    """
    
    def __init__(self, num_classes: int = 4, embedding_size: int = 512, pretrained: bool = True):
        """
        Initialize face recognition model
        
        Args:
            num_classes: Number of classes (team members)
            embedding_size: Size of face embedding (512 for FaceNet)
            pretrained: Whether to use pretrained weights
        """
        super(FaceRecognitionModel, self).__init__()
        
        # Load pretrained FaceNet backbone
        self.backbone = InceptionResnetV1(pretrained='vggface2' if pretrained else None)
        
        # Freeze backbone parameters (optional - can fine-tune)
        # Uncomment to freeze:
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        # Classifier head for face recognition
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        logger.info(f"Initialized FaceRecognitionModel with {num_classes} classes")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images (batch_size, 3, 160, 160)
            
        Returns:
            embeddings: Face embeddings (batch_size, embedding_size)
            logits: Classification logits (batch_size, num_classes)
        """
        # Get face embeddings from backbone
        embeddings = self.backbone(x)
        
        # Get classification logits
        logits = self.classifier(embeddings)
        
        return embeddings, logits
    
    def get_embeddings(self, x):
        """
        Get face embeddings only (without classification)
        
        Args:
            x: Input images
            
        Returns:
            embeddings: Face embeddings
        """
        return self.backbone(x)


def create_model(num_classes: int = 4, embedding_size: int = 512, pretrained: bool = True) -> FaceRecognitionModel:
    """
    Create a face recognition model
    
    Args:
        num_classes: Number of classes
        embedding_size: Embedding dimension
        pretrained: Use pretrained weights
        
    Returns:
        Face recognition model
    """
    model = FaceRecognitionModel(
        num_classes=num_classes,
        embedding_size=embedding_size,
        pretrained=pretrained
    )
    return model


def initialize_model(config: dict) -> FaceRecognitionModel:
    """
    Initialize model from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model_config = config.get('model', {})
    model = create_model(
        num_classes=model_config.get('num_classes', 4),
        embedding_size=model_config.get('embedding_size', 512),
        pretrained=model_config.get('pretrained', True)
    )
    return model

