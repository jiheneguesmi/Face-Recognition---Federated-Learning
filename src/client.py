"""
Client code for federated learning
"""

import socket
import pickle
import torch
import argparse
import logging
from pathlib import Path

from src.model import initialize_model
from src.train import train_client
from src.utils import load_config

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Client for federated learning
    """
    
    def __init__(self, member_name: str, config_path: str = "config.yaml"):
        """
        Initialize federated client
        
        Args:
            member_name: Name of the member (e.g., 'member1')
            config_path: Path to configuration file
        """
        self.member_name = member_name
        self.config = load_config(config_path)
        
        # Server configuration
        server_config = self.config.get('server', {})
        self.server_host = server_config.get('host', 'localhost')
        self.server_port = server_config.get('port', 5000)
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Client {member_name} initialized, using device: {self.device}")
    
    def connect_to_server(self):
        """
        Connect to the federated learning server
        
        Returns:
            Socket connection to server
        """
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.server_host, self.server_port))
            logger.info(f"Connected to server at {self.server_host}:{self.server_port}")
            return client_socket
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            raise
    
    def get_global_model(self, client_socket):
        """
        Get global model from server
        
        Args:
            client_socket: Socket connection to server
            
        Returns:
            Global model
        """
        # Request model from server
        message = {'type': 'get_model'}
        client_socket.send(pickle.dumps(message))
        
        # Receive model from server
        data = client_socket.recv(4096 * 10)
        response = pickle.loads(data)
        
        if response.get('type') == 'model':
            # Initialize model
            model = initialize_model(self.config)
            model.load_state_dict(response['model_state'])
            model = model.to(self.device)
            logger.info("Received global model from server")
            return model
        else:
            raise ValueError("Invalid response from server")
    
    def send_model_update(self, client_socket, model_state: dict, sample_count: int):
        """
        Send model update to server
        
        Args:
            client_socket: Socket connection to server
            model_state: Trained model state dictionary
            sample_count: Number of training samples
        """
        message = {
            'type': 'send_update',
            'member_name': self.member_name,
            'model_state': model_state,
            'sample_count': sample_count
        }
        
        client_socket.send(pickle.dumps(message))
        
        # Wait for acknowledgment
        data = client_socket.recv(1024)
        response = pickle.loads(data)
        
        if response.get('type') == 'ack':
            logger.info("Model update sent successfully")
        else:
            logger.warning("No acknowledgment from server")
    
    def train_and_update(self, round_num: int):
        """
        Train model locally and send update to server
        
        Args:
            round_num: Current round number
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"CLIENT {self.member_name} - ROUND {round_num}")
        logger.info(f"{'='*50}\n")
        
        # Connect to server
        client_socket = self.connect_to_server()
        
        try:
            # Get global model from server
            model = self.get_global_model(client_socket)
            
            # Train model locally
            training_config = self.config.get('training', {})
            num_epochs = training_config.get('epochs_per_client', 5)
            data_config = self.config.get('data', {})
            data_dir = data_config.get('processed_data_path', 'data/processed')
            
            model_state, sample_count = train_client(
                model=model,
                data_dir=data_dir,
                member_name=self.member_name,
                config=self.config,
                device=self.device,
                num_epochs=num_epochs
            )
            
            # Send model update to server
            self.send_model_update(client_socket, model_state, sample_count)
            
            logger.info(f"Round {round_num} completed for {self.member_name}")
        
        except Exception as e:
            logger.error(f"Error in training round: {e}")
        finally:
            client_socket.close()
    
    def run(self, num_rounds: int = None):
        """
        Run federated learning client
        
        Args:
            num_rounds: Number of rounds (if None, use config value)
        """
        if num_rounds is None:
            training_config = self.config.get('training', {})
            num_rounds = training_config.get('num_rounds', 20)
        
        logger.info(f"Starting federated learning for {self.member_name} ({num_rounds} rounds)")
        
        for round_num in range(1, num_rounds + 1):
            try:
                self.train_and_update(round_num)
                # Wait a bit before next round
                import time
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error in round {round_num}: {e}")
                break
        
        logger.info(f"Federated learning completed for {self.member_name}")


def main():
    """
    Main function to run the client
    """
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--member', type=str, required=True, help='Member name (e.g., member1)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--server-address', type=str, default=None, help='Server address (host:port)')
    parser.add_argument('--rounds', type=int, default=None, help='Number of rounds')
    
    args = parser.parse_args()
    
    client = FederatedClient(args.member, args.config)
    
    if args.server_address:
        host, port = args.server_address.split(':')
        client.server_host = host
        client.server_port = int(port)
    
    client.run(args.rounds)


if __name__ == "__main__":
    main()

