"""
Central server for federated learning
"""

import socket
import pickle
import threading
import time
import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, List

from src.model import initialize_model
from src.aggregation import update_global_model, federated_averaging
from src.utils import load_config, create_directories, save_model

logger = logging.getLogger(__name__)


class FederatedServer:
    """
    Central server for coordinating federated learning
    """
    
    def __init__(self, config_path: str = "config.yaml", host: str = None, port: int = None):
        """
        Initialize federated server
        
        Args:
            config_path: Path to configuration file
            host: Server host address (default: from config, or '0.0.0.0' for network access)
            port: Server port (default: from config)
        """
        self.config = load_config(config_path)
        create_directories(self.config)
        
        # Initialize global model
        self.global_model = initialize_model(self.config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.global_model = self.global_model.to(self.device)
        
        # Server configuration
        server_config = self.config.get('server', {})
        self.host = host if host is not None else server_config.get('host', '0.0.0.0')
        self.port = port if port is not None else server_config.get('port', 5000)
        
        # Training configuration
        training_config = self.config.get('training', {})
        self.num_rounds = training_config.get('num_rounds', 20)
        
        # Client management
        self.client_models = {}
        self.client_sample_counts = {}
        self.connected_clients = set()
        
        logger.info(f"Federated server initialized on {self.host}:{self.port}")
    
    def handle_client(self, client_socket, address):
        """
        Handle client connection
        
        Args:
            client_socket: Client socket
            address: Client address
        """
        try:
            logger.info(f"Client connected from {address}")
            
            while True:
                # Receive message from client
                data = client_socket.recv(4096 * 10)  # Increased buffer size
                if not data:
                    break
                
                message = pickle.loads(data)
                msg_type = message.get('type')
                
                if msg_type == 'get_model':
                    # Send global model to client
                    logger.info(f"Sending model to client {address}")
                    model_state = self.global_model.state_dict()
                    response = {
                        'type': 'model',
                        'model_state': model_state,
                        'config': self.config
                    }
                    client_socket.send(pickle.dumps(response))
                
                elif msg_type == 'send_update':
                    # Receive model update from client
                    member_name = message.get('member_name')
                    model_state = message.get('model_state')
                    sample_count = message.get('sample_count')
                    
                    logger.info(f"Received update from {member_name} ({sample_count} samples)")
                    
                    self.client_models[member_name] = model_state
                    self.client_sample_counts[member_name] = sample_count
                    self.connected_clients.add(member_name)
                    
                    # Send acknowledgment
                    response = {'type': 'ack', 'status': 'received'}
                    client_socket.send(pickle.dumps(response))
                
                elif msg_type == 'disconnect':
                    logger.info(f"Client {address} disconnected")
                    break
        
        except Exception as e:
            logger.error(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
    
    def aggregate_models(self):
        """
        Aggregate models from all clients
        """
        if len(self.client_models) == 0:
            logger.warning("No client models to aggregate")
            return
        
        # Get models and sample counts
        models = list(self.client_models.values())
        sample_counts = list(self.client_sample_counts.values())
        
        # Aggregate using FedAvg
        aggregated_state = federated_averaging(models, sample_counts)
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state)
        
        logger.info(f"Aggregated models from {len(models)} clients")
        
        # Clear client models for next round
        self.client_models.clear()
        self.client_sample_counts.clear()
    
    def run_round(self, round_num: int):
        """
        Run one round of federated learning
        
        Args:
            round_num: Current round number
        """
        logger.info(f"Starting round {round_num}/{self.num_rounds}")
        
        # Wait for all clients to connect and train
        # In a real implementation, you'd have a more sophisticated synchronization
        time.sleep(30)  # Wait for clients to train (adjust as needed)
        
        # Aggregate models
        self.aggregate_models()
        
        # Save checkpoint
        if self.config.get('logging', {}).get('save_checkpoints', True):
            checkpoint_path = Path(self.config['paths']['checkpoints_dir']) / f"round_{round_num}.pth"
            save_model(self.global_model, str(checkpoint_path), {'round': round_num})
    
    def start(self):
        """
        Start the federated learning server
        """
        # Create socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        
        logger.info(f"Server listening on {self.host}:{self.port}")
        
        # Accept client connections in separate threads
        client_threads = []
        
        try:
            # Run federated learning rounds
            for round_num in range(1, self.num_rounds + 1):
                logger.info(f"\n{'='*50}")
                logger.info(f"FEDERATED LEARNING ROUND {round_num}/{self.num_rounds}")
                logger.info(f"{'='*50}\n")
                
                # Reset connected clients for this round
                self.connected_clients.clear()
                
                # Accept client connections
                server_socket.settimeout(60)  # Timeout for accepting connections
                start_time = time.time()
                
                while len(self.connected_clients) < 4 and (time.time() - start_time) < 60:
                    try:
                        client_socket, address = server_socket.accept()
                        client_thread = threading.Thread(
                            target=self.handle_client,
                            args=(client_socket, address)
                        )
                        client_thread.start()
                        client_threads.append(client_thread)
                    except socket.timeout:
                        logger.warning("Timeout waiting for clients")
                        break
                
                # Run the round
                self.run_round(round_num)
                
                # Wait a bit before next round
                time.sleep(5)
            
            # Save final model
            final_model_path = Path(self.config['paths']['saved_models_dir']) / "final_model.pth"
            save_model(self.global_model, str(final_model_path), {'rounds': self.num_rounds})
            logger.info(f"Final model saved to {final_model_path}")
        
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        finally:
            server_socket.close()
            for thread in client_threads:
                thread.join()


def main():
    """
    Main function to run the server
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--rounds', type=int, default=None, help='Number of rounds (overrides config)')
    parser.add_argument('--host', type=str, default=None, help='Server host (default: 0.0.0.0 for network access)')
    parser.add_argument('--port', type=int, default=None, help='Server port (default: 5000)')
    
    args = parser.parse_args()
    
    server = FederatedServer(args.config, host=args.host, port=args.port)
    
    if args.rounds:
        server.num_rounds = args.rounds
    
    logger.info(f"Starting server on {server.host}:{server.port}")
    logger.info("For network access, use --host 0.0.0.0 (default)")
    logger.info("Share your IP address with team members for them to connect")
    
    server.start()


if __name__ == "__main__":
    main()

