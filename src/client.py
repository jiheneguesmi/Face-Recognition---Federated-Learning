"""
Client code for federated learning with synchronization
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import socket
import pickle
import struct
import io
import torch
import argparse
import logging
import time

from typing import Any

from src.model import initialize_model
from src.train import train_client
from src.utils import load_config

logger = logging.getLogger(__name__)


def send_pickle(sock: socket.socket, obj: Any):
    """
    Send a pickled object over socket with size header
    """
    data = pickle.dumps(obj)
    size = len(data)
    sock.sendall(struct.pack('!Q', size))
    sock.sendall(data)


def recv_pickle(sock: socket.socket, map_location: str = None) -> Any:
    """
    Receive a pickled object from socket
    
    Args:
        sock: Socket to receive from
        map_location: Device to map tensors to (e.g., 'cpu', 'cuda:0')
                     If None and torch tensors are detected, maps to 'cpu'
    """
    size_data = b''
    while len(size_data) < 8:
        chunk = sock.recv(8 - len(size_data))
        if not chunk:
            raise ConnectionError("Connection closed while receiving size")
        size_data += chunk
    
    size = struct.unpack('!Q', size_data)[0]
    
    data = b''
    while len(data) < size:
        chunk = sock.recv(min(size - len(data), 4096 * 10))
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        data += chunk
    
    # Handle PyTorch tensor device mapping
    if map_location is None:
        # Default to CPU if not specified to avoid CUDA/CPU mismatches
        map_location = 'cpu'
    
    try:
        # Use torch.load with map_location for PyTorch tensors
        buffer = io.BytesIO(data)
        return torch.load(buffer, map_location=map_location, weights_only=False)
    except Exception:
        # If torch.load fails (not a torch object), fall back to pickle
        return pickle.loads(data)


class FederatedClient:
    """
    Client for federated learning with synchronization
    """
    
    def __init__(self, member_name: str, config_path: str = "config.yaml"):
        """
        Initialize federated client
        """
        self.member_name = member_name
        self.config = load_config(config_path)
        
        server_config = self.config.get('server', {})
        self.server_host = server_config.get('host', 'localhost')
        self.server_port = server_config.get('port', 5000)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Client {member_name} initialized, using device: {self.device}")
    
    def connect_to_server(self):
        """
        Connect to the federated learning server
        """
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((self.server_host, self.server_port))
                logger.info(f"Connected to server at {self.server_host}:{self.server_port}")
                return client_socket
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect to server after {max_retries} attempts")
                    raise
    
    def register_with_server(self, client_socket):
        """
        Register client with the server and wait for acknowledgment
        """
        message = {
            'type': 'register',
            'member_name': self.member_name
        }
        send_pickle(client_socket, message)
        logger.info(f"Sent registration request as {self.member_name}")
        
        # Wait for registration response
        response = recv_pickle(client_socket)
        if response.get('type') == 'register_response':
            if response.get('status') == 'success':
                registered_count = response.get('registered_clients', 0)
                expected_count = response.get('expected_clients', 0)
                logger.info(f"✓ Registered with server ({registered_count}/{expected_count} clients registered)")
            else:
                error_msg = response.get('message', 'Unknown error')
                raise ConnectionError(f"Registration failed: {error_msg}")
        else:
            raise ValueError(f"Unexpected response type during registration: {response.get('type')}")
    
    def receive_global_model(self, client_socket):
        """
        Receive global model from server
        
        Note: This will wait until the server sends the model, which happens
        only after all clients have registered for the current round.
        """
        logger.info("Waiting for global model from server...")
        logger.info("(This may take a while if other clients are still connecting...)")
        
        try:
            # Receive with CPU mapping first, then move to client's device
            # This call will block until server sends the model
            response = recv_pickle(client_socket, map_location='cpu')
            
            if response.get('type') == 'model':
                model = initialize_model(self.config)
                # Load state dict (already on CPU from recv_pickle)
                model.load_state_dict(response['model_state'])
                # Move model to client's device (cpu or cuda)
                model = model.to(self.device)
                round_num = response.get('round', 'N/A')
                logger.info(f"✓ Received global model for round {round_num}")
                return model
            else:
                raise ValueError(f"Invalid response from server: expected 'model', got '{response.get('type')}'")
        except (ConnectionError, OSError) as e:
            logger.error(f"Connection error while waiting for model: {e}")
            logger.error("The server may have timed out waiting for other clients or closed the connection.")
            raise
    
    def send_model_update(self, client_socket, model_state: dict, sample_count: int):
        """
        Send trained model update to server
        """
        logger.info(f"Sending trained model to server ({sample_count} samples)...")
        
        # Convert all tensors in state dict to CPU for device-agnostic transfer
        cpu_model_state = {}
        for key, value in model_state.items():
            if isinstance(value, torch.Tensor):
                cpu_model_state[key] = value.cpu()
            else:
                cpu_model_state[key] = value
        
        message = {
            'type': 'send_update',
            'member_name': self.member_name,
            'model_state': cpu_model_state,
            'sample_count': sample_count
        }
        
        send_pickle(client_socket, message)
        
        # Wait for acknowledgment
        response = recv_pickle(client_socket)
        
        if response.get('type') == 'ack':
            logger.info("✓ Model update acknowledged by server")
        else:
            logger.warning("No acknowledgment from server")
    
    def train_and_update(self, round_num: int):
        """
        Connect to server, receive model, train, and send update
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"CLIENT {self.member_name} - ROUND {round_num}")
        logger.info(f"{'='*60}\n")
        
        client_socket = None
        try:
            # Connect to server
            client_socket = self.connect_to_server()
            
            # Register with server
            self.register_with_server(client_socket)
            
            # Receive global model from server
            model = self.receive_global_model(client_socket)
            
            # Train model locally
            logger.info(f"Starting local training...")
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
            
            logger.info(f"✓ Local training completed")
            
            # Send model update to server
            self.send_model_update(client_socket, model_state, sample_count)
            
            logger.info(f"✓ Round {round_num} completed for {self.member_name}\n")
        
        except Exception as e:
            logger.error(f"Error in training round: {e}", exc_info=True)
        finally:
            if client_socket:
                client_socket.close()
    
    def run(self, num_rounds: int = None):
        """
        Run federated learning client for multiple rounds
        """
        if num_rounds is None:
            training_config = self.config.get('training', {})
            num_rounds = training_config.get('num_rounds', 20)
        
        logger.info(f"Starting federated learning for {self.member_name} ({num_rounds} rounds)")
        logger.info(f"Server: {self.server_host}:{self.server_port}\n")
        
        for round_num in range(1, num_rounds + 1):
            try:
                self.train_and_update(round_num)
                
                if round_num < num_rounds:
                    # Wait before next round
                    logger.info(f"Waiting before next round...\n")
                    time.sleep(10)
            
            except Exception as e:
                logger.error(f"Error in round {round_num}: {e}")
                break
        
        logger.info(f"✓ Federated learning completed for {self.member_name}")


def main():
    """
    Main function to run the client
    """
    parser = argparse.ArgumentParser(description='Federated Learning Client (Synchronized)')
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