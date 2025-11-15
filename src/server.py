"""
Central server for federated learning with proper synchronization
"""

import socket
import pickle
import struct
import io
import threading
import time
import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any

from src.model import initialize_model
from src.aggregation import update_global_model, federated_averaging
from src.utils import load_config, create_directories, save_model

logger = logging.getLogger(__name__)


def send_pickle(sock: socket.socket, obj: Any):
    """
    Send a pickled object over socket with size header (matches client protocol)
    """
    data = pickle.dumps(obj)
    size = len(data)
    sock.sendall(struct.pack('!Q', size))
    sock.sendall(data)


def recv_pickle(sock: socket.socket, map_location: str = None) -> Any:
    """
    Receive a pickled object from socket (matches client protocol)
    
    Args:
        sock: Socket to receive from
        map_location: Device to map tensors to (e.g., 'cpu', 'cuda:0')
                     If None, defaults to 'cpu' to avoid device mismatches
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
        
        # Get expected number of clients from config
        client_config = self.config.get('client', {})
        member_names = client_config.get('member_names', [])
        self.expected_clients = len(member_names) if member_names else 4
        logger.info(f"Expecting {self.expected_clients} clients per round")
        
        # Client management - per round
        self.registered_clients = {}  # member_name -> (socket, address)
        self.client_models = {}  # member_name -> model_state
        self.client_sample_counts = {}  # member_name -> sample_count
        
        # Synchronization primitives
        self.registration_lock = threading.Lock()
        self.update_lock = threading.Lock()
        self.registration_event = threading.Event()  # Set when all clients registered
        self.model_sent_event = threading.Event()  # Set when models sent to all
        self.updates_received_event = threading.Event()  # Set when all updates received
        
        # Round management
        self.current_round = 0
        self.round_lock = threading.Lock()
        
        logger.info(f"Federated server initialized on {self.host}:{self.port}")
    
    def handle_client(self, client_socket, address):
        """
        Handle client connection with proper synchronization
        
        Args:
            client_socket: Client socket
            address: Client address
        """
        member_name = None
        try:
            logger.info(f"Client connected from {address}")
            
            while True:
                # Receive message from client using proper protocol
                message = recv_pickle(client_socket)
                msg_type = message.get('type')
                
                if msg_type == 'register':
                    # Client registration
                    member_name = message.get('member_name')
                    logger.info(f"Registration request from {member_name} at {address}")
                    
                    with self.registration_lock:
                        # Check if already registered
                        if member_name in self.registered_clients:
                            logger.warning(f"Client {member_name} already registered, rejecting duplicate")
                            response = {
                                'type': 'register_response',
                                'status': 'error',
                                'message': 'Already registered'
                            }
                            send_pickle(client_socket, response)
                            continue
                        
                        # Register client
                        self.registered_clients[member_name] = (client_socket, address)
                        logger.info(f"✓ Registered {member_name} ({len(self.registered_clients)}/{self.expected_clients} clients registered)")
                        
                        # Check if all clients registered
                        if len(self.registered_clients) >= self.expected_clients:
                            logger.info(f"✓ All {self.expected_clients} clients registered! Proceeding to send models...")
                            self.registration_event.set()
                        
                        # Send registration acknowledgment
                        response = {
                            'type': 'register_response',
                            'status': 'success',
                            'registered_clients': len(self.registered_clients),
                            'expected_clients': self.expected_clients
                        }
                        send_pickle(client_socket, response)
                    
                    # Wait for all clients to register and models to be sent
                    if self.registration_event.wait(timeout=120):  # Wait up to 2 minutes
                        # Wait a bit more to ensure all clients are ready
                        time.sleep(0.5)
                    else:
                        logger.warning(f"Registration timeout for {member_name}")
                    
                    # Now wait for server to send model (handled in run_round)
                    # The client will receive model automatically after registration
                
                elif msg_type == 'send_update':
                    # Receive model update from client
                    if member_name is None:
                        member_name = message.get('member_name')
                    
                    model_state = message.get('model_state')
                    sample_count = message.get('sample_count')
                    
                    logger.info(f"Received update from {member_name} ({sample_count} samples)")
                    
                    with self.update_lock:
                        self.client_models[member_name] = model_state
                        self.client_sample_counts[member_name] = sample_count
                        
                        logger.info(f"✓ Update received from {member_name} ({len(self.client_models)}/{self.expected_clients} updates received)")
                        
                        # Check if all updates received
                        if len(self.client_models) >= self.expected_clients:
                            logger.info(f"✓ All {self.expected_clients} updates received! Ready for aggregation...")
                            self.updates_received_event.set()
                    
                    # Send acknowledgment
                    response = {'type': 'ack', 'status': 'received'}
                    send_pickle(client_socket, response)
                
                elif msg_type == 'disconnect':
                    logger.info(f"Client {member_name} at {address} disconnected")
                    break
                else:
                    logger.warning(f"Unknown message type: {msg_type}")
        
        except ConnectionError as e:
            logger.info(f"Client {member_name or address} disconnected: {e}")
        except Exception as e:
            logger.error(f"Error handling client {member_name or address}: {e}", exc_info=True)
        finally:
            # Clean up registration if client disconnects
            if member_name and member_name in self.registered_clients:
                with self.registration_lock:
                    if member_name in self.registered_clients:
                        del self.registered_clients[member_name]
                        logger.info(f"Removed {member_name} from registered clients")
            client_socket.close()
    
    def send_models_to_all_clients(self, round_num: int):
        """
        Send global model to all registered clients
        
        Args:
            round_num: Current round number
        """
        logger.info(f"Sending global model (round {round_num}) to {len(self.registered_clients)} clients...")
        
        # Get model state dict and move all tensors to CPU for device-agnostic transfer
        model_state = self.global_model.state_dict()
        # Convert all tensors in state dict to CPU to avoid device mismatch
        cpu_model_state = {}
        for key, value in model_state.items():
            if isinstance(value, torch.Tensor):
                cpu_model_state[key] = value.cpu()
            else:
                cpu_model_state[key] = value
        
        clients_sent = 0
        
        for member_name, (client_socket, address) in self.registered_clients.items():
            try:
                response = {
                    'type': 'model',
                    'model_state': cpu_model_state,
                    'round': round_num,
                    'config': self.config
                }
                send_pickle(client_socket, response)
                clients_sent += 1
                logger.info(f"✓ Sent model to {member_name}")
            except Exception as e:
                logger.error(f"Failed to send model to {member_name}: {e}")
        
        logger.info(f"✓ Sent model to {clients_sent}/{len(self.registered_clients)} clients")
        self.model_sent_event.set()
    
    def aggregate_models(self):
        """
        Aggregate models from all clients (only if all updates received)
        """
        if len(self.client_models) < self.expected_clients:
            logger.warning(f"Only {len(self.client_models)}/{self.expected_clients} updates received, cannot aggregate")
            return False
        
        # Get models and sample counts
        models = list(self.client_models.values())
        sample_counts = list(self.client_sample_counts.values())
        
        # Aggregate using FedAvg
        aggregated_state = federated_averaging(models, sample_counts)
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state)
        
        logger.info(f"✓ Aggregated models from {len(models)} clients")
        return True
    
    def run_round(self, round_num: int):
        """
        Run one round of federated learning with proper synchronization
        
        Args:
            round_num: Current round number
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting round {round_num}/{self.num_rounds}")
        logger.info(f"{'='*60}\n")
        
        # Reset synchronization events
        self.registration_event.clear()
        self.model_sent_event.clear()
        self.updates_received_event.clear()
        
        # Reset per-round state
        with self.registration_lock:
            self.registered_clients.clear()
        with self.update_lock:
            self.client_models.clear()
            self.client_sample_counts.clear()
        
        logger.info(f"Waiting for {self.expected_clients} clients to register...")
        
        # Wait for all clients to register (with timeout)
        # Increased timeout to 10 minutes to account for variable client completion times
        registration_timeout = 600  # 10 minutes
        if not self.registration_event.wait(timeout=registration_timeout):
            with self.registration_lock:
                registered_count = len(self.registered_clients)
            logger.warning(f"Registration timeout: Only {registered_count}/{self.expected_clients} clients registered after {registration_timeout} seconds")
            logger.warning("This may be normal if clients are still finishing previous round or training...")
            
            # Check again after a short wait
            time.sleep(5)
            with self.registration_lock:
                registered_count = len(self.registered_clients)
            
            if registered_count < self.expected_clients:
                logger.error(f"Still only {registered_count}/{self.expected_clients} clients registered. Cannot proceed.")
                return False
            else:
                logger.info(f"✓ All {registered_count} clients registered after additional wait!")
        
        # Double-check we have all clients
        with self.registration_lock:
            if len(self.registered_clients) < self.expected_clients:
                logger.error(f"Only {len(self.registered_clients)}/{self.expected_clients} clients registered")
                return False
            
            logger.info(f"✓ All {self.expected_clients} clients registered!")
        
        # Send models to all clients
        self.send_models_to_all_clients(round_num)
        
        # Wait for all clients to complete training and send updates
        logger.info(f"Waiting for {self.expected_clients} clients to send their trained models...")
        logger.info("(Clients are now training locally...)")
        
        if not self.updates_received_event.wait(timeout=600):  # 10 minutes timeout for training
            updates_count = len(self.client_models)
            logger.error(f"Timeout: Only {updates_count}/{self.expected_clients} updates received")
            logger.error("Cannot aggregate. Some clients may have failed during training.")
            return False
        
        # Double-check we have all updates
        with self.update_lock:
            if len(self.client_models) < self.expected_clients:
                logger.error(f"Only {len(self.client_models)}/{self.expected_clients} updates received")
                return False
        
        # Aggregate models
        if not self.aggregate_models():
            return False
        
        # Save checkpoint
        if self.config.get('logging', {}).get('save_checkpoints', True):
            checkpoint_path = Path(self.config['paths']['checkpoints_dir']) / f"round_{round_num}.pth"
            save_model(self.global_model, str(checkpoint_path), {'round': round_num})
            logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
        
        logger.info(f"✓ Round {round_num} completed successfully\n")
        return True
    
    def start(self):
        """
        Start the federated learning server with proper synchronization
        """
        # Create socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(10)  # Allow more pending connections
        
        logger.info(f"Server listening on {self.host}:{self.port}")
        logger.info(f"Waiting for {self.expected_clients} clients to connect...")
        
        # Store client threads for cleanup
        client_threads = []
        active_round = None
        
        def accept_clients():
            """Accept client connections in a separate thread"""
            nonlocal active_round
            while True:
                try:
                    client_socket, address = server_socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    client_threads.append(client_thread)
                except OSError:
                    # Server socket closed
                    break
                except Exception as e:
                    logger.error(f"Error accepting client: {e}")
        
        # Start accepting clients in background thread
        accept_thread = threading.Thread(target=accept_clients, daemon=True)
        accept_thread.start()
        
        try:
            # Run federated learning rounds
            for round_num in range(1, self.num_rounds + 1):
                with self.round_lock:
                    self.current_round = round_num
                
                # Wait a bit before starting the round to allow clients to reconnect
                # This is especially important for rounds after the first one
                if round_num > 1:
                    logger.info(f"Waiting 15 seconds before starting round {round_num} to allow all clients to finish previous round...")
                    time.sleep(15)
                
                # Run the round (this will wait for clients, send models, wait for updates, aggregate)
                success = self.run_round(round_num)
                
                if not success:
                    logger.error(f"Round {round_num} failed. Stopping training.")
                    break
                
                # Wait a bit after completing the round to allow clients to disconnect
                if round_num < self.num_rounds:
                    logger.info(f"Round {round_num} completed. Waiting 5 seconds before next round...\n")
                    time.sleep(5)
            
            # Save final model
            if self.current_round > 0:
                final_model_path = Path(self.config['paths']['saved_models_dir']) / "final_model.pth"
                save_model(self.global_model, str(final_model_path), {'rounds': self.num_rounds})
                logger.info(f"✓ Final model saved to {final_model_path}")
        
        except KeyboardInterrupt:
            logger.info("\nServer stopped by user")
        finally:
            # Close server socket to stop accepting new connections
            try:
                server_socket.close()
            except:
                pass
            logger.info("Server shutdown complete")


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

