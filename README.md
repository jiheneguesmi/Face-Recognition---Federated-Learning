# Federated Learning for Facial Recognition

## Project Overview

This project implements a **Federated Learning** system for facial recognition that can recognize 4 team members. Each team member trains a model on their own face data locally, and the models are aggregated at a central server to create a unified facial recognition model.

## Project Goals

- Build a facial recognition model that can identify all 4 team members
- Implement federated learning to preserve privacy (each member's data stays local)
- Fine-tune a pretrained face recognition model on custom data
- Aggregate models from multiple clients using Federated Averaging (FedAvg)

## Project Architecture

### Federated Learning Approach

We use **Centralized Federated Learning** with a **coordinator/server** pattern:

1. **Central Server/Coordinator**: 
   - Initializes the global model
   - Receives model updates from each client
   - Aggregates models using Federated Averaging (FedAvg)
   - Distributes the aggregated model back to clients
   - Coordinates training rounds

2. **Clients (Each Team Member)**:
   - Train the model on their own face images locally
   - Send model weights (not data) to the server
   - Receive updated global model from server
   - Continue training for multiple rounds

### Workflow
```
Round 1:
1. Server initializes global model (pretrained face recognition)
2. Server sends model to omarmej
3. omarmej trains on their face images → sends updated weights to server
4. Server sends model to abir
5. abir trains on their face images → sends updated weights to server
6. Server sends model to omarbr
7. omarbr trains on their face images → sends updated weights to server
8. Server sends model to jihene
9. jihene trains on their face images → sends updated weights to server
10. Server aggregates all weights using FedAvg → creates new global model

Round 2:
- Repeat steps 2-10 with the aggregated global model

... continue for N rounds
```

## Project Structure
```
federated-learning-project/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
├── config.yaml              # Configuration file
│
├── data/                    # Data directory
│   ├── raw/                 # Raw images (each member puts their images here)
│   │   ├── omarmej/         # omarmej's face images
│   │   ├── abir/            # abir's face images
│   │   ├── omarbr/          # omarbr's face images
│   │   └── jihene/          # jihene's face images
│   └── processed/           # Processed/cleaned data (auto-generated)
│
├── models/                  # Model directory
│   ├── pretrained/          # Pretrained models
│   ├── checkpoints/         # Training checkpoints
│   └── saved/               # Final saved models
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── model.py             # Model definition (FaceNet-based)
│   ├── dataset.py           # Dataset class for face images
│   ├── train.py             # Local training script
│   ├── aggregation.py       # Federated averaging logic
│   ├── server.py            # Central server/coordinator
│   ├── client.py            # Client-side code
│   └── utils.py             # Utility functions
│
├── scripts/                 # Utility scripts
│   ├── prepare_data.py      # Data preprocessing script
│   ├── evaluate.py          # Model evaluation script
│   └── verify_setup.py      # Setup verification
│
└── logs/                    # Training logs
    └── tensorboard/         # TensorBoard logs
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)

### Installation

1. **Clone the repository** (if using Git):
```bash
   git clone <repository-url>
   cd federated-learning-project
```

2. **Create a virtual environment** (recommended):
```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
```

3. **Install dependencies**:
```bash
   pip install -r requirements.txt
```

### Setup Steps for Each Team Member

1. **Prepare your face images**:
   - Collect 50-100 photos of your face (various angles, lighting, expressions)
   - Place all your images in your folder: `data/raw/omarmej/`, `data/raw/abir/`, `data/raw/omarbr/`, or `data/raw/jihene/`
   - Supported formats: `.jpg`, `.jpeg`, `.png`

2. **Preprocess your data**:
```bash
   # For omarmej
   python scripts/prepare_data.py --member omarmej
   
   # For abir
   python scripts/prepare_data.py --member abir
   
   # For omarbr
   python scripts/prepare_data.py --member omarbr
   
   # For jihene
   python scripts/prepare_data.py --member jihene
```
   This will:
   - Detect and align faces in your images
   - Resize images to standard size
   - Save processed images to `data/processed/[your_name]/`

## How Federated Learning Works

### Key Concepts

1. **Privacy Preservation**: Each member's face images never leave their local machine. Only model weights are shared.

2. **Federated Averaging (FedAvg)**:
   - Server receives model weights from all clients
   - Aggregates weights by averaging them
   - Weighted average based on number of training samples per client
   - Formula: `w_global = Σ(n_i * w_i) / Σ(n_i)`
     - `w_i`: weights from client i
     - `n_i`: number of samples from client i

3. **Training Rounds**:
   - Multiple rounds of training-aggregation cycle
   - Each round improves the global model
   - Typically 10-50 rounds depending on convergence

### Why Federated Learning?

- **Privacy**: Face images stay on each member's device
- **Collaborative Learning**: Model learns from all members without sharing data
- **Distributed**: No need for centralized data collection
- **Real-world Application**: Mimics real federated learning scenarios

## Complete Guide

**For detailed instructions, see [GUIDE.md](GUIDE.md)** - Complete guide covering:
- Installation & Setup
- Data Collection & Preprocessing
- Network Setup
- Training Workflow
- How Aggregation Works
- Troubleshooting
- Evaluation

## Usage

### Centralized Server Approach

This project uses a **centralized server** for federated learning coordination.

**Important**: For distributed training (each member on their own machine), see [DISTRIBUTED_WORKFLOW.md](DISTRIBUTED_WORKFLOW.md)

**Step 1: Start the server** (run by PM - omarmej):
```bash
# For network access (each member on different machine):
python src/server.py --host 0.0.0.0 --port 5000 --rounds 20

# Find your IP address and share with team:
# Windows: ipconfig
# Linux/Mac: ifconfig
```

**Step 2: Each member runs the client** (in separate terminals):
```bash
# Replace [SERVER_IP] with actual server IP (e.g., 192.168.1.100)

# omarmej (PM) - can use localhost since server is on same machine
python src/client.py --member omarmej --server-address localhost:5000

# abir (on abir's machine)
python src/client.py --member abir --server-address [SERVER_IP]:5000

# omarbr (on omarbr's machine)
python src/client.py --member omarbr --server-address [SERVER_IP]:5000

# jihene (on jihene's machine)
python src/client.py --member jihene --server-address [SERVER_IP]:5000
```

**See [NETWORK_SETUP.md](NETWORK_SETUP.md) for detailed network setup instructions.**

**How it works**:
1. Server initializes the global model
2. Each client connects to the server
3. Server sends the global model to each client
4. Each client trains locally on their own face images
5. Clients send updated model weights to the server
6. Server aggregates all weights using Federated Averaging (FedAvg)
7. Process repeats for the specified number of rounds

## Model Architecture

We use a **FaceNet-based** architecture:
- **Base Model**: Pretrained InceptionResnetV1 from `facenet-pytorch`
- **Embedding Size**: 512-dimensional face embeddings
- **Classifier**: Fully connected layer for 4-class classification (one per team member)
- **Loss Function**: Cross-entropy loss

## Configuration

Edit `config.yaml` to customize:
- Number of training rounds
- Learning rate
- Batch size
- Number of epochs per client
- Model hyperparameters
- Data paths

## Evaluation

Evaluate the final model:
```bash
python scripts/evaluate.py --model-path models/saved/final_model.pth
```

This will:
- Test the model on a validation set
- Calculate accuracy for each member
- Generate confusion matrix
- Show sample predictions

## Workflow Summary

### For Project Manager (omarmej):
1. Set up the project structure (this repository)
2. Coordinate the training process
3. Run the central server
4. Monitor training progress
5. Evaluate final model

### For Each Team Member:
1. Collect your face images (50-100 photos)
2. Place them in `data/raw/[your_name]/` (omarmej, abir, omarbr, or jihene)
3. Run data preprocessing: `python scripts/prepare_data.py --member [your_name]`
4. Run client training script: `python src/client.py --member [your_name] --server-address localhost:5000`
5. Wait for aggregation and next round

## Notes

- **Data Quality**: More diverse images (different angles, lighting, expressions) = better model
- **Training Time**: Each round may take 10-30 minutes depending on hardware and data size
- **Communication**: Coordinate with team members about when to start each round
- **Backup**: Save model checkpoints regularly
- **Testing**: Test the model after each round to monitor improvement

## Troubleshooting

### Common Issues:

1. **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

2. **Face detection fails**: 
   - Ensure images contain clear faces
   - Check image quality and lighting
   - Try different images

3. **Out of memory**:
   - Reduce batch size in `config.yaml`
   - Use smaller image resolution
   - Process fewer images

4. **Model not converging**:
   - Increase number of training rounds
   - Increase epochs per client
   - Check data quality and quantity

5. **Connection issues**:
   - Ensure server is running before clients connect
   - Check server address and port
   - Verify firewall settings

## Resources

- [Federated Learning Paper](https://arxiv.org/abs/1602.05629)
- [FaceNet Paper](https://arxiv.org/abs/1503.03832)
- [facenet-pytorch Documentation](https://github.com/timesler/facenet-pytorch)

## Contributing

Each team member should:
1. Work on their own branch (if using Git)
2. Test their code before pushing
3. Follow the project structure
4. Document any changes

## License

Academic project for educational purposes.
