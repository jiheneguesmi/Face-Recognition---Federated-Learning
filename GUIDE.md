# Complete Guide - Federated Learning for Facial Recognition

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Team Members](#team-members)
3. [Understanding Federated Learning](#understanding-federated-learning)
4. [Installation & Setup](#installation--setup)
5. [Data Collection & Preprocessing](#data-collection--preprocessing)
6. [Network Setup](#network-setup)
7. [Training Workflow](#training-workflow)
8. [How Aggregation Works](#how-aggregation-works)
9. [Troubleshooting](#troubleshooting)
10. [Evaluation](#evaluation)

---

## 📋 Project Overview

This project implements a **Federated Learning** system for facial recognition that can recognize 4 team members. Each team member trains a model on their own face data locally, and the models are aggregated at a central server to create a unified facial recognition model.

### Key Features
- **Privacy-Preserving**: Face images never leave each member's machine
- **Distributed Training**: Each member trains on their own machine
- **Automatic Aggregation**: Server automatically aggregates models using FedAvg
- **Collaborative Learning**: Model learns from all members without sharing data

---

## 👥 Team Members

1. **omarmej** (PM - Project Manager)
2. **abir**
3. **omarbr**
4. **jihene**

---

## 🎓 Understanding Federated Learning

### What is Federated Learning?

Federated Learning is a machine learning approach where:
- **Training data remains on each client's device** (privacy-preserving)
- **Only model weights/updates are shared**, not the actual data
- **A central server aggregates model updates** from all clients
- **The aggregated model learns from all clients' data** without seeing it

### Why Use Federated Learning?

1. **Privacy**: Face images never leave each member's computer
2. **Security**: Sensitive biometric data stays local
3. **Collaborative Learning**: Model benefits from all members' data
4. **Real-world Application**: Mimics how federated learning works in production

### Key Concept: How Does Aggregation Work?

**Question**: How does aggregation work when everyone is on different machines?

**Answer**: The server coordinates everything automatically!

- Each member trains locally on their own machine
- Only model weights (not images) are sent to the server
- Server aggregates weights using Federated Averaging (FedAvg)
- Server distributes the updated model back to all clients
- Process repeats for multiple rounds

---

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)
- All machines on the same network (same WiFi/LAN)

### Step 1: Clone Repository

```bash
git clone git@github.com:Mejri1/federated-learning.git
cd federated-learning
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter Pillow installation errors, they can be safely ignored if Pillow 11.3.0 or higher is already installed.

### Step 3: Verify Setup

```bash
python scripts/verify_setup.py
```

This will check:
- All dependencies are installed
- All directories exist
- All required files are present

### Step 4: Setup Project Structure

```bash
python setup.py
```

This creates all necessary directories automatically.

---

## 📸 Data Collection & Preprocessing

### Step 1: Collect Face Images

Each member needs to collect **50-100 photos** of themselves with:
- Different angles (front, side, slightly turned)
- Different expressions (smile, neutral, etc.)
- Different lighting conditions
- Different backgrounds
- With and without glasses (if applicable)

**Image Requirements**:
- Clear face visibility
- Good lighting
- Minimum resolution: 160x160 pixels
- Formats: `.jpg`, `.jpeg`, `.png`

### Step 2: Organize Images

Place all your images in your folder:
- `data/raw/omarmej/` (for omarmej)
- `data/raw/abir/` (for abir)
- `data/raw/omarbr/` (for omarbr)
- `data/raw/jihene/` (for jihene)

### Step 3: Preprocess Data

Run the preprocessing script on your machine:

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

**What this does**:
- Detects faces in your images using MTCNN
- Aligns and crops faces
- Resizes to 160x160 pixels
- Saves processed images to `data/processed/[your_name]/`
- Removes images where no face is detected

**Check Results**:
- Verify processed images in `data/processed/[your_name]/`
- Ensure you have at least 20-30 processed images
- If too few images, add more raw images and re-run

---

## 🌐 Network Setup

Since each team member has their own machine, you need to connect over the network.

### Step 1: Find Server IP Address (PM's Machine)

#### On Windows:
```bash
ipconfig
```
Look for **"IPv4 Address"** under your network adapter (usually WiFi or Ethernet).
Example: `192.168.1.100`

#### On Linux/Mac:
```bash
ifconfig
# or
ip addr show
```
Look for **"inet"** address (usually starts with 192.168.x.x or 10.x.x.x)

### Step 2: Share IP Address

PM (omarmej) should share their IP address with the team.
- Example: "Server will be at 192.168.1.100:5000"
- Make sure everyone is on the same network (same WiFi/LAN)

### Step 3: Configure Firewall (If Needed)

#### Windows:
1. Open **Windows Defender Firewall**
2. Click **"Allow an app or feature through Windows Defender Firewall"**
3. Click **"Allow another app"**
4. Add Python and allow it for **Private networks**
5. Or allow **port 5000** for incoming connections

#### Linux:
```bash
sudo ufw allow 5000
# or
sudo firewall-cmd --add-port=5000/tcp --permanent
sudo firewall-cmd --reload
```

#### Mac:
1. System Preferences → Security & Privacy → Firewall
2. Click **"Firewall Options"**
3. Allow Python or add port 5000

### Step 4: Test Connection

From client machines:
```bash
# Test if you can reach the server
ping [SERVER_IP]
# Example: ping 192.168.1.100
```

---

## 🚀 Training Workflow

### Step 1: Start Server (PM's Machine - omarmej)

**Important**: Use `--host 0.0.0.0` to accept connections from other machines!

```bash
python src/server.py --host 0.0.0.0 --port 5000 --rounds 20
```

You should see:
```
Server listening on 0.0.0.0:5000
Waiting for clients to connect...
```

### Step 2: Connect Clients (Each Member's Machine)

**Replace `[SERVER_IP]` with the actual server IP address!**

#### On omarmej's machine (PM - also runs client):
```bash
# Can use localhost since server is on same machine
python src/client.py --member omarmej --server-address localhost:5000
# OR use the IP address like others
python src/client.py --member omarmej --server-address [SERVER_IP]:5000
```

#### On abir's machine:
```bash
python src/client.py --member abir --server-address [SERVER_IP]:5000
```

#### On omarbr's machine:
```bash
python src/client.py --member omarbr --server-address [SERVER_IP]:5000
```

#### On jihene's machine:
```bash
python src/client.py --member jihene --server-address [SERVER_IP]:5000
```

### Step 3: Training Process

Once all clients are connected, the training process begins automatically:

1. **Server sends global model** to all clients
2. **Each client trains locally** on their own face images
3. **Clients send model weights** to server (not images!)
4. **Server aggregates weights** using FedAvg
5. **Server sends updated model** back to all clients
6. **Process repeats** for the specified number of rounds

### Step 4: Monitor Progress

- Watch server logs to see client connections
- Monitor training progress in each client terminal
- Checkpoints are saved automatically in `models/checkpoints/`
- Final model is saved in `models/saved/final_model.pth`

---

## 🔄 How Aggregation Works

### The Process

```
Round 1:
┌─────────────────────────────────────────────────────────┐
│ Server (omarmej's machine - [SERVER_IP])               │
│ - Has initial pretrained model                         │
│ - Sends model to all clients                           │
└─────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ omarmej      │ │ abir         │ │ omarbr       │ │ jihene       │
│ Machine      │ │ Machine      │ │ Machine      │ │ Machine      │
│              │ │              │ │              │ │              │
│ Receives     │ │ Receives     │ │ Receives     │ │ Receives     │
│ model        │ │ model        │ │ model        │ │ model        │
│              │ │              │ │              │ │              │
│ Trains on    │ │ Trains on    │ │ Trains on    │ │ Trains on    │
│ local images │ │ local images │ │ local images │ │ local images │
│ (50-100 pics)│ │ (50-100 pics)│ │ (50-100 pics)│ │ (50-100 pics)│
│              │ │              │ │              │ │              │
│ Sends        │ │ Sends        │ │ Sends        │ │ Sends        │
│ weights ─────┼─┼──────────────┼─┼──────────────┼─┼──────────────┤
│ (~50 MB)     │ │ (~50 MB)     │ │ (~50 MB)     │ │ (~50 MB)     │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
         │              │              │              │
         └──────────────┴──────────────┴──────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Server Aggregates (Federated Averaging):                │
│                                                         │
│ New Global Model =                                      │
│   (omarmej_weights × omarmej_samples +                  │
│    abir_weights × abir_samples +                        │
│    omarbr_weights × omarbr_samples +                    │
│    jihene_weights × jihene_samples) /                   │
│   (total_samples)                                       │
│                                                         │
│ - Creates new improved global model                    │
│ - Sends updated model to all clients                   │
└─────────────────────────────────────────────────────────┘

Round 2: Repeat with improved model...
Round 3: Repeat with even better model...
...
Round 20: Final model ready!
```

### Key Points

#### What Gets Transmitted?
- ✅ **Model weights** (trained parameters) - ~50-100 MB
- ❌ **NOT images** - Images stay on each machine
- ❌ **NOT raw data** - Only model parameters

#### Where Does Training Happen?
- ✅ **On each client's machine** - Local training
- ✅ **Server only aggregates** - Doesn't train

#### Where Is Data Stored?
- ✅ **On each member's machine** - Never leaves
- ❌ **NOT on server** - Server never sees images

#### How Does Aggregation Work?
- ✅ **Server receives weights from all clients**
- ✅ **Server averages the weights** (FedAvg algorithm)
- ✅ **Server creates new global model**
- ✅ **Server distributes updated model**

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "Connection refused"

**Solutions**:
- Check server is running
- Check server is using `--host 0.0.0.0` (not `localhost`)
- Check firewall allows port 5000
- Verify IP address is correct

#### 2. "Cannot connect to server"

**Solutions**:
- Ensure all machines are on the **same network** (same WiFi)
- Check IP address is correct
- Try pinging the server: `ping [SERVER_IP]`
- Check firewall settings

#### 3. "No images found"

**Solutions**:
- Check image paths in `data/raw/[your_name]/`
- Ensure correct file formats (.jpg, .png)
- Verify member name matches folder name

#### 4. "No face detected"

**Solutions**:
- Ensure images contain clear faces
- Check image quality and lighting
- Try different images

#### 5. "Out of memory"

**Solutions**:
- Reduce batch size in `config.yaml`
- Use smaller image resolution
- Process fewer images

#### 6. "Timeout waiting for clients"

**Solutions**:
- Ensure all clients are ready before server starts
- Check network connectivity
- Verify server is still running
- Coordinate with team to connect simultaneously

#### 7. Import errors

**Solutions**:
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8 or higher)
- Verify virtual environment is activated (if using)

---

## 📊 Evaluation

After training completes, evaluate the final model:

```bash
python scripts/evaluate.py --model-path models/saved/final_model.pth
```

This will:
- Test the model on a validation set
- Calculate accuracy for each member
- Generate confusion matrix
- Show classification report
- Save visualization plots

### Expected Performance

- **Round 1-5**: Learning basic features (40-60% accuracy)
- **Round 6-10**: Improving discrimination (60-80% accuracy)
- **Round 11-20**: Fine-tuning (80-95% accuracy)

---

## 📝 Complete Checklist

### Before Training:
- [ ] All members have cloned the repository
- [ ] All members have installed dependencies
- [ ] All members have collected face images (50-100 per person)
- [ ] All members have preprocessed their data
- [ ] PM found their IP address
- [ ] PM shared IP address with team
- [ ] Firewall configured (if needed)
- [ ] All machines on same network

### During Training:
- [ ] PM started server with `--host 0.0.0.0`
- [ ] All clients connected with correct server IP
- [ ] Server shows all 4 clients connected
- [ ] Training progresses through rounds
- [ ] Checkpoints saved regularly

### After Training:
- [ ] Final model saved on server
- [ ] Model evaluated
- [ ] Results documented
- [ ] Results shared with team

---

## 🎯 Summary

### The Key Insight

**Each member**: 
- Has their own machine
- Has their own data (never shared)
- Trains locally on their images

**Server**: 
- Coordinates training
- Aggregates model weights
- Distributes updated models

**Network**: 
- Used only for transmitting model weights (not images)
- Small data transfer (~50-100 MB per update)

**Result**: 
- A model that recognizes all 4 members
- Without ever sharing face images
- Privacy preserved!

---

## 📚 Additional Resources

- **Configuration**: Edit `config.yaml` to customize training parameters
- **Model Architecture**: FaceNet-based (InceptionResnetV1) with 512-dimensional embeddings
- **Federated Averaging**: Weighted average based on number of samples per client
- **Project Structure**: See `README.md` for complete project structure

---

## 🆘 Need Help?

1. Check this guide for common issues
2. Review error messages carefully
3. Check network connectivity
4. Verify all dependencies are installed
5. Consult with team members
6. Check server and client logs

---

**This is the power of federated learning: Learn together, without sharing data!** 🚀

