# FaultGNN: A Graph Attention Network-Based Approach for System-Level Intermittent Fault Diagnosis

This repository contains the implementation of FaultGNN, a Graph Attention Network-Based Approach for System-Level Intermittent Fault Diagnosis

## 🚀 Quick Start

### Local Installation

**Requirements:**
- **Python 3.10.x** (Strictly required - other versions will not work)
- PyTorch >= 1.9.0
- PyTorch Geometric >= 2.0.0

```bash
# Clone the repository
git clone https://github.com/shuangxiangkan/FaultGNN.git
cd FaultGNN

# Create virtual environment with Python 3.10 (REQUIRED)
python3.10 -m venv pyg_env
source pyg_env/bin/activate  # On Windows: pyg_env\Scripts\activate

# Verify Python version
python --version  # Should show Python 3.10.x

# Install dependencies
pip install -r requirements.txt
```

## 📋 Running Model Comparison Experiments

### Using run_comparison.py

The `run_comparison.py` script provides a unified framework for comparing the performance of FaultGNN (Graph Attention Network) and RNNIFDCOM models on intermittent fault diagnosis tasks.

#### Features

- **Unified Comparison**: Compare GNN and RNNIFDCOM models side-by-side
- **Flexible Configuration**: Customizable parameters for different experimental setups
- **Comprehensive Evaluation**: Includes accuracy, F1-score, precision, recall metrics
- **Visualization**: Generates training curves and performance plots
- **Partial Symptom Support**: Test model robustness under missing symptom conditions

#### Basic Usage

```bash
# Activate the virtual environment
source pyg_env/bin/activate

# Run with default parameters
python run_comparison.py

# View all available options
python run_comparison.py --help
```

#### Default Parameters

- **Graph Type**: `bc` (Binary Cube/Hypercube Network)
- **Graph Size**: `n=8` (8 nodes)
- **Fault Configuration**: 5 fault nodes, intermittent probability 0.5
- **Dataset**: 1000 graphs, 10 rounds per graph
- **GAT Model**: 64 hidden dimensions, 2 layers, 8 attention heads
- **RNNIFDCOM Model**: [64, 32] hidden dimensions
- **Training**: 100 epochs, learning rate 0.002

#### Example Commands

```bash
# Basic comparison with default settings
python run_comparison.py

# Custom graph size and fault configuration
python run_comparison.py --n 16 --fault_count 8 --intermittent_prob 0.3

# Adjust model parameters
python run_comparison.py --gat_hidden_dim 128 --gat_num_layers 3 --rnn_hidden_dims 128 64

# Test with partial symptoms (20% nodes disabled)
python run_comparison.py --missing_ratio 0.2 --missing_type node_disable

# Extended training
python run_comparison.py --epochs 200 --lr 0.001

# Use different graph topology
python run_comparison.py --graph_type augmented_k_ary_n_cube --n 12
```

#### Output

The script generates:
- **Training logs**: Real-time progress and metrics
- **Performance comparison**: Accuracy, F1-score, precision, recall for both models
- **Training curves**: Loss and F1-score progression plots
- **Confusion matrices**: Detailed classification results
- **Results directory**: All outputs saved to timestamped folders

#### Available Graph Types

- `bc`: BC (Binary Cube/Hypercube) Network
- `augmented_k_ary_n_cube`: Augmented K-ary N-cube

#### Command Line Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--graph_type` | Type of graph topology | `bc` |
| `--n` | Number of nodes in the graph | `8` |
| `--fault_count` | Number of fault nodes | `5` |
| `--intermittent_prob` | Intermittent fault probability | `0.5` |
| `--num_rounds` | Number of diagnosis rounds | `10` |
| `--num_graphs` | Number of graphs in dataset | `1000` |
| `--gat_hidden_dim` | GAT hidden layer dimension | `64` |
| `--gat_num_layers` | Number of GAT layers | `2` |
| `--gat_heads` | Number of attention heads | `8` |
| `--rnn_hidden_dims` | RNNIFDCOM hidden dimensions | `[64, 32]` |
| `--epochs` | Training epochs | `100` |
| `--lr` | Learning rate | `0.002` |
| `--missing_ratio` | Ratio of missing symptoms | `0.0` |
| `--missing_type` | Type of symptom missing | `node_disable` |

### Version Compatibility

If you encounter import errors:

```python
RuntimeError: Cannot load FaultGNN model!
Your Python version: X.Y
Supported versions: 3.10
```

**Solutions:**
1. Use Python 3.10: `pyenv install 3.10.12 && pyenv local 3.10.12`
2. Use conda: `conda create -n faultgnn python=3.10`

## 📊 Understanding the Results

The comparison script evaluates both models on:
- **Classification Accuracy**: Overall correctness of fault detection
- **F1-Score**: Balanced measure of precision and recall
- **Precision**: Accuracy of positive fault predictions
- **Recall**: Coverage of actual faults detected
- **Robustness**: Performance under partial symptom conditions

Results help determine which model performs better for specific network topologies and fault scenarios.


