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

Script for comparing FaultGNN and RNNIFDCOM model performance.

#### Basic Usage

```bash
# Activate virtual environment
source pyg_env/bin/activate

# Run with default parameters
python run_comparison.py

# View all parameters
python run_comparison.py --help
```

#### Main Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--graph_type` | Graph type (`bc`, `augmented_k_ary_n_cube`) | `bc` |
| `--n` | Number of nodes | `8` |
| `--fault_count` | Number of fault nodes | `5` |
| `--epochs` | Training epochs | `100` |
| `--lr` | Learning rate | `0.002` |
| `--gat_hidden_dim` | GAT hidden dimension | `64` |
| `--gat_num_layers` | GAT layers | `2` |
| `--rnn_hidden_dims` | RNN hidden dimensions | `[64, 32]` |

#### Usage Examples

```bash
# Custom graph size and fault configuration
python run_comparison.py --n 16 --fault_count 8

# Adjust model parameters
python run_comparison.py --gat_hidden_dim 128 --epochs 200

# Test with partial symptom missing
python run_comparison.py --missing_ratio 0.2
```

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


