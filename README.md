# FaultGNN: A Graph Attention Network-Based Approach for System-Level Intermittent Fault Diagnosis

This repository contains the implementation of FaultGNN, a Graph Attention Network-Based Approach for System-Level Intermittent Fault Diagnosis

## 🚀 Quick Start

### Local Installation

**Requirements:**
- **Python >= 3.10** (Python 3.10 or higher required)
- PyTorch >= 1.9.0
- PyTorch Geometric >= 2.0.0

```bash
# Clone the repository
git clone https://github.com/shuangxiangkan/FaultGNN.git
cd FaultGNN

# Create virtual environment with Python 3.10+ (REQUIRED)
python3 -m venv pyg_env
source pyg_env/bin/activate  # On Windows: pyg_env\Scripts\activate

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

#### Usage Examples

```bash
# Custom graph size and fault configuration (10 dimensional hypercube with 8 fault nodes)
python run_comparison.py --n 10 --fault_count 8

# Watts-Strogatz: k_ws=6 => n_ws=64 nodes (like hypercube 2^n), p_ws=0.1
python run_comparison.py --graph_type watts_strogatz --k 6 --fault_count 2
```

#### Supported Graph Types

| Type | Description | Scale Parameter |
|------|-------------|-----------------|
| `bc` | BC Network (hypercube) | `n` = dimension, nodes = 2^n |
| `watts_strogatz` | Watts-Strogatz small-world (irregular) | `k` = k_ws, n_ws = 2^k_ws, p_ws = 0.1 |
| `augmented_k_ary_n_cube` | Augmented k-ary n-cube | `n`, `k` |


