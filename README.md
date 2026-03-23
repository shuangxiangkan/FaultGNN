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
# BC network (hypercube), 10 dimensional (1024 nodes), 8 fault nodes
python run_comparison.py --graph_type bc --n 10 --fault_count 8

# Augmented k-ary n-cube AQ_{n,k}, n=3 dimension, k=3 base (27 nodes), 2 fault nodes
python run_comparison.py --graph_type augmented_k_ary_n_cube --n 3 --k 3 --fault_count 2

# Watts-Strogatz small-world graph: k_ws=6 => n_ws=64 nodes, p_ws=0.1
python run_comparison.py --graph_type watts_strogatz --k 6 --fault_count 2

# Use outgoing test features instead of the default incoming
python run_comparison.py --graph_type bc --n 8 --fault_count 2 --feature_mode outgoing

# Run feature ablation study (compares incoming / outgoing / concat)
python run_comparison.py --graph_type bc --n 8 --fault_count 2 --ablation_feature

# Custom training parameters
python run_comparison.py --graph_type bc --n 8 --fault_count 2 \
    --num_graphs 1000 --epochs 100 --lr 0.002 --seed 42
```

#### Supported Graph Types

| Type | Description | Scale Parameter |
|------|-------------|-----------------|
| `bc` | BC Network (hypercube) | `n` = dimension, nodes = 2^n |
| `watts_strogatz` | Watts-Strogatz small-world (irregular) | `k` = k_ws, n_ws = 2^k_ws, p_ws = 0.1 (default) |
| `augmented_k_ary_n_cube` | Augmented k-ary n-cube | `n`, `k` |

#### Node Feature Modes

The `--feature_mode` flag controls how GNN node features are constructed from PMC test results:

| Mode | Description |
|------|-------------|
| `incoming` (default) | Neighbors test the current node. Feature = proportion of times neighbor v reports node u as faulty. |
| `outgoing` | Current node tests its neighbors. Feature = proportion of times node u reports neighbor v as faulty. |
| `concat` | Concatenation of outgoing and incoming features (doubles feature dimension). |

#### Feature Ablation Study

Use `--ablation_feature` to automatically run all three feature modes and produce a comparison table:

```bash
python run_comparison.py --graph_type bc --n 8 --fault_count 2 --ablation_feature
```

Results are saved to `results/unified_comparison/ablation_feature/ablation_summary.txt`.

#### Full Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--graph_type` | `bc` | Graph type (`bc`, `watts_strogatz`, `augmented_k_ary_n_cube`) |
| `--n` | `8` | Graph scale (BC: dimension; ignored for watts_strogatz) |
| `--k` | `None` | watts_strogatz: k_ws (required, n_ws=2^k_ws); k-ary cube: base |
| `--p` | `None` | Watts-Strogatz rewiring probability (default 0.1) |
| `--fault_rate` | `None` | Fault node ratio (0.0-1.0) |
| `--fault_count` | `None` | Exact number of fault nodes |
| `--feature_mode` | `incoming` | GNN feature mode (`incoming`, `outgoing`, `concat`) |
| `--ablation_feature` | `False` | Run feature mode ablation study |
| `--num_graphs` | `1000` | Number of graphs to generate |
| `--num_rounds` | `10` | PMC test rounds per graph |
| `--epochs` | `100` | Training epochs |
| `--lr` | `0.002` | Learning rate |
| `--seed` | `42` | Random seed |
| `--gnn_hidden_dim` | `64` | GNN hidden layer dimension |
| `--gnn_num_layers` | `2` | Number of GNN layers |
| `--gnn_heads` | `8` | Number of GNN attention heads |
| `--gnn_batch_size` | `16` | GNN batch size (auto-adjusted for large graphs) |
| `--rnn_hidden_dims` | `64 32` | RNNIFDCom hidden layer dimensions |
| `--intermittent_prob` | `0.5` | Intermittent fault probability |

