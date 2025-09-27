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

# Verify Python version
python --version  # Should show Python 3.10 or higher

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
# Custom graph size and fault configuration (BC graph)
python run_comparison.py --n 16 --fault_count 8

# Adjust model parameters
python run_comparison.py --gat_hidden_dim 128 --epochs 200
```


