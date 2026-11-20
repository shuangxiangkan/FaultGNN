# FaultGNN: A Graph Neural Network Approach for Intermittent Fault Diagnosis in Multiprocessor Systems

This repository contains the implementation of FaultGNN, a GraphSAGE-based approach for intermittent fault diagnosis under the PMC model, together with the two baselines used in the paper (RNNIFDCom and FIFPDPMC).

## Installation

Requires Python >= 3.10.

```bash
git clone https://github.com/shuangxiangkan/FaultGNN.git
cd FaultGNN

python3 -m venv pyg_env
source pyg_env/bin/activate  # On Windows: pyg_env\Scripts\activate

pip install -r requirements.txt
```

## Usage

`run_comparison.py` generates the dataset, trains FaultGNN and the baselines, and reports accuracy and F1-score.

```bash
# Hypercube Q_10 with 7 faulty nodes, concatenated features, 5 syndrome rounds
python run_comparison.py --graph_type bc --n 10 --fault_count 7 --feature_mode concat --num_rounds 5

# Augmented k-ary n-cube AQ_{3,10} with a 10% fault ratio
python run_comparison.py --graph_type augmented_k_ary_n_cube --n 3 --k 10 --fault_rate 0.10

# Watts-Strogatz network WS_6 (64 nodes, rewiring probability 0.1) with 3 faulty nodes
python run_comparison.py --graph_type watts_strogatz --k 6 --fault_count 3

# List all options
python run_comparison.py --help
```

Supported graph types:

| `--graph_type` | Network | Size parameters |
|----------------|---------|-----------------|
| `bc` | Hypercube $Q_n$ | `--n` (dimension, $2^n$ nodes) |
| `augmented_k_ary_n_cube` | Augmented $k$-ary $n$-cube $AQ_{n,k}$ | `--n`, `--k` |
| `watts_strogatz` | Watts-Strogatz network $WS_k$ | `--k` ($2^k$ nodes, initial neighborhood size $k$) |

The number of faulty nodes is set with either `--fault_count` (exact number) or `--fault_rate` (ratio of the nodes). The node features are selected with `--feature_mode` (`incoming`, `outgoing`, or `concat`).

Results are written to `results/unified_comparison/`.
