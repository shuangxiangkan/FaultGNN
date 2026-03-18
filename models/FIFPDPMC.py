"""
FIFPDPMC: Fast Intermittent Fault Probabilistic Diagnosis under PMC Model

Traditional/classical baseline algorithm from:
  Song, J., Lin, L., Huang, Y., & Hsieh, S.-Y. (2023).
  "Intermittent Fault Diagnosis of Split-Star Networks and its Applications."
  IEEE Transactions on Parallel and Distributed Systems, 34(4), 1253-1264.

Algorithm summary:
  1. For each stage j: collect syndrome, remove edges with at least one test result=1
  2. Find largest connected component in remaining graph (fault-free nodes in stage j)
  3. Vfault.free = intersection of all stages' largest components
  4. Vintermittent.fault = union of nodes excluded in any stage
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional


def _find_largest_component(G: nx.Graph) -> set:
    """
    Find the largest connected component using DFS (OLC algorithm from paper).

    Args:
        G: NetworkX graph

    Returns:
        Set of nodes in the largest component
    """
    if G.number_of_nodes() == 0:
        return set()

    components = list(nx.connected_components(G))
    if not components:
        return set()

    largest = max(components, key=len)
    return set(largest)


def fifpdpmc_single_graph(
    sparse_syndromes: List[List[Tuple]],
    graph_structure: Dict[str, Any],
    num_stages: Optional[int] = None,
) -> np.ndarray:
    """
    Run FIFPDPMC algorithm on a single graph.

    Args:
        sparse_syndromes: List of rounds, each round is list of (u, v, test_u_to_v, test_v_to_u)
            for each edge. u, v are node labels from the graph.
        graph_structure: Dict with keys 'edges', 'vertices', 'node_to_idx'
        num_stages: Number of stages (rounds) to use. If None, use all available rounds.

    Returns:
        np.ndarray: Predictions for each node (0=fault-free, 1=faulty), indexed by node_to_idx order
    """
    edges = graph_structure['edges']
    vertices = graph_structure['vertices']
    node_to_idx = graph_structure['node_to_idx']

    num_vertices = len(vertices)
    num_rounds = len(sparse_syndromes)

    if num_rounds == 0:
        # No syndrome data: predict all fault-free (conservative)
        return np.zeros(num_vertices, dtype=np.int64)

    x = num_stages if num_stages is not None else num_rounds
    x = min(x, num_rounds)

    # Build base graph from edges (undirected)
    G_base = nx.Graph()
    G_base.add_edges_from(edges)

    # Per-stage: fault-free nodes = nodes in largest component after edge removal
    fault_free_per_stage: List[set] = []

    for stage_idx in range(x):
        round_syndrome = sparse_syndromes[stage_idx]

        # Build dict: (u,v) -> (test_u_to_v, test_v_to_u) for quick lookup
        # Edge may be stored as (u,v) or (v,u) in sparse format
        edge_tests = {}
        for u, v, test_u_to_v, test_v_to_u in round_syndrome:
            edge_tests[(u, v)] = (test_u_to_v, test_v_to_u)
            edge_tests[(v, u)] = (test_v_to_u, test_u_to_v)

        # Test subgraph: keep only edges where BOTH test results are 0
        # (remove edges with at least one 1)
        kept_edges = []
        for u, v in edges:
            tests = edge_tests.get((u, v), edge_tests.get((v, u), (1, 1)))
            t_uv, t_vu = tests
            if t_uv == 0 and t_vu == 0:
                kept_edges.append((u, v))

        G_j = nx.Graph()
        G_j.add_edges_from(kept_edges)
        # Ensure all vertices exist (isolated nodes)
        G_j.add_nodes_from(vertices)

        largest = _find_largest_component(G_j)
        fault_free_per_stage.append(largest)

    # Vfault.free = N1 ∩ N2 ∩ ... ∩ Nx
    v_fault_free = fault_free_per_stage[0]
    for s in range(1, x):
        v_fault_free = v_fault_free.intersection(fault_free_per_stage[s])

    # Predictions: 0 if in Vfault.free, else 1
    preds = np.ones(num_vertices, dtype=np.int64)
    for node in v_fault_free:
        idx = node_to_idx.get(node)
        if idx is not None:
            preds[idx] = 0

    return preds


def _get_test_indices(
    num_graphs: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> List[int]:
    """Compute test indices using same split logic as UnifiedDatasetLoader.split_dataset."""
    rng = np.random.default_rng(seed)
    graph_indices = rng.permutation(num_graphs)
    train_end = int(num_graphs * train_ratio)
    val_end = train_end + int(num_graphs * val_ratio)
    return graph_indices[val_end:].tolist()


def evaluate_fifpdpmc(
    raw_data: Dict,
    test_indices: Optional[List[int]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_stages: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate FIFPDPMC on test set from raw dataset.

    Args:
        raw_data: Dataset dict with 'sparse_syndromes', 'fault_states', 'graph_structures'
        test_indices: List of graph indices to evaluate. If None, use same split as
            UnifiedDatasetLoader (train_ratio, val_ratio, test_ratio, seed).
        train_ratio, val_ratio, test_ratio, seed: Used when test_indices is None
        num_stages: Number of stages for FIFPDPMC. If None, use all rounds from metadata.

    Returns:
        Dict with accuracy, f1_score, precision, recall, false_negative_rate, false_positive_rate,
        confusion_matrix, evaluated_nodes
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )

    sparse_syndromes_list = raw_data['sparse_syndromes']
    fault_states_list = raw_data['fault_states']
    graph_structures = raw_data['graph_structures']

    metadata = raw_data.get('metadata', {})
    num_rounds = metadata.get('num_rounds', 10)
    stages = num_stages if num_stages is not None else num_rounds

    if test_indices is None:
        test_indices = _get_test_indices(
            len(sparse_syndromes_list), train_ratio, val_ratio, test_ratio, seed
        )
    if not test_indices:
        return {
            'accuracy': 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'false_negative_rate': 0.0,
            'false_positive_rate': 0.0,
            'confusion_matrix': [[0, 0], [0, 0]],
            'evaluated_nodes': 0,
        }

    all_preds = []
    all_true = []

    for idx in test_indices:
        sparse_syndromes = sparse_syndromes_list[idx]
        fault_states = fault_states_list[idx]
        graph_structure = graph_structures[idx]

        preds = fifpdpmc_single_graph(
            sparse_syndromes, graph_structure, num_stages=stages
        )
        true_labels = np.array(fault_states, dtype=np.int64)

        all_preds.extend(preds.tolist())
        all_true.extend(true_labels.tolist())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    if len(all_preds) == 0:
        return {
            'accuracy': 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'false_negative_rate': 0.0,
            'false_positive_rate': 0.0,
            'confusion_matrix': [[0, 0], [0, 0]],
            'evaluated_nodes': 0,
        }

    accuracy = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, zero_division=0)
    precision = precision_score(all_true, all_preds, zero_division=0)
    recall = recall_score(all_true, all_preds, zero_division=0)
    cm = confusion_matrix(all_true, all_preds)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    else:
        fnr = 0.0
        fpr = 0.0

    return {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'false_negative_rate': float(fnr),
        'false_positive_rate': float(fpr),
        'confusion_matrix': cm.tolist(),
        'evaluated_nodes': int(len(all_preds)),
    }


class FIFPDPMC:
    """
    FIFPDPMC as a callable baseline (no training required).

    Usage:
        fifpdpmc = FIFPDPMC(num_stages=10)
        preds = fifpdpmc.predict(sparse_syndromes, graph_structure)
    """

    def __init__(self, num_stages: Optional[int] = None):
        """
        Args:
            num_stages: Number of stages (syndrome rounds) to use. None = use all available.
        """
        self.num_stages = num_stages

    def predict(
        self,
        sparse_syndromes: List[List[Tuple]],
        graph_structure: Dict[str, Any],
    ) -> np.ndarray:
        """Run FIFPDPMC and return per-node predictions (0=fault-free, 1=faulty)."""
        return fifpdpmc_single_graph(
            sparse_syndromes, graph_structure, num_stages=self.num_stages
        )
