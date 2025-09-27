import torch
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data, Dataset
from graphs import GraphFactory
from logging_config import get_logger, init_default_logging
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import argparse
import gc
import atexit

# Initialize logging configuration
init_default_logging()
logger = get_logger(__name__)

# Global process pool management
_global_pool = None
_pool_size = None

def _cleanup_global_pool():
    """Clean up global process pool"""
    global _global_pool
    if _global_pool is not None:
        try:
            _global_pool.close()
            _global_pool.join()
            _global_pool = None
            logger.debug("Global process pool cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up global process pool: {e}")

def _get_or_create_global_pool(n_jobs: int):
    """Get or create global process pool"""
    global _global_pool, _pool_size
    
    if _global_pool is None or _pool_size != n_jobs:
        # Clean up old process pool
        _cleanup_global_pool()
        
        # Create new process pool
        _global_pool = Pool(processes=n_jobs)
        _pool_size = n_jobs
        logger.debug(f"Created new global process pool: {n_jobs} processes")
    
    return _global_pool

# Register cleanup on exit
atexit.register(_cleanup_global_pool)


class UnifiedDatasetGenerator:
    """
    Unified dataset generator - supports parallel generation and resource management
    """
    
    def __init__(self, graph_type: str, n: int, k: Optional[int] = None, 
                 fault_rate: Optional[float] = None, fault_count: Optional[int] = None,
                 intermittent_prob: float = 0.5, num_rounds: int = 10, seed: int = 42,
                 n_jobs: Optional[int] = None, use_global_pool: bool = True):
        """
        Initialize unified dataset generator
        
        Args:
            use_global_pool: Whether to use global process pool (recommended for multi-round training)
        """
        self.graph_type = graph_type
        self.n = n
        self.k = k
        self.fault_rate = fault_rate
        self.fault_count = fault_count
        self.intermittent_prob = intermittent_prob
        self.num_rounds = num_rounds
        self.rng = np.random.default_rng(seed)
        self.use_global_pool = use_global_pool
        
        # Set up parallel configuration
        if n_jobs is None:
            self.n_jobs = min(8, cpu_count())
        else:
            self.n_jobs = min(n_jobs, cpu_count())
        
        # Create a temporary graph to estimate size and configuration
        temp_graph = GraphFactory.create_graph(
            graph_type, n, k, fault_rate, fault_count, intermittent_prob, seed
        )
        
        self.num_vertices = temp_graph.num_vertices
        self.num_edges = len(temp_graph.edges)
        self.theoretical_diagnosability = temp_graph.theoretical_diagnosability
        
        # Intelligently determine parallel strategy
        estimated_nodes = self._estimate_node_count(graph_type, n, k)
        if estimated_nodes > 500:
            suggested_jobs = 2
            process_type = "large-scale graph"
        elif estimated_nodes > 100:
            suggested_jobs = min(4, self.n_jobs)
            process_type = "medium-scale graph"
        else:
            suggested_jobs = self.n_jobs
            process_type = "small-scale graph"
        
        self.n_jobs = suggested_jobs
        logger.info(f"Detected {process_type} (estimated {estimated_nodes} nodes), using {self.n_jobs} processes for parallel generation")
        
        # Log graph configuration information
        logger.info(f"Graph configuration: {graph_type} n={n}" + (f" k={k}" if k is not None else ""))
        logger.info(f"Graph scale: {self.num_vertices} nodes, {self.num_edges} edges")
        logger.info(f"Parallel configuration: {self.n_jobs} processes" + ("(using global process pool)" if use_global_pool else ""))

    def cleanup_resources(self):
        """Manually clean up resources"""
        if not self.use_global_pool:
            # If not using global process pool, no special cleanup needed
            pass
        
        # Force garbage collection
        gc.collect()
        logger.debug("Resource cleanup and garbage collection completed")

    def _get_process_pool(self):
        """Get process pool (global or local)"""
        if self.use_global_pool:
            return _get_or_create_global_pool(self.n_jobs)
        else:
            return Pool(processes=self.n_jobs)

    def _close_process_pool(self, pool):
        """Close process pool (only effective for local process pool)"""
        if not self.use_global_pool:
            pool.close()
            pool.join()

    def _estimate_node_count(self, graph_type: str, n: int, k: Optional[int]) -> int:
        """Estimate the number of nodes in the graph"""
        if graph_type == 'augmented_k_ary_n_cube':
            k = k or 3
            return k ** n
        elif graph_type == 'bc':
            return 2 ** n
        else:
            return 100  # Conservative estimate
    
    def _should_use_single_process(self) -> bool:
        """Determine if single process mode should be used"""
        # For large graphs (>500 nodes), force using single process
        return self.num_vertices > 500
    
    def generate_default_save_dir(self) -> str:
        """
        Generate default save directory name
        Format: {graph_type}_n{n}[_k{k}]_{fault_info}
        """
        # Base part: graph type and dimension
        dir_name = f"{self.graph_type}_n{self.n}"
        
        # If k parameter exists, add k information
        if self.k is not None:
            dir_name += f"_k{self.k}"
        
        # Fault information part
        if self.fault_rate is not None:
            dir_name += f"_rate{self.fault_rate:.2f}"
        elif self.fault_count is not None:
            dir_name += f"_count{self.fault_count}"
        else:
            # Use theoretical diagnosability
            dir_name += f"_diag{self.theoretical_diagnosability}"
        
        return f"datasets/{dir_name}"
    
    def generate_raw_dataset(self, num_graphs: int) -> Dict:
        """
        Parallel generation of raw PMC symptom dataset - sparse storage version
        """
        logger.info(f"Starting to generate {num_graphs} graphs of raw PMC symptom data (sparse storage)...")
        start_time = time.time()
        
        # Prepare parallel task parameters
        task_args = []
        for i in range(num_graphs):
            seed = self.rng.integers(0, 100000)
            task_args.append((
                self.graph_type, self.n, self.k, self.fault_rate, 
                self.fault_count, self.intermittent_prob, 
                self.num_rounds, seed, i
            ))
        
        # Select generation strategy based on graph scale
        if self._should_use_single_process():
            logger.info(f"Detected large graph ({self.num_vertices} nodes), using single process sequential generation to avoid memory issues...")
            results = self._generate_sequential(task_args, start_time, num_graphs)
        else:
            logger.info(f"Using {self.n_jobs} processes for parallel generation...")
            results = self._generate_parallel(task_args, start_time, num_graphs)
        
        if len(results) != num_graphs:
            logger.warning(f"Only {len(results)}/{num_graphs} graphs were successfully generated")
        
        # Sort results by graph index
        results.sort(key=lambda x: x['graph_idx'])
        
        # Organize data
        raw_data = {
            'graph_configs': [],
            'sparse_syndromes': [],
            'fault_states': [],
            'graph_structures': [],
            'metadata': {
                'graph_type': self.graph_type,
                'n': self.n,
                'k': self.k,
                'fault_rate': self.fault_rate,
                'fault_count': self.fault_count,
                'intermittent_prob': self.intermittent_prob,
                'num_rounds': self.num_rounds,
                'storage_format': 'sparse',
                'num_vertices': self.num_vertices,
                'num_edges': self.num_edges,
                'theoretical_diagnosability': self.theoretical_diagnosability,
                'generation_time': time.time() - start_time,
                'n_jobs': self.n_jobs if not self._should_use_single_process() else 1,
                'successful_graphs': len(results),
                'generation_mode': 'sequential' if self._should_use_single_process() else 'parallel'
            }
        }
        
        for result in results:
            raw_data['graph_configs'].append(result['config'])
            raw_data['sparse_syndromes'].append(result['sparse_syndromes'])
            raw_data['fault_states'].append(result['fault_states'])
            raw_data['graph_structures'].append(result['graph_structure'])
        
        # Calculate space savings and performance statistics
        dense_size = len(results) * self.num_rounds * self.num_vertices * self.num_vertices * 8
        sparse_size = len(results) * self.num_rounds * self.num_edges * 4 * 1
        space_saved = (dense_size - sparse_size) / dense_size * 100 if dense_size > 0 else 0
        
        generation_time = time.time() - start_time
        graphs_per_second = len(results) / generation_time if generation_time > 0 else 0
        
        logger.info(f"Generation completed:")
        logger.info(f"  Successfully generated: {len(results)}/{num_graphs} graphs")
        logger.info(f"  Total time: {generation_time:.1f} seconds")
        logger.info(f"  Generation speed: {graphs_per_second:.1f} graphs/second")
        logger.info(f"  Sparse storage space saved: {space_saved:.1f}% "
                   f"(from {dense_size/1e9:.2f}GB to {sparse_size/1e6:.1f}MB)")
        
        return raw_data
    
    def _generate_sequential(self, task_args: list, start_time: float, num_graphs: int) -> list:
        """Single process sequential generation (for large graphs)"""
        results = []
        
        for i, args in enumerate(task_args):
            try:
                # Generate single graph
                result = generate_single_graph_data(args)
                if result is not None:
                    results.append(result)
                
                # Progress report
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (num_graphs - i - 1) / rate if rate > 0 else 0
                logger.info(f"已生成 {i + 1}/{num_graphs} 个图 "
                          f"(速度: {rate:.2f} 图/秒, 预计剩余: {eta:.1f}秒)")
                
                # Active garbage collection, release memory
                if (i + 1) % 2 == 0:  # Garbage collection every 2 graphs
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Error generating graph {i}: {e}")
                continue
        
        return results
    
    def _generate_parallel(self, task_args: list, start_time: float, num_graphs: int) -> list:
        """Multi-process parallel generation (for small and medium-sized graphs) - improved resource management"""
        results = []
        pool = None
        
        try:
            if self.use_global_pool:
                pool = _get_or_create_global_pool(self.n_jobs)
                pool_context = None  # No context manager needed
            else:
                pool_context = Pool(processes=self.n_jobs)
                pool = pool_context.__enter__()
            
            # Use imap_unordered to improve efficiency
            completed = 0
            for result in pool.imap_unordered(generate_single_graph_data, task_args, chunksize=1):
                if result is not None:
                    results.append(result)
                
                completed += 1
                # Progress report
                if completed % max(1, min(10, num_graphs // 10)) == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (num_graphs - completed) / rate if rate > 0 else 0
                    logger.info(f"Generated {completed}/{num_graphs} graphs "
                              f"(speed: {rate:.1f} graphs/second, ETA: {eta:.1f} seconds)")
                    
                    # Check if any process is stuck
                    if completed > 0 and rate < 0.1 and self.num_vertices > 500:
                        logger.warning(f"Slow generation speed detected ({rate:.3f} graphs/second), possibly due to large graph size")
            
            # Only close local process pool
            if not self.use_global_pool and pool_context:
                pool_context.__exit__(None, None, None)
                
        except KeyboardInterrupt:
            logger.error("Data generation process interrupted by user")
            if not self.use_global_pool and pool_context:
                pool_context.__exit__(None, None, None)
            raise
        except Exception as e:
            logger.error(f"Error during parallel generation: {e}")
            if not self.use_global_pool and pool_context:
                pool_context.__exit__(None, None, None)
            raise
        
        return results
    
    def convert_to_gnn_format_parallel(self, raw_data: Dict) -> List[Data]:
        """
        Parallel conversion to GNN format - improved resource management
        """
        logger.info("Parallel conversion of sparse data to GNN format...")
        start_time = time.time()
        
        # Prepare conversion task parameters
        conversion_args = []
        for i in range(len(raw_data['sparse_syndromes'])):
            conversion_args.append((
                i,
                raw_data['sparse_syndromes'][i],
                raw_data['fault_states'][i],
                raw_data['graph_configs'][i],
                self.graph_type, self.n, self.k, self.fault_rate,
                self.fault_count, self.intermittent_prob
            ))
        
        # Parallel conversion
        pool = None
        try:
            if self.use_global_pool:
                pool = _get_or_create_global_pool(self.n_jobs)
                gnn_results = pool.map(convert_single_graph_to_gnn, conversion_args)
            else:
                with Pool(processes=self.n_jobs) as pool:
                    gnn_results = pool.map(convert_single_graph_to_gnn, conversion_args)
        except Exception as e:
            logger.error(f"GNN format conversion failed: {e}")
            raise
        
        # Sort by index
        gnn_results.sort(key=lambda x: x[0])
        gnn_dataset = [result[1] for result in gnn_results]
        
        conversion_time = time.time() - start_time
        logger.info(f"GNN format conversion completed: {len(gnn_dataset)} graphs "
                   f"(time: {conversion_time:.1f} seconds)")
        
        return gnn_dataset
    
    def convert_to_rnn_format_parallel(self, raw_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parallel conversion to RNN format - improved resource management
        """
        logger.info("Parallel conversion of sparse data to RNNIFDCom_PMC format (utilizing all rounds)...")
        start_time = time.time()
        
        # Prepare conversion task parameters - each round of each graph is a task
        conversion_args = []
        sample_idx = 0
        
        for graph_idx in range(len(raw_data['sparse_syndromes'])):
            sparse_syndromes = raw_data['sparse_syndromes'][graph_idx]
            fault_states = raw_data['fault_states'][graph_idx]
            
            # Create an RNN sample for each round of testing
            for round_idx, round_syndrome in enumerate(sparse_syndromes):
                conversion_args.append((
                    sample_idx,      # RNN sample index
                    graph_idx,       # Original graph index
                    round_idx,       # Round index
                    round_syndrome,  # Sparse symptoms of this round
                    fault_states     # Fault states (shared across all rounds)
                ))
                sample_idx += 1
        
        logger.info(f"Will generate {len(conversion_args)} RNN samples "
                   f"(from {len(raw_data['sparse_syndromes'])} graphs × {self.num_rounds} rounds)")
        
        # Parallel conversion
        try:
            if self.use_global_pool:
                pool = _get_or_create_global_pool(self.n_jobs)
                rnn_results = pool.map(convert_single_round_to_rnn, conversion_args)
            else:
                with Pool(processes=self.n_jobs) as pool:
                    rnn_results = pool.map(convert_single_round_to_rnn, conversion_args)
        except Exception as e:
            logger.error(f"RNN format conversion failed: {e}")
            raise
        
        # Sort by sample index and organize data
        rnn_results.sort(key=lambda x: x[0])
        X = np.array([result[1] for result in rnn_results], dtype=np.float32)
        y = np.array([result[2] for result in rnn_results], dtype=np.float32)
        
        conversion_time = time.time() - start_time
        logger.info(f"RNNIFDCom_PMC format conversion completed: X.shape={X.shape}, y.shape={y.shape} "
                   f"(time: {conversion_time:.1f} seconds)")
        logger.info(f"Data utilization: {X.shape[0]} RNN samples vs {len(raw_data['sparse_syndromes'])} GNN samples "
                   f"(ratio {X.shape[0]/len(raw_data['sparse_syndromes']):.1f}:1)")
        
        return X, y
    
    def generate_complete_dataset(self, num_graphs: int, save_dir: Optional[str] = None) -> Dict:
        """
        Parallel generation of complete dataset
        
        Args:
            num_graphs: Number of graphs to generate
            save_dir: Save directory, if None, generate automatically
        """
        # If no save directory is specified, generate automatically
        if save_dir is None:
            save_dir = self.generate_default_save_dir()
        
        logger.info("=" * 60)
        logger.info("Starting parallel generation of unified dataset")
        logger.info(f"Save directory: {save_dir}")
        logger.info("=" * 60)
        
        total_start_time = time.time()
        
        # 1. Parallel generation of raw data
        raw_data = self.generate_raw_dataset(num_graphs)
        
        # 2. Parallel conversion to GNN format
        gnn_data = self.convert_to_gnn_format_parallel(raw_data)
        
        # 3. Parallel conversion to RNNIFDCom_PMC format
        rnn_data = self.convert_to_rnn_format_parallel(raw_data)
        
        # 4. Save dataset
        self.save_dataset(raw_data, gnn_data, rnn_data, save_dir)
        
        total_time = time.time() - total_start_time
        logger.info("=" * 60)
        logger.info(f"Parallel unified dataset generation completed (total time: {total_time:.1f} seconds)")
        logger.info("=" * 60)
        
        return {
            'raw_data': raw_data,
            'gnn_data': gnn_data,
            'rnn_data': rnn_data,
            'metadata': raw_data['metadata']  # Add metadata key
        }
    
    def save_dataset(self, raw_data: Dict, gnn_data: List[Data], 
                     rnn_data: Tuple[np.ndarray, np.ndarray], 
                     save_dir: str) -> None:
        """
        Save dataset to file
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save raw data
        with open(os.path.join(save_dir, 'raw_data.pkl'), 'wb') as f:
            pickle.dump(raw_data, f)
        
        # Save GNN data
        torch.save(gnn_data, os.path.join(save_dir, 'gnn_data.pt'), _use_new_zipfile_serialization=False)
        
        # Save RNNIFDCom_PMC data
        rnn_X, rnn_y = rnn_data
        np.savez(os.path.join(save_dir, 'rnn_data.npz'), X=rnn_X, y=rnn_y)
        
        # Save metadata
        metadata = raw_data['metadata'].copy()
        metadata['num_graphs'] = len(raw_data['sparse_syndromes'])
        metadata['gnn_feature_dim'] = gnn_data[0].x.shape[1] if gnn_data else 0
        metadata['rnn_feature_dim'] = rnn_X.shape[1] if len(rnn_X) > 0 else 0
        
        # Add new statistics
        metadata['gnn_samples'] = len(gnn_data)
        metadata['rnn_samples'] = rnn_X.shape[0]
        metadata['rnn_to_gnn_ratio'] = rnn_X.shape[0] / len(gnn_data) if len(gnn_data) > 0 else 0
        
        with open(os.path.join(save_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Dataset saved to: {save_dir}")
        logger.info(f"File list: raw_data.pkl, gnn_data.pt, rnn_data.npz, metadata.pkl")
        logger.info(f"Statistics: GNN={len(gnn_data)} samples, RNN={rnn_X.shape[0]} samples")
        

def generate_single_graph_data(args):
    """
    Generate data for a single graph (for parallel processing)
    
    Args:
        args: Tuple containing generation parameters
        
    Returns:
        Dictionary containing data for a single graph
    """
    (graph_type, n, k, fault_rate, fault_count, intermittent_prob, 
     num_rounds, seed, graph_idx) = args
    
    try:
        # Create graph instance
        graph = GraphFactory.create_graph(
            graph_type, n, k, fault_rate, fault_count, intermittent_prob, seed
        )
        
        # Generate multiple rounds of sparse symptom data
        sparse_syndromes = []
        for round_idx in range(num_rounds):
            sparse_syndrome = graph.generate_syndrome_single_round_sparse()
            sparse_syndromes.append(sparse_syndrome)
        
        # Get fault node indices
        fault_indices = np.where(graph.fault_states == 1)[0].tolist()
        
        # Return all data for a single graph
        return {
            'graph_idx': graph_idx,
            'config': {
                'seed': seed,
                'actual_fault_count': len(fault_indices),
                'fault_indices': fault_indices
            },
            'sparse_syndromes': sparse_syndromes,
            'fault_states': graph.fault_states,
            'graph_structure': {
                'edges': list(graph.edges),
                'vertices': list(graph.vertices),
                'node_to_idx': graph.node_to_idx.copy()
            }
        }
    except Exception as e:
        logger.error(f"Error generating graph {graph_idx}: {e}")
        return None


def convert_single_graph_to_gnn(args):
    """
    Convert single graph to GNN format (for parallel processing)
    """
    (idx, sparse_syndromes, fault_states, config, 
     graph_type, n, k, fault_rate, fault_count, intermittent_prob) = args
    
    try:
        # Reconstruct graph object
        graph = GraphFactory.create_graph(
            graph_type, n, k, fault_rate, fault_count, 
            intermittent_prob, config['seed']
        )
        
        # Generate node features using sparse data
        x = graph.get_node_features_from_sparse_syndromes(sparse_syndromes)
        
        # Generate edge indices
        edge_index = graph.get_edge_index()
        
        # Fault state labels
        y = torch.tensor(fault_states, dtype=torch.long)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        
        return (idx, data)
    except Exception as e:
        logger.error(f"Error converting graph {idx} to GNN format: {e}")
        return (idx, None)


def convert_single_round_to_rnn(args):
    """
    Convert single round of test data to RNN format (for parallel processing)
    """
    sample_idx, graph_idx, round_idx, round_syndrome, fault_states = args
    
    try:
        # Build the syndrome vector for this round
        all_comparisons = []
        for u, v, test_u_to_v, test_v_to_u in round_syndrome:
            # Simulate MM model comparison results
            comparison_result = int(test_u_to_v != test_v_to_u)
            all_comparisons.append(comparison_result)
        
        # Convert according to paper requirements: 0 → -0.5, 1 → 0.5
        converted_syndrome = [(x - 0.5) for x in all_comparisons]
        
        return (sample_idx, converted_syndrome, fault_states.astype(np.float32))
    except Exception as e:
        logger.error(f"Error converting graph {graph_idx} round {round_idx} to RNN format: {e}")
        return (sample_idx, None, None)


class UnifiedDatasetLoader:
    """
    Unified dataset loader
    """
    
    @staticmethod
    def load_dataset(save_dir: str) -> Dict:
        """
        Load dataset, automatically detect storage format
        """
        logger.info(f"Loading dataset from {save_dir}...")
        
        # Load raw data
        with open(os.path.join(save_dir, 'raw_data.pkl'), 'rb') as f:
            raw_data = pickle.load(f)
        
        # Check storage format
        storage_format = raw_data['metadata'].get('storage_format', 'dense')
        logger.info(f"Detected storage format: {storage_format}")
        
        # Remaining loading logic remains unchanged
        gnn_data = torch.load(os.path.join(save_dir, 'gnn_data.pt'), weights_only=False)
        rnn_file = np.load(os.path.join(save_dir, 'rnn_data.npz'))
        rnn_data = (rnn_file['X'], rnn_file['y'])
        
        with open(os.path.join(save_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Ensure the returned dictionary contains the metadata key
        return {
            'raw_data': raw_data,
            'gnn_data': gnn_data,
            'rnn_data': rnn_data,
            'metadata': metadata  # Use data loaded from metadata.pkl
        }
    
    @staticmethod
    def split_dataset(dataset: Dict, train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, test_ratio: float = 0.15,
                     seed: int = 42) -> Dict:
        """
        Split dataset into training/validation/test sets
        Note: GNN and RNN have different split strategies
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Sum of ratios must be 1"
        
        rng = np.random.default_rng(seed)
        gnn_data = dataset['gnn_data']
        rnn_X, rnn_y = dataset['rnn_data']
        
        # GNN data split (by graph)
        num_graphs = len(gnn_data)
        graph_indices = rng.permutation(num_graphs)
        
        train_end = int(num_graphs * train_ratio)
        val_end = train_end + int(num_graphs * val_ratio)
        
        train_graph_indices = graph_indices[:train_end]
        val_graph_indices = graph_indices[train_end:val_end]
        test_graph_indices = graph_indices[val_end:]
        
        # Split GNN data
        gnn_train = [gnn_data[i] for i in train_graph_indices]
        gnn_val = [gnn_data[i] for i in val_graph_indices]
        gnn_test = [gnn_data[i] for i in test_graph_indices]
        
        # RNN data split (by sample, but maintain graph integrity)
        # Assume each graph has num_rounds samples
        metadata = dataset.get('metadata', {})
        num_rounds = metadata.get('num_rounds', 10)
        
        # Create corresponding indices for RNN data
        rnn_train_indices = []
        rnn_val_indices = []
        rnn_test_indices = []
        
        for graph_idx in train_graph_indices:
            # All rounds of this graph are assigned to the training set
            for round_idx in range(num_rounds):
                rnn_sample_idx = graph_idx * num_rounds + round_idx
                rnn_train_indices.append(rnn_sample_idx)
        
        for graph_idx in val_graph_indices:
            # All rounds of this graph are assigned to the validation set
            for round_idx in range(num_rounds):
                rnn_sample_idx = graph_idx * num_rounds + round_idx
                rnn_val_indices.append(rnn_sample_idx)
        
        for graph_idx in test_graph_indices:
            # All rounds of this graph are assigned to the test set
            for round_idx in range(num_rounds):
                rnn_sample_idx = graph_idx * num_rounds + round_idx
                rnn_test_indices.append(rnn_sample_idx)
        
        # Split RNN data
        rnn_train = (rnn_X[rnn_train_indices], rnn_y[rnn_train_indices])
        rnn_val = (rnn_X[rnn_val_indices], rnn_y[rnn_val_indices])
        rnn_test = (rnn_X[rnn_test_indices], rnn_y[rnn_test_indices])
        
        logger.info(f"Dataset split completed:")
        logger.info(f"  GNN - Training set: {len(train_graph_indices)} graphs")
        logger.info(f"  GNN - Validation set: {len(val_graph_indices)} graphs")
        logger.info(f"  GNN - Test set: {len(test_graph_indices)} graphs")
        logger.info(f"  RNN - Training set: {len(rnn_train_indices)} samples")
        logger.info(f"  RNN - Validation set: {len(rnn_val_indices)} samples")
        logger.info(f"  RNN - Test set: {len(rnn_test_indices)} samples")
        logger.info(f"  RNN/GNN sample ratio: {num_rounds}:1")
        
        return {
            'gnn': {
                'train': gnn_train,
                'val': gnn_val,
                'test': gnn_test
            },
            'rnn': {
                'train': rnn_train,
                'val': rnn_val,
                'test': rnn_test
            },
            'metadata': dataset['metadata']
        }


def generate_dataset_from_config(graph_type: str = 'bc', n: int = 8, k: Optional[int] = None,
                                fault_rate: Optional[float] = None, fault_count: Optional[int] = None,
                                num_graphs: int = 10, num_rounds: int = 10, 
                                intermittent_prob: float = 0.5, seed: int = 42,
                                n_jobs: Optional[int] = None, save_dir: Optional[str] = None) -> Dict:
    """
    Convenient function to generate dataset based on configuration
    
    Args:
        graph_type: Graph type
        n: Graph scale parameter
        k: k-ary cube parameter
        fault_rate: Fault node ratio (mutually exclusive with fault_count)
        fault_count: Fault node count (mutually exclusive with fault_rate)
        num_graphs: Number of graphs to generate
        num_rounds: Number of test rounds
        intermittent_prob: Intermittent fault probability
        seed: Random seed
        n_jobs: Number of parallel processes, None means using all CPU cores
        save_dir: Save directory, None means automatically generate
        
    Returns:
        Dictionary containing generated dataset
    """
    # Create parallel generator
    generator = UnifiedDatasetGenerator(
        graph_type=graph_type,
        n=n,
        k=k,
        fault_rate=fault_rate,
        fault_count=fault_count,
        intermittent_prob=intermittent_prob,
        num_rounds=num_rounds,
        seed=seed,
        n_jobs=n_jobs
    )
    
    # Parallel generation of dataset
    start_time = time.time()
    dataset = generator.generate_complete_dataset(
        num_graphs=num_graphs,
        save_dir=save_dir
    )
    total_time = time.time() - start_time
    
    logger.info(f"Dataset generation completed!")
    logger.info(f"Total time: {total_time:.1f} seconds")
    logger.info(f"Average speed: {num_graphs/total_time:.1f} graphs/second")
    
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel generation of unified dataset')
    parser.add_argument('--graph_type', type=str, default='bc', help='Graph type')
    parser.add_argument('--n', type=int, default=8, help='Graph scale parameter')
    parser.add_argument('--k', type=int, default=None, help='k-ary cube parameter')
    parser.add_argument('--fault_rate', type=float, default=None, help='Fault node ratio')
    parser.add_argument('--fault_count', type=int, default=None, help='Fault node count')
    parser.add_argument('--num_graphs', type=int, default=10, help='Number of graphs to generate')
    parser.add_argument('--num_rounds', type=int, default=10, help='Number of test rounds')
    parser.add_argument('--intermittent_prob', type=float, default=0.5, help='Intermittent fault probability')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_jobs', type=int, default=None, help='Number of parallel processes')
    parser.add_argument('--save_dir', type=str, default=None, help='Save directory (automatically generated if not specified)')
    
    args = parser.parse_args()
    
    # Call convenient function to generate dataset
    dataset = generate_dataset_from_config(
        graph_type=args.graph_type,
        n=args.n,
        k=args.k,
        fault_rate=args.fault_rate,
        fault_count=args.fault_count,
        num_graphs=args.num_graphs,
        num_rounds=args.num_rounds,
        intermittent_prob=args.intermittent_prob,
        seed=args.seed,
        n_jobs=args.n_jobs,
        save_dir=args.save_dir
    ) 