#!/usr/bin/env python3
"""
RQ1: Fault diagnosis performance comparison
Randomly generate fault node counts from 1 to theoretical diagnosability-1
"""

import torch
import numpy as np
import argparse
import os
import time
from datetime import datetime
import json

# Import project modules
from graphs import GraphFactory
from run_comparison import run_single_experiment
from logging_config import get_logger, init_default_logging

# Initialize logging
init_default_logging()
logger = get_logger(__name__)


class SimpleRQComparison:
    """RQ1 comparison experiment class"""
    
    def __init__(self, graph_type: str, n: int, k: int = None, 
                 intermittent_prob: float = 0.5, num_rounds: int = 10,
                 num_graphs: int = 100, num_runs: int = 3, seed: int = 42):
        """
        Initialize simplified experiment
        
        Args:
            graph_type: Graph type ('bc', 'star', 'alternating_group', etc.)
            n: Graph scale parameter
            k: k-ary cube parameter (if needed)
            intermittent_prob: Intermittent fault probability
            num_rounds: Number of test rounds
            num_graphs: Number of generated graphs
            num_runs: Number of runs
            seed: Random seed
        """
        self.graph_type = graph_type
        self.n = n
        self.k = k
        self.intermittent_prob = intermittent_prob
        self.num_rounds = num_rounds
        self.num_graphs = num_graphs
        self.num_runs = num_runs
        self.seed = seed
        
        # Get basic graph information
        sample_graph = GraphFactory.create_graph(
            graph_type, n, k, None, None, intermittent_prob, seed
        )
        self.num_vertices = sample_graph.num_vertices
        self.theoretical_diagnosability = sample_graph.theoretical_diagnosability
        
        # Calculate fault node count range (1 to theoretical diagnosability-1)
        self.max_test_fault_count = max(1, self.theoretical_diagnosability - 1)
        self.min_test_fault_count = 1
        
        logger.info(f"Initialize simplified experiment:")
        logger.info(f"  Graph type: {graph_type}")
        logger.info(f"  Graph scale: n={n}" + (f", k={k}" if k else ""))
        logger.info(f"  Number of vertices: {self.num_vertices}")
        logger.info(f"  Theoretical diagnosability: {self.theoretical_diagnosability}")
        logger.info(f"  Fault node count range: {self.min_test_fault_count} to {self.max_test_fault_count}")
        
        # Validate fault node count reasonableness
        if self.max_test_fault_count >= self.num_vertices:
            raise ValueError(f"Maximum fault node count({self.max_test_fault_count}) cannot be greater than or equal to total node count({self.num_vertices})")
        if self.max_test_fault_count < self.min_test_fault_count:
            raise ValueError(f"Theoretical diagnosability too small({self.theoretical_diagnosability}), cannot generate valid fault node count range")
    
    def create_experiment_config(self, fault_count: int) -> dict:
        """Create experiment configuration"""
        config = {
            'graph_type': self.graph_type,
            'n': self.n,
            'k': self.k,
            'fault_count': fault_count,
            'fault_rate': None  # Use fault_count instead of fault_rate
        }
        return config
    
    def create_args_for_experiment(self, base_args) -> argparse.Namespace:
        """Create parameter object for experiment"""
        args = argparse.Namespace()
        
        # Copy all attributes from base_args
        for key, value in vars(base_args).items():
            setattr(args, key, value)
        
        # Set experiment-specific parameters
        args.intermittent_prob = self.intermittent_prob
        args.num_rounds = self.num_rounds
        args.num_graphs = self.num_graphs
        args.seed = self.seed
        
        return args
    
    def run_experiment(self, base_args, output_dir: str = None) -> dict:
        """
        Run simplified comparison experiment
        
        Args:
            base_args: Base experiment parameters
            output_dir: Output directory
            
        Returns:
            Experiment results dictionary
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            graph_desc = f"{self.graph_type}_n{self.n}"
            if self.k:
                graph_desc += f"_k{self.k}"
            output_dir = f"results/simple_rq/{graph_desc}_{timestamp}_random_faults"
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Simplified experiment results will be saved to: {output_dir}")
        
        # Prepare result storage
        experiment_results = {
            'metadata': {
                'graph_type': self.graph_type,
                'n': self.n,
                'k': self.k,
                'num_vertices': self.num_vertices,
                'theoretical_diagnosability': self.theoretical_diagnosability,
                'min_test_fault_count': self.min_test_fault_count,
                'max_test_fault_count': self.max_test_fault_count,
                'intermittent_prob': self.intermittent_prob,
                'num_rounds': self.num_rounds,
                'num_graphs': self.num_graphs,
                'num_runs': self.num_runs,
                'seed': self.seed,
                'output_dir': output_dir,
                'mode': 'random_fault_count'  # Identify random fault node count mode
            },
            'runs': [],
            'summary': {}
        }
        
        total_start_time = time.time()
        
        logger.info("=" * 60)
        logger.info(f"Starting simplified experiment: random fault node count range {self.min_test_fault_count}-{self.max_test_fault_count}")
        logger.info("=" * 60)
        
        # run multiple experiments
        successful_runs = 0
        
        for run_idx in range(self.num_runs):
            logger.info(f"运行 {run_idx+1}/{self.num_runs}...")
            
            try:
                # use different random seeds for each run
                run_args = self.create_args_for_experiment(base_args)
                # use more random seed generation strategy
                current_time_seed = int(time.time() * 1000) % 100000
                run_args.seed = self.seed + run_idx * 1000 + current_time_seed
                
                # randomly generate fault node count (in valid range)
                # use independent random generator to ensure true randomness
                rng = np.random.default_rng(run_args.seed + run_idx * 123)
                random_fault_count = rng.integers(self.min_test_fault_count, self.max_test_fault_count + 1)
                
                logger.info(f"  randomly select fault node count: {random_fault_count} (range: {self.min_test_fault_count}-{self.max_test_fault_count})")
                logger.info(f"  use random seed: {run_args.seed}")
                
                # force regenerate dataset, ensure each time use new fault node count
                run_args.force_regenerate = True
                
                # create configuration and run single experiment
                config = self.create_experiment_config(random_fault_count)
                result = run_single_experiment(config, run_args, output_dir)
                
                if 'error' not in result and result['gnn_results'] is not None:
                    # add fault node count information to result
                    result['fault_count'] = random_fault_count
                    experiment_results['runs'].append(result)
                    successful_runs += 1
                    
                    gnn_f1 = result['gnn_results']['f1_score']
                    rnn_f1 = result['rnn_results']['f1_score']
                    gnn_acc = result['gnn_results']['accuracy']
                    rnn_acc = result['rnn_results']['accuracy']
                    gnn_fnr = result['gnn_results']['false_negnnive_rate']
                    gnn_fpr = result['gnn_results']['false_positive_rate']
                    rnn_fnr = result['rnn_results']['false_negnnive_rate']
                    rnn_fpr = result['rnn_results']['false_positive_rate']
                    
                    logger.info(f"  GNN - F1: {gnn_f1:.4f}, Acc: {gnn_acc:.4f}, FNR: {gnn_fnr:.4f}, FPR: {gnn_fpr:.4f}")
                    logger.info(f"  RNN - F1: {rnn_f1:.4f}, Acc: {rnn_acc:.4f}, FNR: {rnn_fnr:.4f}, FPR: {rnn_fpr:.4f}")
                else:
                    logger.warning(f"  run {run_idx+1} failed")
                    
            except Exception as e:
                logger.warning(f"  run {run_idx+1} exception: {e}")
        
        total_time = time.time() - total_start_time
        
        # calculate summary results
        if successful_runs > 0:
            experiment_results['summary'] = self._calculate_summary(experiment_results['runs'])
            experiment_results['metadata']['successful_runs'] = successful_runs
            experiment_results['metadata']['total_experiment_time'] = total_time
            
            # save results
            self._save_results(experiment_results, output_dir)
            
            # output summary information
            summary = experiment_results['summary']
            logger.info("=" * 60)
            logger.info(f"Experiment completed! Successful runs: {successful_runs}/{self.num_runs}")
            logger.info(f"Total time: {total_time:.1f} seconds")
            logger.info(f"Actual fault node count range: {summary['min_fault_count']}-{summary['max_fault_count']}")
            logger.info(f"Average fault node count: {summary['avg_fault_count']:.1f}")
            logger.info(f"Fault node count distribution: {summary['fault_counts']}")
            logger.info(f"GNN average F1: {summary['avg_gnn_f1']:.4f} ± {summary['std_gnn_f1']:.4f}")
            logger.info(f"RNN average F1: {summary['avg_rnn_f1']:.4f} ± {summary['std_rnn_f1']:.4f}")
            logger.info(f"GNN average accuracy: {summary['avg_gnn_acc']:.4f} ± {summary['std_gnn_acc']:.4f}")
            logger.info(f"RNN average accuracy: {summary['avg_rnn_acc']:.4f} ± {summary['std_rnn_acc']:.4f}")
            logger.info(f"GNN average false negnnive rate: {summary['avg_gnn_fnr']:.4f} ± {summary['std_gnn_fnr']:.4f}")
            logger.info(f"RNN average false negnnive rate: {summary['avg_rnn_fnr']:.4f} ± {summary['std_rnn_fnr']:.4f}")
            logger.info(f"GNN average false positive rate: {summary['avg_gnn_fpr']:.4f} ± {summary['std_gnn_fpr']:.4f}")
            logger.info(f"RNN average false positive rate: {summary['avg_rnn_fpr']:.4f} ± {summary['std_rnn_fpr']:.4f}")
            
            if summary['avg_gnn_f1'] > summary['avg_rnn_f1']:
                logger.info("Conclusion: GNN performs better")
            elif summary['avg_rnn_f1'] > summary['avg_gnn_f1']:
                logger.info("Conclusion: RNN performs better")
            else:
                logger.info("Conclusion: two models perform equally")
            logger.info("=" * 60)
        else:
            logger.error("All runs failed!")
        
        return experiment_results
    
    def _calculate_summary(self, runs: list) -> dict:
        """Calculate summary statistics"""
        if not runs:
            return {}
        
        # extract all metrics
        gnn_f1_scores = [r['gnn_results']['f1_score'] for r in runs]
        gnn_accuracies = [r['gnn_results']['accuracy'] for r in runs]
        gnn_train_times = [r['gnn_results']['train_time'] for r in runs]
        gnn_fnrs = [r['gnn_results']['false_negnnive_rate'] for r in runs]
        gnn_fprs = [r['gnn_results']['false_positive_rate'] for r in runs]
        
        rnn_f1_scores = [r['rnn_results']['f1_score'] for r in runs]
        rnn_accuracies = [r['rnn_results']['accuracy'] for r in runs]
        rnn_train_times = [r['rnn_results']['train_time'] for r in runs]
        rnn_fnrs = [r['rnn_results']['false_negnnive_rate'] for r in runs]
        rnn_fprs = [r['rnn_results']['false_positive_rate'] for r in runs]
        
        # extract fault node count information
        fault_counts = [r.get('fault_count', 0) for r in runs]
        
        summary = {
            'total_runs': len(runs),
            'avg_gnn_f1': np.mean(gnn_f1_scores),
            'std_gnn_f1': np.std(gnn_f1_scores),
            'avg_gnn_acc': np.mean(gnn_accuracies),
            'std_gnn_acc': np.std(gnn_accuracies),
            'avg_gnn_time': np.mean(gnn_train_times),
            'std_gnn_time': np.std(gnn_train_times),
            'avg_gnn_fnr': np.mean(gnn_fnrs),
            'std_gnn_fnr': np.std(gnn_fnrs),
            'avg_gnn_fpr': np.mean(gnn_fprs),
            'std_gnn_fpr': np.std(gnn_fprs),
            'avg_rnn_f1': np.mean(rnn_f1_scores),
            'std_rnn_f1': np.std(rnn_f1_scores),
            'avg_rnn_acc': np.mean(rnn_accuracies),
            'std_rnn_acc': np.std(rnn_accuracies),
            'avg_rnn_time': np.mean(rnn_train_times),
            'std_rnn_time': np.std(rnn_train_times),
            'avg_rnn_fnr': np.mean(rnn_fnrs),
            'std_rnn_fnr': np.std(rnn_fnrs),
            'avg_rnn_fpr': np.mean(rnn_fprs),
            'std_rnn_fpr': np.std(rnn_fprs),
            'best_gnn_f1': max(gnn_f1_scores),
            'best_rnn_f1': max(rnn_f1_scores),
            'worst_gnn_f1': min(gnn_f1_scores),
            'worst_rnn_f1': min(rnn_f1_scores),
            'gnn_wins': sum(1 for i in range(len(gnn_f1_scores)) if gnn_f1_scores[i] > rnn_f1_scores[i]),
            'fault_counts': fault_counts,
            'avg_fault_count': np.mean(fault_counts),
            'min_fault_count': min(fault_counts),
            'max_fault_count': max(fault_counts)
        }
        
        return summary
    
    def _save_results(self, experiment_results: dict, output_dir: str):
        """Save experiment results"""
        # save JSON format results
        json_file = os.path.join(output_dir, 'results.json')
        
        # prepare JSON serializable data
        json_data = {
            'metadata': experiment_results['metadata'],
            'summary': experiment_results['summary'],
            'runs': []
        }
        
        # convert run results to JSON serializable format
        for run in experiment_results['runs']:
            json_run = {
                'experiment_name': run.get('experiment_name', ''),
                'config': run.get('config', {}),
                'fault_count': run.get('fault_count', 0),
                'gnn_results': run.get('gnn_results', {}),
                'rnn_results': run.get('rnn_results', {})
            }
            json_data['runs'].append(json_run)
        
        # custom serializer for NumPy types
        def numpy_serializer(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):  # handle 0-dimensional NumPy arrays
                return obj.item()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=numpy_serializer)
        
        # save text report
        report_file = os.path.join(output_dir, 'report.txt')
        self._save_text_report(experiment_results, report_file)
        
        logger.info(f"Results saved:")
        logger.info(f"  JSON results: {json_file}")
        logger.info(f"  Text report: {report_file}")
    
    def _save_text_report(self, experiment_results: dict, report_file: str):
        """Save text format report"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("RQ1 experiment report\n")
            f.write("=" * 40 + "\n\n")
            
            # metadata
            metadata = experiment_results['metadata']
            f.write("Experiment configuration:\n")
            f.write(f"  Graph type: {metadata['graph_type']}\n")
            f.write(f"  Graph scale: n={metadata['n']}")
            if metadata['k']:
                f.write(f", k={metadata['k']}")
            f.write(f"\n  Number of vertices: {metadata['num_vertices']}\n")
            f.write(f"  Theoretical diagnosability: {metadata['theoretical_diagnosability']}\n")
            f.write(f"  Fault node count range: {metadata['min_test_fault_count']}-{metadata['max_test_fault_count']}\n")
            f.write(f"  Number of runs: {metadata['num_runs']}\n")
            f.write(f"  Successful runs: {metadata.get('successful_runs', 0)}\n\n")
            
            # summary results
            if 'summary' in experiment_results and experiment_results['summary']:
                summary = experiment_results['summary']
                f.write("Experiment results summary:\n")
                f.write(f"  Actual fault node count range: {summary['min_fault_count']}-{summary['max_fault_count']}\n")
                f.write(f"  Average fault node count: {summary['avg_fault_count']:.1f}\n\n")
                
                f.write("GNN model results:\n")
                f.write(f"  Average F1: {summary['avg_gnn_f1']:.4f} ± {summary['std_gnn_f1']:.4f}\n")
                f.write(f"  Average accuracy: {summary['avg_gnn_acc']:.4f} ± {summary['std_gnn_acc']:.4f}\n")
                f.write(f"  Average false negnnive rate: {summary['avg_gnn_fnr']:.4f} ± {summary['std_gnn_fnr']:.4f}\n")
                f.write(f"  Average false positive rate: {summary['avg_gnn_fpr']:.4f} ± {summary['std_gnn_fpr']:.4f}\n\n")
                
                f.write("RNN model results:\n")
                f.write(f"  Average F1: {summary['avg_rnn_f1']:.4f} ± {summary['std_rnn_f1']:.4f}\n")
                f.write(f"  Average accuracy: {summary['avg_rnn_acc']:.4f} ± {summary['std_rnn_acc']:.4f}\n")
                f.write(f"  Average false negnnive rate: {summary['avg_rnn_fnr']:.4f} ± {summary['std_rnn_fnr']:.4f}\n")
                f.write(f"  Average false positive rate: {summary['avg_rnn_fpr']:.4f} ± {summary['std_rnn_fpr']:.4f}\n\n")
                
                f.write(f"Performance comparison:\n")
                f.write(f"  GNN wins: {summary['gnn_wins']}/{summary['total_runs']}\n")
                
                if summary['avg_gnn_f1'] > summary['avg_rnn_f1']:
                    f.write(f"  Conclusion: GNN performs better\n")
                elif summary['avg_rnn_f1'] > summary['avg_gnn_f1']:
                    f.write(f"  Conclusion: RNN performs better\n")
                else:
                    f.write(f"  Conclusion: two models perform equally\n")
                
                # detailed run results
                f.write(f"\nDetailed run results:\n")
                f.write(f"{'Run':<4} {'Fault count':<6} {'GNN_F1':<8} {'RNN_F1':<8} {'GNN_FNR':<9} {'GNN_FPR':<9} {'RNN_FNR':<9} {'RNN_FPR':<9}\n")
                f.write("-" * 70 + "\n")
                
                for i, run in enumerate(experiment_results['runs'], 1):
                    fault_count = run.get('fault_count', 0)
                    gnn_f1 = run['gnn_results']['f1_score']
                    rnn_f1 = run['rnn_results']['f1_score']
                    gnn_fnr = run['gnn_results']['false_negnnive_rate']
                    gnn_fpr = run['gnn_results']['false_positive_rate']
                    rnn_fnr = run['rnn_results']['false_negnnive_rate']
                    rnn_fpr = run['rnn_results']['false_positive_rate']
                    
                    f.write(f"{i:<4} {fault_count:<6} {gnn_f1:<8.4f} {rnn_f1:<8.4f} "
                           f"{gnn_fnr:<9.4f} {gnn_fpr:<9.4f} {rnn_fnr:<9.4f} {rnn_fpr:<9.4f}\n")
            
            f.write(f"\nTotal experiment time: {metadata.get('total_experiment_time', 0):.1f} seconds\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='RQ1: GNN vs RNN performance comparison')
    
    # graph configuration parameters
    parser.add_argument('--graph_type', type=str, default='bc', 
                       help='Graph type (bc, star, alternating_group, etc.)')
    parser.add_argument('--n', type=int, default=5, help='Graph scale parameter')
    parser.add_argument('--k', type=int, default=None, help='k-ary cube parameter')
    
    # experiment parameters
    parser.add_argument('--intermittent_prob', type=float, default=0.5, help='Intermittent fault probability')
    parser.add_argument('--num_rounds', type=int, default=5, help='Test rounds')
    parser.add_argument('--num_graphs', type=int, default=50, help='Number of graphs')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_jobs', type=int, default=None, help='Number of parallel processes')
    
    # GNN parameters
    parser.add_argument('--gnn_hidden_dim', type=int, default=64, help='GNN hidden layer dimension')
    parser.add_argument('--gnn_num_layers', type=int, default=2, help='GNN number of layers')
    parser.add_argument('--gnn_heads', type=int, default=8, help='GNN number of attention heads')
    parser.add_argument('--gnn_batch_size', type=int, default=16, help='GNN batch size')
    
    # RNN parameters
    parser.add_argument('--rnn_hidden_dims', type=int, nargs='+', default=[64, 32], 
                       help='RNN hidden layer dimension')
    
    # training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    
    # output parameters
    parser.add_argument('--dataset_base_dir', type=str, default='datasets', help='Dataset base directory')
    parser.add_argument('--force_regenerate', action='store_true', help='Force regenerate dataset')
    parser.add_argument('--random_fault_mode', action='store_true',
                        help='Enable random fault node count mode (randomly select fault node count in range 1 to theoretical diagnosability-1)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("Starting RQ1 experiment")
    logger.info("=" * 60)
    
    try:
        # create simplified experiment object
        comparison = SimpleRQComparison(
            graph_type=args.graph_type,
            n=args.n,
            k=args.k,
            intermittent_prob=args.intermittent_prob,
            num_rounds=args.num_rounds,
            num_graphs=args.num_graphs,
            num_runs=args.num_runs,
            seed=args.seed
        )
        
        # run experiment
        results = comparison.run_experiment(args, args.output_dir)
        
        if results['metadata'].get('successful_runs', 0) > 0:
            logger.info("RQ1 experiment completed successfully!")
        else:
            logger.error("RQ1 experiment failed!")
    
    except Exception as e:
        logger.error(f"RQ1 experiment execution failed: {e}")
        raise


if __name__ == "__main__":
    main() 