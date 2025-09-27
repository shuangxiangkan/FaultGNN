#!/usr/bin/env python3
"""
Fault Ratio Comparison Experiment
=================================

This script is used to compare GNN and RNNIFDCOM model performance under different fault node ratios.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple
import seaborn as sns

# Import project modules
from graphs import GraphFactory
from run_comparison import run_single_experiment, get_or_generate_dataset
from logging_config import get_logger, init_default_logging

# Initialize logging
init_default_logging()
logger = get_logger(__name__)


class FaultRatioComparison:
    """Fault ratio comparison experiment class"""
    
    def __init__(self, graph_type: str, n: int, k: int = None, 
                 max_fault_ratio: float = 0.3, ratio_step: float = 0.05,
                 intermittent_prob: float = 0.5, num_rounds: int = 10,
                 num_graphs: int = 500, num_runs: int = 10, seed: int = 42):
        """
        Initialize fault ratio comparison experiment
        
        Args:
            graph_type: Graph type ('bc', 'star', 'alternating_group', etc.)
            n: Graph scale parameter
            k: k-ary cube parameter (if needed)
            max_fault_ratio: Maximum fault node ratio (0.0-1.0)
            ratio_step: Ratio step size, default 0.05 (5%)
            intermittent_prob: Intermittent fault probability
            num_rounds: Number of test rounds
            num_graphs: Number of graphs generated for each ratio
            num_runs: Number of runs for each fault situation
            seed: Random seed
        """
        self.graph_type = graph_type
        self.n = n
        self.k = k
        self.max_fault_ratio = max_fault_ratio
        self.ratio_step = ratio_step
        self.intermittent_prob = intermittent_prob
        self.num_rounds = num_rounds
        self.num_graphs = num_graphs
        self.num_runs = num_runs
        self.seed = seed
        
        # Get basic information of the graph
        sample_graph = GraphFactory.create_graph(
            graph_type, n, k, None, None, intermittent_prob, seed
        )
        self.num_vertices = sample_graph.num_vertices
        self.theoretical_diagnosability = sample_graph.theoretical_diagnosability
        
        logger.info(f"Initialize fault ratio comparison experiment:")
        logger.info(f"  Graph type: {graph_type}")
        logger.info(f"  Graph scale: n={n}" + (f", k={k}" if k else ""))
        logger.info(f"  Node count: {self.num_vertices}")
        logger.info(f"  Theoretical diagnosability: {self.theoretical_diagnosability}")
        logger.info(f"  Maximum fault ratio: {max_fault_ratio*100:.1f}%")
        logger.info(f"  Ratio step: {ratio_step*100:.1f}%")
        
        # Verify parameter validity
        if max_fault_ratio <= 0 or max_fault_ratio > 1:
            raise ValueError(f"Maximum fault ratio must be in (0,1] range, current: {max_fault_ratio}")
        if ratio_step <= 0 or ratio_step > 1:
            raise ValueError(f"Ratio step must be in (0,1] range, current: {ratio_step}")
        
        max_fault_count = int(self.num_vertices * max_fault_ratio)
        if max_fault_count >= self.num_vertices:
            logger.warning(f"Maximum fault node count ({max_fault_count}) is close to node total ({self.num_vertices})")
    
    def generate_fault_ratio_configs(self) -> List[Dict]:
        """
        Generate fault ratio experiment configuration list
        Start from ratio_step, generate a configuration every ratio_step until max_fault_ratio
        
        Returns:
            Configuration list
        """
        configs = []
        
        # Generate ratio sequence
        fault_ratios = []
        current_ratio = self.ratio_step
        while current_ratio <= self.max_fault_ratio + 1e-10:  # Add small tolerance to avoid floating point precision problem
            fault_ratios.append(current_ratio)
            current_ratio += self.ratio_step
        
        # Ensure maximum ratio is included
        if abs(fault_ratios[-1] - self.max_fault_ratio) > 1e-10:
            fault_ratios.append(self.max_fault_ratio)
        
        # Convert to fault node count and generate configuration
        fault_counts = []
        for ratio in fault_ratios:
            fault_count = max(1, int(self.num_vertices * ratio))
            if fault_count not in fault_counts and fault_count < self.num_vertices:
                fault_counts.append(fault_count)
        
        # Generate configuration
        for fault_count in fault_counts:
            actual_ratio = fault_count / self.num_vertices
            config = {
                'graph_type': self.graph_type,
                'n': self.n,
                'k': self.k,
                'fault_count': fault_count,
                'fault_ratio': actual_ratio,
                'fault_rate': None  # Use fault_count instead of fault_rate
            }
            configs.append(config)
        
        logger.info(f"Generated {len(configs)} fault ratio configurations:")
        for config in configs:
            ratio_percent = config['fault_ratio'] * 100
            logger.info(f"  {config['fault_count']} fault nodes ({ratio_percent:.1f}%)")
        
        return configs
    
    def create_args_for_fault_ratio_experiment(self, base_args) -> argparse.Namespace:
        """
        Create parameter object for fault ratio experiment
        
        Args:
            base_args: Base parameters
            
        Returns:
            argparse.Namespace object
        """
        # Create a new Namespace object, containing all required parameters
        args = argparse.Namespace()
        
        # Copy all attributes of base_args
        for key, value in vars(base_args).items():
            setattr(args, key, value)
        
        # Set experiment-specific parameters
        args.intermittent_prob = self.intermittent_prob
        args.num_rounds = self.num_rounds
        args.num_graphs = self.num_graphs
        args.seed = self.seed
        
        return args
    
    def run_fault_ratio_experiments(self, base_args, output_dir: str = None) -> Dict:
        """
        Run all fault ratio comparison experiments
        
        Args:
            base_args: Base experiment parameters
            output_dir: Output directory
            
        Returns:
            Dictionary containing all experiment results
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            graph_desc = f"{self.graph_type}_n{self.n}"
            if self.k:
                graph_desc += f"_k{self.k}"
            max_ratio_percent = int(self.max_fault_ratio * 100)
            output_dir = f"results/fault_ratio_comparison/{graph_desc}_max{max_ratio_percent}pct_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Fault ratio experiment results will be saved to: {output_dir}")
        
        # Generate experiment configuration
        configs = self.generate_fault_ratio_configs()
        
        # Prepare experiment parameters
        args = self.create_args_for_fault_ratio_experiment(base_args)
        
        # Store all experiment results
        all_results = {
            'metadata': {
                'graph_type': self.graph_type,
                'n': self.n,
                'k': self.k,
                'num_vertices': self.num_vertices,
                'theoretical_diagnosability': self.theoretical_diagnosability,
                'max_fault_ratio': self.max_fault_ratio,
                'ratio_step': self.ratio_step,
                'intermittent_prob': self.intermittent_prob,
                'num_rounds': self.num_rounds,
                'num_graphs': self.num_graphs,
                'num_runs': self.num_runs,
                'seed': self.seed,
                'experiment_count': len(configs),
                'output_dir': output_dir
            },
            'results': [],
            'summary': {}
        }
        
        total_start_time = time.time()
        
        logger.info("=" * 80)
        logger.info(f"Start fault ratio comparison experiment: {len(configs)} configurations")
        logger.info("=" * 80)
        
        # Run experiments one by one (multiple times for each fault situation)
        for i, config in enumerate(configs, 1):
            fault_count = config['fault_count']
            fault_ratio = config['fault_ratio']
            logger.info(f"\n[{i}/{len(configs)}] Testing fault ratio: {fault_ratio*100:.1f}% ({fault_count} nodes)")
            
            exp_start_time = time.time()
            
            # Store results of multiple runs
            run_results = []
            successful_runs = 0
            
            # Run multiple times and take average
            for run_idx in range(self.num_runs):
                logger.info(f"    Running {run_idx+1}/{self.num_runs}...")
                
                try:
                    # Use different random seeds for each run
                    run_args = self.create_args_for_fault_ratio_experiment(base_args)
                    run_args.seed = self.seed + run_idx * 1000 + fault_count
                    
                    # Run single experiment
                    result = run_single_experiment(config, run_args, output_dir)
                    
                    if 'error' not in result and result['gnn_results'] is not None:
                        run_results.append(result)
                        successful_runs += 1
                        
                        gnn_f1 = result['gnn_results']['f1_score']
                        rnn_f1 = result['rnn_results']['f1_score']
                        logger.info(f"      GNN F1: {gnn_f1:.4f}, RNNIFDCOM F1: {rnn_f1:.4f}")
                    else:
                        logger.warning(f"      Run {run_idx+1} failed")
                        
                except Exception as e:
                    logger.warning(f"      Run {run_idx+1} failed: {e}")
            
            exp_time = time.time() - exp_start_time
            
            if successful_runs > 0:
                # Calculate average results and standard deviation
                avg_result = self._calculate_average_results(run_results, fault_count, fault_ratio, i)
                all_results['results'].append(avg_result)
                
                logger.info(f"  Completed {successful_runs}/{self.num_runs} runs (time: {exp_time:.1f}s)")
                logger.info(f"  Average GNN F1: {avg_result['gnn_results']['f1_score']:.4f} ± {avg_result['gnn_results']['f1_std']:.4f}")
                logger.info(f"  Average RNNIFDCOM F1: {avg_result['rnn_results']['f1_score']:.4f} ± {avg_result['rnn_results']['f1_std']:.4f}")
            else:
                logger.error(f"  All runs failed")
                error_result = {
                    'experiment_name': f"{self.graph_type}_n{self.n}_ratio{fault_ratio:.2f}",
                    'config': config,
                    'fault_count': fault_count,
                    'fault_ratio': fault_ratio,
                    'experiment_index': i,
                    'error': f"All {self.num_runs} runs failed",
                    'gnn_results': None,
                    'rnn_results': None,
                    'successful_runs': 0,
                    'total_runs': self.num_runs
                }
                all_results['results'].append(error_result)
        
        total_time = time.time() - total_start_time
        
        # Generate summary statistics
        all_results['summary'] = self._generate_fault_ratio_summary(all_results['results'])
        all_results['metadata']['total_experiment_time'] = total_time
        
        logger.info("=" * 80)
        logger.info(f"All fault ratio experiments completed! Total time: {total_time:.1f} seconds")
        logger.info("=" * 80)
        
        # Save results and generate visualizations
        self._save_fault_ratio_results(all_results, output_dir)
        self._generate_fault_ratio_visualizations(all_results, output_dir)
        
        return all_results
    
    def _calculate_average_results(self, run_results: List[Dict], fault_count: int, fault_ratio: float, experiment_index: int) -> Dict:
        """Calculate average results of multiple runs"""
        if not run_results:
            return None
        
        # Collect all results
        gnn_f1_scores = [r['gnn_results']['f1_score'] for r in run_results]
        gnn_accuracies = [r['gnn_results']['accuracy'] for r in run_results]
        gnn_train_times = [r['gnn_results']['train_time'] for r in run_results]
        gnn_precisions = [r['gnn_results']['precision'] for r in run_results]
        gnn_recalls = [r['gnn_results']['recall'] for r in run_results]
        gnn_fnrs = [r['gnn_results']['false_negnnive_rate'] for r in run_results]
        gnn_fprs = [r['gnn_results']['false_positive_rate'] for r in run_results]
        
        rnn_f1_scores = [r['rnn_results']['f1_score'] for r in run_results]
        rnn_accuracies = [r['rnn_results']['accuracy'] for r in run_results]
        rnn_train_times = [r['rnn_results']['train_time'] for r in run_results]
        rnn_precisions = [r['rnn_results']['precision'] for r in run_results]
        rnn_recalls = [r['rnn_results']['recall'] for r in run_results]
        rnn_fnrs = [r['rnn_results']['false_negnnive_rate'] for r in run_results]
        rnn_fprs = [r['rnn_results']['false_positive_rate'] for r in run_results]
        
        # Calculate average and standard deviation
        avg_result = {
            'experiment_name': f"{self.graph_type}_n{self.n}_ratio{fault_ratio:.2f}",
            'config': run_results[0]['config'],
            'fault_count': fault_count,
            'fault_ratio': fault_ratio,
            'experiment_index': experiment_index,
            'successful_runs': len(run_results),
            'total_runs': self.num_runs,
            'gnn_results': {
                'f1_score': np.mean(gnn_f1_scores),
                'f1_std': np.std(gnn_f1_scores),
                'accuracy': np.mean(gnn_accuracies),
                'accuracy_std': np.std(gnn_accuracies),
                'precision': np.mean(gnn_precisions),
                'precision_std': np.std(gnn_precisions),
                'recall': np.mean(gnn_recalls),
                'recall_std': np.std(gnn_recalls),
                'false_negnnive_rate': np.mean(gnn_fnrs),
                'false_negnnive_rate_std': np.std(gnn_fnrs),
                'false_positive_rate': np.mean(gnn_fprs),
                'false_positive_rate_std': np.std(gnn_fprs),
                'train_time': np.mean(gnn_train_times),
                'train_time_std': np.std(gnn_train_times),
                'all_f1_scores': gnn_f1_scores,
                'all_accuracies': gnn_accuracies,
                'all_train_times': gnn_train_times
            },
            'rnn_results': {
                'f1_score': np.mean(rnn_f1_scores),
                'f1_std': np.std(rnn_f1_scores),
                'accuracy': np.mean(rnn_accuracies),
                'accuracy_std': np.std(rnn_accuracies),
                'precision': np.mean(rnn_precisions),
                'precision_std': np.std(rnn_precisions),
                'recall': np.mean(rnn_recalls),
                'recall_std': np.std(rnn_recalls),
                'false_negnnive_rate': np.mean(rnn_fnrs),
                'false_negnnive_rate_std': np.std(rnn_fnrs),
                'false_positive_rate': np.mean(rnn_fprs),
                'false_positive_rate_std': np.std(rnn_fprs),
                'train_time': np.mean(rnn_train_times),
                'train_time_std': np.std(rnn_train_times),
                'all_f1_scores': rnn_f1_scores,
                'all_accuracies': rnn_accuracies,
                'all_train_times': rnn_train_times
            }
        }
        
        return avg_result
    
    def _generate_fault_ratio_summary(self, results: List[Dict]) -> Dict:
        """Generate fault ratio experiment result summary"""
        summary = {
            'total_experiments': len(results),
            'successful_experiments': 0,
            'failed_experiments': 0,
            'gnn_scores': [],
            'rnn_scores': [],
            'fault_ratios': [],
            'fault_counts': [],
            'best_gnn_f1': 0,
            'best_rnn_f1': 0,
            'worst_gnn_f1': 1,
            'worst_rnn_f1': 1
        }
        
        for result in results:
            if 'error' in result or result['gnn_results'] is None:
                summary['failed_experiments'] += 1
                continue
            
            summary['successful_experiments'] += 1
            
            gnn_f1 = result['gnn_results']['f1_score']
            rnn_f1 = result['rnn_results']['f1_score']
            fault_ratio = result['fault_ratio']
            fault_count = result['fault_count']
            
            summary['gnn_scores'].append(gnn_f1)
            summary['rnn_scores'].append(rnn_f1)
            summary['fault_ratios'].append(fault_ratio)
            summary['fault_counts'].append(fault_count)
            
            summary['best_gnn_f1'] = max(summary['best_gnn_f1'], gnn_f1)
            summary['best_rnn_f1'] = max(summary['best_rnn_f1'], rnn_f1)
            summary['worst_gnn_f1'] = min(summary['worst_gnn_f1'], gnn_f1)
            summary['worst_rnn_f1'] = min(summary['worst_rnn_f1'], rnn_f1)
        
        if summary['gnn_scores']:
            summary['avg_gnn_f1'] = np.mean(summary['gnn_scores'])
            summary['avg_rnn_f1'] = np.mean(summary['rnn_scores'])
            summary['std_gnn_f1'] = np.std(summary['gnn_scores'])
            summary['std_rnn_f1'] = np.std(summary['rnn_scores'])
        
        return summary
    
    def _save_fault_ratio_results(self, all_results: Dict, output_dir: str):
        """Save fault ratio experiment results"""
        import json
        
        # Save JSON format detailed results
        json_file = os.path.join(output_dir, 'fault_ratio_results.json')
        
        # Prepare JSON serializable data
        json_data = {
            'metadata': all_results['metadata'],
            'summary': all_results['summary'],
            'results': []
        }
        
        # Convert results to JSON serializable format
        for result in all_results['results']:
            json_result = {
                'experiment_name': result.get('experiment_name', ''),
                'fault_count': result.get('fault_count', 0),
                'fault_ratio': result.get('fault_ratio', 0),
                'experiment_index': result.get('experiment_index', 0),
                'config': result.get('config', {}),
            }
            
            if 'error' in result:
                json_result['error'] = result['error']
                json_result['gnn_results'] = None
                json_result['rnn_results'] = None
            else:
                json_result['gnn_results'] = result.get('gnn_results', {})
                json_result['rnn_results'] = result.get('rnn_results', {})
            
            json_data['results'].append(json_result)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Save CSV format summary results
        csv_file = os.path.join(output_dir, 'fault_ratio_summary.csv')
        self._save_fault_ratio_csv_summary(all_results, csv_file)
        
        # Save text format report
        report_file = os.path.join(output_dir, 'fault_ratio_report.txt')
        self._save_fault_ratio_text_report(all_results, report_file)
        
        logger.info(f"Fault ratio results saved:")
        logger.info(f"  Detailed results: {json_file}")
        logger.info(f"  Summary CSV: {csv_file}")
        logger.info(f"  Experiment report: {report_file}")
    
    def _save_fault_ratio_csv_summary(self, all_results: Dict, csv_file: str):
        """Save CSV format fault ratio summary results"""
        data = []
        
        for result in all_results['results']:
            if 'error' in result or result['gnn_results'] is None:
                continue
            
            data.append({
                'fault_ratio': result['fault_ratio'],
                'fault_count': result['fault_count'],
                'successful_runs': result.get('successful_runs', 1),
                'total_runs': result.get('total_runs', 1),
                'gnn_accuracy': result['gnn_results']['accuracy'],
                'gnn_accuracy_std': result['gnn_results'].get('accuracy_std', 0),
                'gnn_f1_score': result['gnn_results']['f1_score'],
                'gnn_f1_std': result['gnn_results'].get('f1_std', 0),
                'gnn_precision': result['gnn_results'].get('precision', 0),
                'gnn_precision_std': result['gnn_results'].get('precision_std', 0),
                'gnn_recall': result['gnn_results'].get('recall', 0),
                'gnn_recall_std': result['gnn_results'].get('recall_std', 0),
                'gnn_fnr': result['gnn_results'].get('false_negnnive_rate', 0),
                'gnn_fnr_std': result['gnn_results'].get('false_negnnive_rate_std', 0),
                'gnn_fpr': result['gnn_results'].get('false_positive_rate', 0),
                'gnn_fpr_std': result['gnn_results'].get('false_positive_rate_std', 0),
                'gnn_train_time': result['gnn_results']['train_time'],
                'gnn_train_time_std': result['gnn_results'].get('train_time_std', 0),
                'rnn_accuracy': result['rnn_results']['accuracy'],
                'rnn_accuracy_std': result['rnn_results'].get('accuracy_std', 0),
                'rnn_f1_score': result['rnn_results']['f1_score'],
                'rnn_f1_std': result['rnn_results'].get('f1_std', 0),
                'rnn_precision': result['rnn_results'].get('precision', 0),
                'rnn_precision_std': result['rnn_results'].get('precision_std', 0),
                'rnn_recall': result['rnn_results'].get('recall', 0),
                'rnn_recall_std': result['rnn_results'].get('recall_std', 0),
                'rnn_fnr': result['rnn_results'].get('false_negnnive_rate', 0),
                'rnn_fnr_std': result['rnn_results'].get('false_negnnive_rate_std', 0),
                'rnn_fpr': result['rnn_results'].get('false_positive_rate', 0),
                'rnn_fpr_std': result['rnn_results'].get('false_positive_rate_std', 0),
                'rnn_train_time': result['rnn_results']['train_time'],
                'rnn_train_time_std': result['rnn_results'].get('train_time_std', 0),
                'gnn_better': result['gnn_results']['f1_score'] > result['rnn_results']['f1_score']
            })
        
        if data:
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)
    
    def _save_fault_ratio_text_report(self, all_results: Dict, report_file: str):
        """Save text format fault ratio experiment report"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Fault ratio comparison experiment report\n")
            f.write("=" * 50 + "\n\n")
            
            # Metadata
            metadata = all_results['metadata']
            f.write("Experiment configuration:\n")
            f.write(f"  Graph type: {metadata['graph_type']}\n")
            f.write(f"  Graph scale: n={metadata['n']}")
            if metadata['k']:
                f.write(f", k={metadata['k']}")
            f.write(f"\n  Node count: {metadata['num_vertices']}\n")
            f.write(f"  Theoretical diagnosability: {metadata['theoretical_diagnosability']}\n")
            f.write(f"  Maximum fault ratio: {metadata['max_fault_ratio']*100:.1f}%\n")
            f.write(f"  Ratio step: {metadata['ratio_step']*100:.1f}%\n")
            f.write(f"  Intermittent fault probability: {metadata['intermittent_prob']}\n")
            f.write(f"  Test rounds: {metadata['num_rounds']}\n")
            f.write(f"  Number of graphs per configuration: {metadata['num_graphs']}\n")
            f.write(f"  Number of runs per fault situation: {metadata['num_runs']}\n\n")
            
            # Summary statistics
            summary = all_results['summary']
            f.write("Experiment summary:\n")
            f.write(f"  Total experiments: {summary['total_experiments']}\n")
            f.write(f"  Successful experiments: {summary['successful_experiments']}\n")
            f.write(f"  Failed experiments: {summary['failed_experiments']}\n")
            
            if summary['successful_experiments'] > 0:
                f.write(f"  GNN average F1: {summary['avg_gnn_f1']:.4f} ± {summary['std_gnn_f1']:.4f}\n")
                f.write(f"  RNN average F1: {summary['avg_rnn_f1']:.4f} ± {summary['std_rnn_f1']:.4f}\n")
                f.write(f"  GNN best F1: {summary['best_gnn_f1']:.4f}\n")
                f.write(f"  RNN best F1: {summary['best_rnn_f1']:.4f}\n")
                
                gnn_wins = sum(1 for i in range(len(summary['gnn_scores'])) 
                              if summary['gnn_scores'][i] > summary['rnn_scores'][i])
                f.write(f"  GNN wins: {gnn_wins}/{summary['successful_experiments']}\n")
            
            f.write(f"\nTotal experiment time: {metadata['total_experiment_time']:.1f} seconds\n\n")
            
            # Detailed results
            f.write("Detailed results:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Fault ratio':>8} {'Fault count':>6} {'Success/Total':>10} {'GNN-F1':>12} {'RNN-F1':>12} {'GNN-Acc':>12} {'RNN-Acc':>12} {'Better model':>8}\n")
            f.write("-" * 120 + "\n")
            
            for result in all_results['results']:
                if 'error' in result or result['gnn_results'] is None:
                    ratio_percent = result.get('fault_ratio', 0) * 100
                    runs_info = f"{result.get('successful_runs', 0)}/{result.get('total_runs', 1)}"
                    f.write(f"{ratio_percent:>7.1f}% {result['fault_count']:>6} {runs_info:>10} {'ERROR':>12} {'ERROR':>12} {'ERROR':>12} {'ERROR':>12} {'N/A':>8}\n")
                else:
                    ratio_percent = result['fault_ratio'] * 100
                    gnn_f1 = result['gnn_results']['f1_score']
                    gnn_f1_std = result['gnn_results'].get('f1_std', 0)
                    rnn_f1 = result['rnn_results']['f1_score']
                    rnn_f1_std = result['rnn_results'].get('f1_std', 0)
                    gnn_acc = result['gnn_results']['accuracy']
                    gnn_acc_std = result['gnn_results'].get('accuracy_std', 0)
                    rnn_acc = result['rnn_results']['accuracy']
                    rnn_acc_std = result['rnn_results'].get('accuracy_std', 0)
                    better = 'GNN' if gnn_f1 > rnn_f1 else 'RNN' if rnn_f1 > gnn_f1 else 'TIE'
                    
                    runs_info = f"{result.get('successful_runs', 1)}/{result.get('total_runs', 1)}"
                    gnn_f1_str = f"{gnn_f1:.3f}±{gnn_f1_std:.3f}" if gnn_f1_std > 0 else f"{gnn_f1:.4f}"
                    rnn_f1_str = f"{rnn_f1:.3f}±{rnn_f1_std:.3f}" if rnn_f1_std > 0 else f"{rnn_f1:.4f}"
                    gnn_acc_str = f"{gnn_acc:.3f}±{gnn_acc_std:.3f}" if gnn_acc_std > 0 else f"{gnn_acc:.4f}"
                    rnn_acc_str = f"{rnn_acc:.3f}±{rnn_acc_std:.3f}" if rnn_acc_std > 0 else f"{rnn_acc:.4f}"
                    
                    f.write(f"{ratio_percent:>7.1f}% {result['fault_count']:>6} {runs_info:>10} {gnn_f1_str:>12} {rnn_f1_str:>12} "
                           f"{gnn_acc_str:>12} {rnn_acc_str:>12} {better:>8}\n")
    
    def _save_theoretical_diagnosability_csv_summary(self, all_results: Dict, csv_file: str):
        """Save CSV format theoretical diagnosability summary results"""
        data = []
        
        for result in all_results['results']:
            if 'error' in result or result['gnn_results'] is None:
                continue
            
            data.append({
                'fault_count': result['fault_count'],
                'successful_runs': result.get('successful_runs', 1),
                'total_runs': result.get('total_runs', 1),
                'gnn_accuracy': result['gnn_results']['accuracy'],
                'gnn_accuracy_std': result['gnn_results'].get('accuracy_std', 0),
                'gnn_f1_score': result['gnn_results']['f1_score'],
                'gnn_f1_std': result['gnn_results'].get('f1_std', 0),
                'gnn_precision': result['gnn_results'].get('precision', 0),
                'gnn_precision_std': result['gnn_results'].get('precision_std', 0),
                'gnn_recall': result['gnn_results'].get('recall', 0),
                'gnn_recall_std': result['gnn_results'].get('recall_std', 0),
                'gnn_fnr': result['gnn_results'].get('false_negnnive_rate', 0),
                'gnn_fnr_std': result['gnn_results'].get('false_negnnive_rate_std', 0),
                'gnn_fpr': result['gnn_results'].get('false_positive_rate', 0),
                'gnn_fpr_std': result['gnn_results'].get('false_positive_rate_std', 0),
                'gnn_train_time': result['gnn_results']['train_time'],
                'gnn_train_time_std': result['gnn_results'].get('train_time_std', 0),
                'rnn_accuracy': result['rnn_results']['accuracy'],
                'rnn_accuracy_std': result['rnn_results'].get('accuracy_std', 0),
                'rnn_f1_score': result['rnn_results']['f1_score'],
                'rnn_f1_std': result['rnn_results'].get('f1_std', 0),
                'rnn_precision': result['rnn_results'].get('precision', 0),
                'rnn_precision_std': result['rnn_results'].get('precision_std', 0),
                'rnn_recall': result['rnn_results'].get('recall', 0),
                'rnn_recall_std': result['rnn_results'].get('recall_std', 0),
                'rnn_fnr': result['rnn_results'].get('false_negnnive_rate', 0),
                'rnn_fnr_std': result['rnn_results'].get('false_negnnive_rate_std', 0),
                'rnn_fpr': result['rnn_results'].get('false_positive_rate', 0),
                'rnn_fpr_std': result['rnn_results'].get('false_positive_rate_std', 0),
                'rnn_train_time': result['rnn_results']['train_time'],
                'rnn_train_time_std': result['rnn_results'].get('train_time_std', 0),
                'gnn_better': result['gnn_results']['f1_score'] > result['rnn_results']['f1_score']
            })
        
        if data:
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)
    
    def _save_theoretical_diagnosability_text_report(self, all_results: Dict, report_file: str):
        """Save text format theoretical diagnosability experiment report"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Theoretical diagnosability comparison experiment report\n")
            f.write("=" * 50 + "\n\n")
            
            # Metadata
            metadata = all_results['metadata']
            f.write("Experiment configuration:\n")
            f.write(f"  Graph type: {metadata['graph_type']}\n")
            f.write(f"  Graph scale: n={metadata['n']}")
            if metadata['k']:
                f.write(f", k={metadata['k']}")
            f.write(f"\n  Node count: {metadata['num_vertices']}\n")
            f.write(f"  Theoretical diagnosability: {metadata['theoretical_diagnosability']}\n")
            f.write(f"  Intermittent fault probability: {metadata['intermittent_prob']}\n")
            f.write(f"  Test rounds: {metadata['num_rounds']}\n")
            f.write(f"  Number of graphs per configuration: {metadata['num_graphs']}\n")
            f.write(f"  Number of runs per fault situation: {metadata['num_runs']}\n\n")
            
            # Summary statistics
            summary = all_results['summary']
            f.write("Experiment summary:\n")
            f.write(f"  Total experiments: {summary['total_experiments']}\n")
            f.write(f"  Successful experiments: {summary['successful_experiments']}\n")
            f.write(f"  Failed experiments: {summary['failed_experiments']}\n")
            
            if summary['successful_experiments'] > 0:
                f.write(f"  GNN average F1: {summary['avg_gnn_f1']:.4f} ± {summary['std_gnn_f1']:.4f}\n")
                f.write(f"  RNN average F1: {summary['avg_rnn_f1']:.4f} ± {summary['std_rnn_f1']:.4f}\n")
                f.write(f"  GNN best F1: {summary['best_gnn_f1']:.4f}\n")
                f.write(f"  RNN best F1: {summary['best_rnn_f1']:.4f}\n")
                
                gnn_wins = sum(1 for i in range(len(summary['gnn_scores'])) 
                              if summary['gnn_scores'][i] > summary['rnn_scores'][i])
                f.write(f"  GNN wins: {gnn_wins}/{summary['successful_experiments']}\n")
            
            f.write(f"\nTotal experiment time: {metadata['total_experiment_time']:.1f} seconds\n\n")
            
            # Detailed results
            f.write("Detailed results:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Fault count':>6} {'Success/Total':>10} {'GNN-F1':>12} {'RNN-F1':>12} {'GNN-Acc':>12} {'RNN-Acc':>12} {'Better model':>8}\n")
            f.write("-" * 100 + "\n")
            
            for result in all_results['results']:
                if 'error' in result or result['gnn_results'] is None:
                    runs_info = f"{result.get('successful_runs', 0)}/{result.get('total_runs', 1)}"
                    f.write(f"{result['fault_count']:>6} {runs_info:>10} {'ERROR':>12} {'ERROR':>12} {'ERROR':>12} {'ERROR':>12} {'N/A':>8}\n")
                else:
                    gnn_f1 = result['gnn_results']['f1_score']
                    gnn_f1_std = result['gnn_results'].get('f1_std', 0)
                    rnn_f1 = result['rnn_results']['f1_score']
                    rnn_f1_std = result['rnn_results'].get('f1_std', 0)
                    gnn_acc = result['gnn_results']['accuracy']
                    gnn_acc_std = result['gnn_results'].get('accuracy_std', 0)
                    rnn_acc = result['rnn_results']['accuracy']
                    rnn_acc_std = result['rnn_results'].get('accuracy_std', 0)
                    better = 'GNN' if gnn_f1 > rnn_f1 else 'RNN' if rnn_f1 > gnn_f1 else 'TIE'
                    
                    runs_info = f"{result.get('successful_runs', 1)}/{result.get('total_runs', 1)}"
                    gnn_f1_str = f"{gnn_f1:.3f}±{gnn_f1_std:.3f}" if gnn_f1_std > 0 else f"{gnn_f1:.4f}"
                    rnn_f1_str = f"{rnn_f1:.3f}±{rnn_f1_std:.3f}" if rnn_f1_std > 0 else f"{rnn_f1:.4f}"
                    gnn_acc_str = f"{gnn_acc:.3f}±{gnn_acc_std:.3f}" if gnn_acc_std > 0 else f"{gnn_acc:.4f}"
                    rnn_acc_str = f"{rnn_acc:.3f}±{rnn_acc_std:.3f}" if rnn_acc_std > 0 else f"{rnn_acc:.4f}"
                    
                    f.write(f"{result['fault_count']:>6} {runs_info:>10} {gnn_f1_str:>12} {rnn_f1_str:>12} "
                           f"{gnn_acc_str:>12} {rnn_acc_str:>12} {better:>8}\n")
    
    def _generate_theoretical_diagnosability_visualizations(self, all_results: Dict, output_dir: str):
        """Generate theoretical diagnosability visualization charts"""
        logger.info("Generating theoretical diagnosability visualization charts...")
        
        # Extract successful experiment data
        fault_counts = []
        gnn_f1_scores = []
        rnn_f1_scores = []
        gnn_f1_stds = []
        rnn_f1_stds = []
        gnn_accuracies = []
        rnn_accuracies = []
        gnn_accuracy_stds = []
        rnn_accuracy_stds = []
        gnn_train_times = []
        rnn_train_times = []
        gnn_train_time_stds = []
        rnn_train_time_stds = []
        
        for result in all_results['results']:
            if 'error' in result or result['gnn_results'] is None:
                continue
            
            fault_counts.append(result['fault_count'])
            gnn_f1_scores.append(result['gnn_results']['f1_score'])
            rnn_f1_scores.append(result['rnn_results']['f1_score'])
            gnn_f1_stds.append(result['gnn_results'].get('f1_std', 0))
            rnn_f1_stds.append(result['rnn_results'].get('f1_std', 0))
            gnn_accuracies.append(result['gnn_results']['accuracy'])
            rnn_accuracies.append(result['rnn_results']['accuracy'])
            gnn_accuracy_stds.append(result['gnn_results'].get('accuracy_std', 0))
            rnn_accuracy_stds.append(result['rnn_results'].get('accuracy_std', 0))
            gnn_train_times.append(result['gnn_results']['train_time'])
            rnn_train_times.append(result['rnn_results']['train_time'])
            gnn_train_time_stds.append(result['gnn_results'].get('train_time_std', 0))
            rnn_train_time_stds.append(result['rnn_results'].get('train_time_std', 0))
        
        if not fault_counts:
            logger.warning("No successful experiment results, cannot generate visualizations")
            return
        
        # Set plot style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. F1 Score theoretical diagnosability trend chart
        self._create_single_chart(
            fault_counts, gnn_f1_scores, rnn_f1_scores, gnn_f1_stds, rnn_f1_stds,
            'F1 Score', 'Number of Fault Nodes', 'F1 Score',
            f'{self.graph_type.upper()} (n={self.n}) Theoretical Diagnosability F1 Trend: GNN vs RNNIFDCOM',
            output_dir, 'theoretical_diagnosability_f1_trend'
        )
        
        # 2. Accuracy theoretical diagnosability trend chart
        self._create_single_chart(
            fault_counts, gnn_accuracies, rnn_accuracies, gnn_accuracy_stds, rnn_accuracy_stds,
            'Accuracy', 'Number of Fault Nodes', 'Accuracy',
            f'{self.graph_type.upper()} (n={self.n}) Theoretical Diagnosability Accuracy Trend: GNN vs RNNIFDCOM',
            output_dir, 'theoretical_diagnosability_accuracy_trend'
        )
        
        # 3. Training time theoretical diagnosability trend chart
        self._create_single_chart(
            fault_counts, gnn_train_times, rnn_train_times, gnn_train_time_stds, rnn_train_time_stds,
            'Training Time', 'Number of Fault Nodes', 'Training Time (seconds)',
            f'{self.graph_type.upper()} (n={self.n}) Theoretical Diagnosability Training Time Trend: GNN vs RNN',
            output_dir, 'theoretical_diagnosability_time_trend',
            ylim_max=None  # Training time does not set upper limit
        )
        
        # 4. GNN vs RNN performance difference chart
        f1_diff = np.array(gnn_f1_scores) - np.array(rnn_f1_scores)
        self._create_difference_chart(
            fault_counts, f1_diff,
            'Number of Fault Nodes', 'F1 Difference (GNN - RNN)',
            f'{self.graph_type.upper()} (n={self.n}) GNN vs RNN Theoretical Diagnosability Performance Difference',
            output_dir, 'theoretical_diagnosability_performance_difference'
        )
        
        logger.info(f"Theoretical diagnosability visualization charts saved to: {output_dir}")
    
    def _create_single_chart(self, x_data, gnn_data, rnn_data, gnn_stds, rnn_stds, metric_name, 
                            xlabel, ylabel, title, output_dir, filename, ylim_max=1.05):
        """Create single comparison chart"""
        plt.figure(figsize=(12, 8))
        
        # Plot curves and error bars
        plt.errorbar(x_data, gnn_data, yerr=gnn_stds, fmt='o-', label='GNN', 
                    linewidth=3, markersize=8, color='#2E86AB', capsize=5, capthick=2)
        plt.errorbar(x_data, rnn_data, yerr=rnn_stds, fmt='s-', label='RNN', 
                    linewidth=3, markersize=8, color='#A23B72', capsize=5, capthick=2)
        
        # Add value labels (display average ± standard deviation)
        for i, (x, gnn_val, rnn_val, gnn_std, rnn_std) in enumerate(zip(x_data, gnn_data, rnn_data, gnn_stds, rnn_stds)):
            if gnn_std > 0:
                plt.annotate(f'{gnn_val:.3f}±{gnn_std:.3f}', (x, gnn_val), textcoords="offset points", 
                            xytext=(0,15), ha='center', fontsize=8, color='#2E86AB')
            else:
                plt.annotate(f'{gnn_val:.3f}', (x, gnn_val), textcoords="offset points", 
                            xytext=(0,15), ha='center', fontsize=9, color='#2E86AB')
            
            if rnn_std > 0:
                plt.annotate(f'{rnn_val:.3f}±{rnn_std:.3f}', (x, rnn_val), textcoords="offset points", 
                            xytext=(0,-20), ha='center', fontsize=8, color='#A23B72')
            else:
                plt.annotate(f'{rnn_val:.3f}', (x, rnn_val), textcoords="offset points", 
                            xytext=(0,-20), ha='center', fontsize=9, color='#A23B72')
        
        # Set chart properties
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16, pad=20)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if ylim_max is not None:
            plt.ylim(0, ylim_max)
        
        # Set x-axis ticks
        plt.xticks(x_data)
        
        plt.tight_layout()
        
        # Save as PNG and PDF format
        png_path = os.path.join(output_dir, f'{filename}.png')
        pdf_path = os.path.join(output_dir, f'{filename}.pdf')
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {filename}.png and {filename}.pdf")
    
    def _generate_fault_ratio_visualizations(self, all_results: Dict, output_dir: str):
        """Generate fault ratio visualization charts"""
        logger.info("Generating fault ratio visualization charts...")
        
        # Extract successful experiment data
        fault_ratios = []
        fault_counts = []
        gnn_f1_scores = []
        rnn_f1_scores = []
        gnn_f1_stds = []
        rnn_f1_stds = []
        gnn_accuracies = []
        rnn_accuracies = []
        gnn_accuracy_stds = []
        rnn_accuracy_stds = []
        gnn_train_times = []
        rnn_train_times = []
        gnn_train_time_stds = []
        rnn_train_time_stds = []
        
        for result in all_results['results']:
            if 'error' in result or result['gnn_results'] is None:
                continue
            
            fault_ratios.append(result['fault_ratio'] * 100)  # Convert to percentage
            fault_counts.append(result['fault_count'])
            gnn_f1_scores.append(result['gnn_results']['f1_score'])
            rnn_f1_scores.append(result['rnn_results']['f1_score'])
            gnn_f1_stds.append(result['gnn_results'].get('f1_std', 0))
            rnn_f1_stds.append(result['rnn_results'].get('f1_std', 0))
            gnn_accuracies.append(result['gnn_results']['accuracy'])
            rnn_accuracies.append(result['rnn_results']['accuracy'])
            gnn_accuracy_stds.append(result['gnn_results'].get('accuracy_std', 0))
            rnn_accuracy_stds.append(result['rnn_results'].get('accuracy_std', 0))
            gnn_train_times.append(result['gnn_results']['train_time'])
            rnn_train_times.append(result['rnn_results']['train_time'])
            gnn_train_time_stds.append(result['gnn_results'].get('train_time_std', 0))
            rnn_train_time_stds.append(result['rnn_results'].get('train_time_std', 0))
        
        if not fault_ratios:
            logger.warning("No successful experiment results, cannot generate visualizations")
            return
        
        # Set plot style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. F1 Score fault ratio trend chart
        self._create_single_chart(
            fault_ratios, gnn_f1_scores, rnn_f1_scores, gnn_f1_stds, rnn_f1_stds,
            'F1 Score', 'Fault Ratio (%)', 'F1 Score',
            f'{self.graph_type.upper()} (n={self.n}) Fault Ratio F1 Trend: GNN vs RNNIFDCOM',
            output_dir, 'fault_ratio_f1_trend'
        )
        
        # 2. Accuracy fault ratio trend chart
        self._create_single_chart(
            fault_ratios, gnn_accuracies, rnn_accuracies, gnn_accuracy_stds, rnn_accuracy_stds,
            'Accuracy', 'Fault Ratio (%)', 'Accuracy',
            f'{self.graph_type.upper()} (n={self.n}) Fault Ratio Accuracy Trend: GNN vs RNNIFDCOM',
            output_dir, 'fault_ratio_accuracy_trend'
        )
        
        # 3. Training time fault ratio trend chart
        self._create_single_chart(
            fault_ratios, gnn_train_times, rnn_train_times, gnn_train_time_stds, rnn_train_time_stds,
            'Training Time', 'Fault Ratio (%)', 'Training Time (seconds)',
            f'{self.graph_type.upper()} (n={self.n}) Fault Ratio Training Time Trend: GNN vs RNN',
            output_dir, 'fault_ratio_time_trend',
            ylim_max=None  # Training time does not set upper limit
        )
        
        # 4. GNN vs RNN performance difference chart
        f1_diff = np.array(gnn_f1_scores) - np.array(rnn_f1_scores)
        self._create_difference_chart(
            fault_ratios, f1_diff,
            'Fault Ratio (%)', 'F1 Difference (GNN - RNN)',
            f'{self.graph_type.upper()} (n={self.n}) GNN vs RNN Fault Ratio Performance Difference',
            output_dir, 'fault_ratio_performance_difference'
        )
        
        logger.info(f"Fault ratio visualization charts saved to: {output_dir}")
    
    def _create_difference_chart(self, x_data, diff_data, xlabel, ylabel, title, output_dir, filename):
        """Create performance difference bar chart"""
        plt.figure(figsize=(12, 8))
        
        # Set colors based on difference values
        colors = ['#27AE60' if diff > 0 else '#E74C3C' if diff < 0 else '#95A5A6' for diff in diff_data]
        
        # Plot bar chart
        bars = plt.bar(x_data, diff_data, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
        
        # Add value labels
        for bar, diff in zip(bars, diff_data):
            height = bar.get_height()
            plt.annotate(f'{diff:.3f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold')
        
        # Set chart properties
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16, pad=20)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Set x-axis ticks
        plt.xticks(x_data)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27AE60', label='GNN Better'),
            Patch(facecolor='#E74C3C', label='RNN Better'),
            Patch(facecolor='#95A5A6', label='Comparable')
        ]
        plt.legend(handles=legend_elements, fontsize=12)
        
        plt.tight_layout()
        
        # Save as PNG and PDF format
        png_path = os.path.join(output_dir, f'{filename}.png')
        pdf_path = os.path.join(output_dir, f'{filename}.pdf')
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {filename}.png and {filename}.pdf")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fault ratio comparison experiment: GNN vs RNN')
    
    # Graph configuration parameters
    parser.add_argument('--graph_type', type=str, default='bc', 
                       help='Graph type (bc, star, alternating_group, etc.)')
    parser.add_argument('--n', type=int, default=7, help='Graph scale parameter')
    parser.add_argument('--k', type=int, default=None, help='k-ary cube parameter')
    
    # Fault ratio parameters
    parser.add_argument('--max_fault_ratio', type=float, default=0.3, 
                       help='Maximum fault node ratio (0.0-1.0)')
    parser.add_argument('--ratio_step', type=float, default=0.05, 
                       help='Ratio step, default 0.05 (5%%)')
    
    # Experiment parameters
    parser.add_argument('--intermittent_prob', type=float, default=0.5, help='Intermittent fault probability')
    parser.add_argument('--num_rounds', type=int, default=10, help='Number of test rounds')
    parser.add_argument('--num_graphs', type=int, default=500, help='Number of graphs per ratio')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs per fault situation')
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
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    
    # Output parameters
    parser.add_argument('--dataset_base_dir', type=str, default='datasets', help='Data set base directory')
    parser.add_argument('--force_regenerate', action='store_true', help='Force regenerate dataset (default: reuse existing dataset)')
    parser.add_argument('--output_dir', type=str, default=None, help='Result output directory (automatically generated if not specified)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info("=" * 80)
    logger.info("Start fault ratio comparison experiment")
    logger.info("=" * 80)
    
    try:
        # Create fault ratio comparison experiment object
        comparison = FaultRatioComparison(
            graph_type=args.graph_type,
            n=args.n,
            k=args.k,
            max_fault_ratio=args.max_fault_ratio,
            ratio_step=args.ratio_step,
            intermittent_prob=args.intermittent_prob,
            num_rounds=args.num_rounds,
            num_graphs=args.num_graphs,
            num_runs=args.num_runs,
            seed=args.seed
        )
        
        # Run fault ratio comparison experiment
        results = comparison.run_fault_ratio_experiments(args, args.output_dir)
        
        # Output final summary
        summary = results['summary']
        if summary['successful_experiments'] > 0:
            logger.info("=" * 80)
            logger.info("Fault ratio experiment final result summary:")
            logger.info(f"  Successfully completed {summary['successful_experiments']} experiments")
            logger.info(f"  GNN average F1: {summary['avg_gnn_f1']:.4f}")
            logger.info(f"  RNN average F1: {summary['avg_rnn_f1']:.4f}")
            
            gnn_wins = sum(1 for i in range(len(summary['gnn_scores'])) 
                          if summary['gnn_scores'][i] > summary['rnn_scores'][i])
            logger.info(f"  GNN wins: {gnn_wins}/{summary['successful_experiments']} times")
            logger.info("=" * 80)
        else:
            logger.error("All fault ratio experiments failed!")
    
    except Exception as e:
        logger.error(f"Fault ratio experiment execution failed: {e}")
        raise


if __name__ == "__main__":
    main() 