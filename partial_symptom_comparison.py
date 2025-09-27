#!/usr/bin/env python3
"""
Partial Symptom Comparison Experiment
===================================

This script is used to compare GNN and RNNIFDCOM model performance under partial symptom conditions.
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


class PartialSymptomComparison:
    """Partial symptom comparison experiment class"""
    
    def __init__(self, graph_type: str, n: int, k: int = None, 
                 max_missing_ratio: float = 0.5, ratio_step: float = 0.05,
                 num_runs: int = 5, num_graphs: int = 100, epochs: int = 100):
        """
        Initialize partial symptom comparison experiment
        
        Args:
            graph_type: Graph type
            n: Graph scale parameter
            k: k-ary cube parameter
            max_missing_ratio: Maximum missing ratio
            ratio_step: Missing ratio step size
            num_runs: Number of repetitions for each missing ratio
            num_graphs: Number of graphs generated each time
            epochs: Number of training epochs
        """
        self.graph_type = graph_type
        self.n = n
        self.k = k
        self.max_missing_ratio = max_missing_ratio
        self.ratio_step = ratio_step
        self.num_runs = num_runs
        self.num_graphs = num_graphs
        self.epochs = epochs
        
        # Generate missing ratio list
        self.missing_ratios = np.arange(0.0, max_missing_ratio + ratio_step/2, ratio_step)
        logger.info(f"Missing ratio list: {self.missing_ratios}")
        
    def run_partial_symptom_experiments(self, args, missing_types: List[str], output_dir: str) -> Dict:
        """
        Run partial symptom experiments
        
        Args:
            args: Command line arguments
            missing_types: Missing type list, e.g. ['node_disable']
            output_dir: Output directory
            
        Returns:
            Experiment results dictionary
        """
        logger.info("Starting partial symptom experiments")
        logger.info(f"Graph configuration: {self.graph_type}, n={self.n}")
        logger.info(f"Missing types: {missing_types}")
        logger.info(f"Missing ratio range: {0}% - {self.max_missing_ratio*100}%")
        logger.info(f"Repetitions per configuration: {self.num_runs} times")
        
        # Store all results
        all_results = []
        experiment_configs = []
        
        # Create experiment configurations for each missing type and each missing ratio
        for missing_type in missing_types:
            for missing_ratio in self.missing_ratios:
                for run_id in range(self.num_runs):
                    config = {
                        'graph_type': self.graph_type,
                        'n': self.n,
                        'k': self.k,
                        'missing_ratio': missing_ratio,
                        'missing_type': missing_type,
                        'run_id': run_id
                    }
                    experiment_configs.append(config)
        
        logger.info(f"Total {len(experiment_configs)} experiments to run")
        
        # Run all experiments
        for i, config in enumerate(experiment_configs):
            logger.info(f"\nProgress: {i+1}/{len(experiment_configs)}")
            logger.info(f"Current configuration: {config}")
            
            try:
                result = self._run_single_partial_symptom_experiment(config, args, output_dir)
                if result is not None:
                    result.update(config)  # Add configuration information to results
                    all_results.append(result)
                    gnn_f1 = result.get('gnn_test_f1', 'N/A')
                    rnn_f1 = result.get('rnn_test_f1', 'N/A')
                    gnn_f1_str = f"{gnn_f1:.4f}" if isinstance(gnn_f1, (int, float)) else str(gnn_f1)
                    rnn_f1_str = f"{rnn_f1:.4f}" if isinstance(rnn_f1, (int, float)) else str(rnn_f1)
                    logger.info(f"Experiment successful: GNN F1={gnn_f1_str}, RNN F1={rnn_f1_str}")
                else:
                    logger.warning(f"Experiment failed: {config}")
                    
            except Exception as e:
                logger.error(f"Experiment exception: {config}, error: {e}")
                continue
        
        logger.info(f"Completed {len(all_results)}/{len(experiment_configs)} experiments")
        
        # Generate summary results
        summary_results = self._summarize_results(all_results, missing_types)
        
        # Save results
        self._save_results(all_results, summary_results, output_dir)
        
        # Generate plots
        self._generate_plots(summary_results, output_dir)
        
        return {
            'all_results': all_results,
            'summary_results': summary_results,
            'configs': experiment_configs
        }
    
    def _run_single_partial_symptom_experiment(self, config: Dict, args, output_dir: str) -> Dict:
        """
        Run single partial symptom experiment
        
        Args:
            config: Experiment configuration
            args: Command line arguments  
            output_dir: Output directory
            
        Returns:
            Experiment results
        """
        missing_ratio = config['missing_ratio']
        missing_type = config['missing_type']
        run_id = config['run_id']
        
        logger.info(f"Running experiment: {missing_type}, missing ratio={missing_ratio*100:.1f}%, run={run_id}")
        
        # Create modified arguments
        run_args = argparse.Namespace(**vars(args))
        
        # Ensure all required parameters exist
        if not hasattr(run_args, 'gnn_batch_size'):
            run_args.gnn_batch_size = 16
        if not hasattr(run_args, 'gnn_hidden_dim'):
            run_args.gnn_hidden_dim = 64
        if not hasattr(run_args, 'gnn_num_layers'):
            run_args.gnn_num_layers = 2
        if not hasattr(run_args, 'gnn_heads'):
            run_args.gnn_heads = 8
        if not hasattr(run_args, 'rnn_hidden_dims'):
            run_args.rnn_hidden_dims = [64, 32]
        if not hasattr(run_args, 'lr'):
            run_args.lr = 0.002
        
        # Prepare experiment configuration
        partial_config = {
            'graph_type': config['graph_type'],
            'n': config['n'],
            'k': config['k'],
            'missing_ratio': missing_ratio,
            'missing_type': missing_type
        }
        
        try:
            # Run experiment
            result = run_single_experiment(partial_config, args, output_dir)
            return result
            
        except Exception as e:
            logger.error(f"Single experiment run failed: {e}")
            return None
    
    def _summarize_results(self, all_results: List[Dict], missing_types: List[str]) -> pd.DataFrame:
        """
        Summarize experiment results
        
        Args:
            all_results: All experiment results
            missing_types: Missing type list
            
        Returns:
            Summary results DataFrame
        """
        logger.info("Summarizing experiment results...")
        
        # Flatten nested result dictionary
        flattened_results = []
        for result in all_results:
            if result is None:
                continue
                
            flattened_result = {
                'missing_type': result.get('missing_type'),
                'missing_ratio': result.get('missing_ratio'),
                'run_id': result.get('run_id', 0)
            }
            
            # Extract GNN results
            if 'gnn_results' in result and result['gnn_results']:
                gnn_results = result['gnn_results']
                flattened_result['gnn_test_accuracy'] = gnn_results.get('accuracy')
                flattened_result['gnn_test_f1'] = gnn_results.get('f1_score')
                flattened_result['gnn_test_precision'] = gnn_results.get('precision')
                flattened_result['gnn_test_recall'] = gnn_results.get('recall')
                flattened_result['gnn_test_fnr'] = gnn_results.get('false_negnnive_rate')
                flattened_result['gnn_test_fpr'] = gnn_results.get('false_positive_rate')
            
            # Extract RNN results
            if 'rnn_results' in result and result['rnn_results']:
                rnn_results = result['rnn_results']
                flattened_result['rnn_test_accuracy'] = rnn_results.get('accuracy')
                flattened_result['rnn_test_f1'] = rnn_results.get('f1_score')
                flattened_result['rnn_test_precision'] = rnn_results.get('precision')
                flattened_result['rnn_test_recall'] = rnn_results.get('recall')
                flattened_result['rnn_test_fnr'] = rnn_results.get('false_negnnive_rate')
                flattened_result['rnn_test_fpr'] = rnn_results.get('false_positive_rate')
            
            flattened_results.append(flattened_result)
        
        # Convert to DataFrame
        df = pd.DataFrame(flattened_results)
        
        if df.empty:
            logger.warning("No valid experiment results")
            return pd.DataFrame()
        
        # Group by missing type and missing ratio, calculate statistics
        summary_rows = []
        
        for missing_type in missing_types:
            for missing_ratio in self.missing_ratios:
                # Filter current configuration results
                mask = (df['missing_type'] == missing_type) & (df['missing_ratio'] == missing_ratio)
                subset = df[mask]
                
                if len(subset) == 0:
                    continue
                
                # Calculate GNN metrics statistics
                gnn_metrics = ['gnn_test_accuracy', 'gnn_test_f1', 'gnn_test_precision', 'gnn_test_recall', 'gnn_test_fnr', 'gnn_test_fpr']
                rnn_metrics = ['rnn_test_accuracy', 'rnn_test_f1', 'rnn_test_precision', 'rnn_test_recall', 'rnn_test_fnr', 'rnn_test_fpr']
                
                summary_row = {
                    'missing_type': missing_type,
                    'missing_ratio': missing_ratio,
                    'missing_percentage': missing_ratio * 100,
                    'num_runs': len(subset)
                }
                
                # GNN statistics
                for metric in gnn_metrics:
                    if metric in subset.columns:
                        values = subset[metric].dropna()
                        if len(values) > 0:
                            summary_row[f'{metric}_mean'] = values.mean()
                            summary_row[f'{metric}_std'] = values.std()
                            summary_row[f'{metric}_min'] = values.min()
                            summary_row[f'{metric}_max'] = values.max()
                
                # RNN statistics
                for metric in rnn_metrics:
                    if metric in subset.columns:
                        values = subset[metric].dropna()
                        if len(values) > 0:
                            summary_row[f'{metric}_mean'] = values.mean()
                            summary_row[f'{metric}_std'] = values.std()
                            summary_row[f'{metric}_min'] = values.min()
                            summary_row[f'{metric}_max'] = values.max()
                
                summary_rows.append(summary_row)
        
        summary_df = pd.DataFrame(summary_rows)
        logger.info(f"Summary completed, {len(summary_df)} configurations")
        
        return summary_df
    
    def _save_results(self, all_results: List[Dict], summary_results: pd.DataFrame, output_dir: str):
        """
        Save experiment results
        
        Args:
            all_results: All experiment results
            summary_results: Summary results
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        detailed_df = pd.DataFrame(all_results)
        detailed_path = os.path.join(output_dir, "detailed_results.csv")
        detailed_df.to_csv(detailed_path, index=False)
        logger.info(f"Detailed results saved: {detailed_path}")
        
        # Save summary results
        summary_path = os.path.join(output_dir, "summary_results.csv")
        summary_results.to_csv(summary_path, index=False)
        logger.info(f"Summary results saved: {summary_path}")
        
        # Generate formatted statistics table
        self._generate_performance_table(summary_results, output_dir)
        
        # Save experiment configuration
        config_info = {
            'graph_type': self.graph_type,
            'n': self.n,
            'k': self.k,
            'max_missing_ratio': self.max_missing_ratio,
            'ratio_step': self.ratio_step,
            'num_runs': self.num_runs,
            'num_graphs': self.num_graphs,
            'epochs': self.epochs,
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = os.path.join(output_dir, "experiment_config.json")
        import json
        with open(config_path, 'w') as f:
            json.dump(config_info, f, indent=2)
        logger.info(f"Experiment configuration saved: {config_path}")
    
    def _generate_performance_table(self, summary_df: pd.DataFrame, output_dir: str):
        """
        Generate formatted performance statistics table
        
        Args:
            summary_df: Summary results DataFrame
            output_dir: Output directory
        """
        if summary_df.empty:
            logger.warning("No data to generate statistics table")
            return
        
        table_path = os.path.join(output_dir, "performance_statistics_table.txt")
        
        with open(table_path, 'w', encoding='utf-8') as f:
            # Write table header
            f.write("RQ4: Performance statistics table for partial symptom fault diagnosis\n")
            f.write("=" * 120 + "\n\n")
            
            # Experiment configuration information
            f.write("Experiment configuration:\n")
            f.write(f"  Graph type: {self.graph_type}\n")
            f.write(f"  Graph size: n={self.n}\n")
            if self.k:
                f.write(f"  k value: {self.k}\n")
            f.write(f"  Maximum missing ratio: {self.max_missing_ratio * 100:.1f}%\n")
            f.write(f"  Missing step: {self.ratio_step * 100:.1f}%\n")
            f.write(f"  Number of experiments: {self.num_runs}\n")
            f.write(f"  Number of graphs per experiment: {self.num_graphs}\n")
            f.write("-" * 120 + "\n\n")
            
            # Get all missing types
            missing_types = summary_df['missing_type'].unique()
            
            for missing_type in missing_types:
                f.write(f"Missing type: {missing_type.upper()}\n")
                f.write("-" * 120 + "\n")
                
                # Table header
                f.write(f"{'Missing ratio':<8} {'Model':<8} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 score':<12} {'FNR':<12} {'FPR':<12}\n")
                f.write("-" * 120 + "\n")
                
                # Filter current missing type data
                subset = summary_df[summary_df['missing_type'] == missing_type].sort_values('missing_ratio')
                
                for _, row in subset.iterrows():
                    missing_ratio = row['missing_ratio']
                    missing_percentage = f"{missing_ratio * 100:.1f}%"
                    
                    # GNN results
                    gnn_accuracy = f"{row.get('gnn_test_accuracy_mean', 0):.4f}" if 'gnn_test_accuracy_mean' in row and pd.notna(row.get('gnn_test_accuracy_mean')) else "N/A"
                    gnn_precision = f"{row.get('gnn_test_precision_mean', 0):.4f}" if 'gnn_test_precision_mean' in row and pd.notna(row.get('gnn_test_precision_mean')) else "N/A"
                    gnn_recall = f"{row.get('gnn_test_recall_mean', 0):.4f}" if 'gnn_test_recall_mean' in row and pd.notna(row.get('gnn_test_recall_mean')) else "N/A"
                    gnn_f1 = f"{row.get('gnn_test_f1_mean', 0):.4f}" if 'gnn_test_f1_mean' in row and pd.notna(row.get('gnn_test_f1_mean')) else "N/A"
                    
                    # GNN false negnnive rate and false positive rate
                    gnn_fnr = f"{row.get('gnn_test_fnr_mean', 0):.4f}" if 'gnn_test_fnr_mean' in row and pd.notna(row.get('gnn_test_fnr_mean')) else "N/A"
                    gnn_fpr = f"{row.get('gnn_test_fpr_mean', 0):.4f}" if 'gnn_test_fpr_mean' in row and pd.notna(row.get('gnn_test_fpr_mean')) else "N/A"
                    
                    # RNN results
                    rnn_accuracy = f"{row.get('rnn_test_accuracy_mean', 0):.4f}" if 'rnn_test_accuracy_mean' in row and pd.notna(row.get('rnn_test_accuracy_mean')) else "N/A"
                    rnn_precision = f"{row.get('rnn_test_precision_mean', 0):.4f}" if 'rnn_test_precision_mean' in row and pd.notna(row.get('rnn_test_precision_mean')) else "N/A"
                    rnn_recall = f"{row.get('rnn_test_recall_mean', 0):.4f}" if 'rnn_test_recall_mean' in row and pd.notna(row.get('rnn_test_recall_mean')) else "N/A"
                    rnn_f1 = f"{row.get('rnn_test_f1_mean', 0):.4f}" if 'rnn_test_f1_mean' in row and pd.notna(row.get('rnn_test_f1_mean')) else "N/A"
                    
                    # RNN false negnnive rate and false positive rate
                    rnn_fnr = f"{row.get('rnn_test_fnr_mean', 0):.4f}" if 'rnn_test_fnr_mean' in row and pd.notna(row.get('rnn_test_fnr_mean')) else "N/A"
                    rnn_fpr = f"{row.get('rnn_test_fpr_mean', 0):.4f}" if 'rnn_test_fpr_mean' in row and pd.notna(row.get('rnn_test_fpr_mean')) else "N/A"
                    
                    # Write GNN row
                    f.write(f"{missing_percentage:<8} {'GNN':<8} {gnn_accuracy:<12} {gnn_precision:<12} {gnn_recall:<12} {gnn_f1:<12} {gnn_fnr:<12} {gnn_fpr:<12}\n")
                    
                    # Write RNN row
                    f.write(f"{'':<8} {'RNN':<8} {rnn_accuracy:<12} {rnn_precision:<12} {rnn_recall:<12} {rnn_f1:<12} {rnn_fnr:<12} {rnn_fpr:<12}\n")
                    
                    # Empty line separator
                    f.write("-" * 120 + "\n")
                
                f.write("\n")
            
            # Add explanation
            f.write("Explanation:\n")
            f.write("- Accuracy (Accuracy): Correctly predicted samples / Total samples\n")
            f.write("- Precision (Precision): True positives / (true positives + false positives)\n")
            f.write("- Recall (Recall): True positives / (true positives + false negnnives)\n")
            f.write("- F1 score (F1 score): 2 × (precision × recall) / (precision + recall)\n")
            f.write("- FNR (FNR): False negnnives / (true positives + false negnnives) = 1 - recall\n")
            f.write("- FPR (FPR): False positives / (false positives + true negnnives)\n")
            f.write("- N/A: Data unavailable or calculation error\n")
            f.write("\nNote: In fault diagnosis, fault nodes are positive, and non-fault nodes are negnnive\n")
        
        logger.info(f"Performance statistics table saved: {table_path}")
    
    def _generate_plots(self, summary_df: pd.DataFrame, output_dir: str):
        """
        Generate experimental plots
        
        Args:
            summary_df: Summary results DataFrame
            output_dir: Output directory
        """
        if summary_df.empty:
            logger.warning("No data to generate plots")
            return
        
        logger.info("Generating experimental plots...")
        
        # Set plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. F1 score comparison plot
        self._plot_f1_comparison(summary_df, output_dir)
        
        # 2. Accuracy comparison plot
        self._plot_accuracy_comparison(summary_df, output_dir)
        
        # 3. Performance degradation analysis plot
        self._plot_performance_degradation(summary_df, output_dir)
        
        # 4. Model robustness comparison plot
        self._plot_robustness_comparison(summary_df, output_dir)
        
        logger.info("Plots generated")
    
    def _plot_f1_comparison(self, summary_df: pd.DataFrame, output_dir: str):
        """Generate F1 score comparison plot"""
        plt.figure(figsize=(12, 8))
        
        missing_types = summary_df['missing_type'].unique()
        
        for missing_type in missing_types:
            subset = summary_df[summary_df['missing_type'] == missing_type]
            
            # Check if columns exist
            if 'gnn_test_f1_mean' in subset.columns:
                # GNN F1
                gnn_yerr = subset['gnn_test_f1_std'] if 'gnn_test_f1_std' in subset.columns else None
                plt.errorbar(subset['missing_percentage'], subset['gnn_test_f1_mean'], 
                            yerr=gnn_yerr, 
                            label=f'GNN ({missing_type})', marker='o', linewidth=2)
            
            if 'rnn_test_f1_mean' in subset.columns:
                # RNN F1  
                rnn_yerr = subset['rnn_test_f1_std'] if 'rnn_test_f1_std' in subset.columns else None
                plt.errorbar(subset['missing_percentage'], subset['rnn_test_f1_mean'],
                            yerr=rnn_yerr, 
                            label=f'RNN ({missing_type})', marker='s', linewidth=2)
        
        plt.xlabel('Missing ratio (%)', fontsize=12)
        plt.ylabel('F1 score', fontsize=12)
        plt.title('F1 performance comparison under partial symptoms', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'f1_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"F1 comparison plot saved: {plot_path}")
    
    def _plot_accuracy_comparison(self, summary_df: pd.DataFrame, output_dir: str):
        """Generate accuracy comparison plot"""
        plt.figure(figsize=(12, 8))
        
        missing_types = summary_df['missing_type'].unique()
        
        for missing_type in missing_types:
            subset = summary_df[summary_df['missing_type'] == missing_type]
            
            # Check if columns exist
            if 'gnn_test_accuracy_mean' in subset.columns:
                # GNN Accuracy
                gnn_yerr = subset['gnn_test_accuracy_std'] if 'gnn_test_accuracy_std' in subset.columns else None
                plt.errorbar(subset['missing_percentage'], subset['gnn_test_accuracy_mean'],
                            yerr=gnn_yerr,
                            label=f'GNN ({missing_type})', marker='o', linewidth=2)
            
            if 'rnn_test_accuracy_mean' in subset.columns:
                # RNN Accuracy
                rnn_yerr = subset['rnn_test_accuracy_std'] if 'rnn_test_accuracy_std' in subset.columns else None
                plt.errorbar(subset['missing_percentage'], subset['rnn_test_accuracy_mean'],
                            yerr=rnn_yerr,
                            label=f'RNN ({missing_type})', marker='s', linewidth=2)
        
        plt.xlabel('Missing ratio (%)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy comparison under partial symptoms', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'accuracy_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Accuracy comparison plot saved: {plot_path}")
    
    def _plot_performance_degradation(self, summary_df: pd.DataFrame, output_dir: str):
        """Generate performance degradation analysis plot"""
        plt.figure(figsize=(12, 8))
        
        missing_types = summary_df['missing_type'].unique()
        
        for missing_type in missing_types:
            subset = summary_df[summary_df['missing_type'] == missing_type].sort_values('missing_percentage')
            
            if len(subset) == 0:
                continue
                
            # Check if columns exist and calculate performance degradation relative to no missing
            if 'gnn_test_f1_mean' in subset.columns and 'rnn_test_f1_mean' in subset.columns:
                baseline_gnn = subset.iloc[0]['gnn_test_f1_mean']  # The first point as baseline
                baseline_rnn = subset.iloc[0]['rnn_test_f1_mean']
                
                if baseline_gnn > 0:  # Avoid division by zero error
                    gnn_degradation = (baseline_gnn - subset['gnn_test_f1_mean']) / baseline_gnn * 100
                    plt.plot(subset['missing_percentage'], gnn_degradation, 
                            label=f'GNN ({missing_type})', marker='o', linewidth=2)
                
                if baseline_rnn > 0:  # Avoid division by zero error
                    rnn_degradation = (baseline_rnn - subset['rnn_test_f1_mean']) / baseline_rnn * 100
                    plt.plot(subset['missing_percentage'], rnn_degradation, 
                            label=f'RNN ({missing_type})', marker='s', linewidth=2)
        
        plt.xlabel('Missing ratio (%)', fontsize=12)
        plt.ylabel('Performance degradation (%)', fontsize=12)
        plt.title('Performance degradation analysis under partial symptoms', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'performance_degradation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Performance degradation plot saved: {plot_path}")
    
    def _plot_robustness_comparison(self, summary_df: pd.DataFrame, output_dir: str):
        """Generate model robustness comparison plot"""
        plt.figure(figsize=(10, 6))
        
        missing_types = summary_df['missing_type'].unique()
        models = ['GNN', 'RNN']
        
        # Calculate average performance standard deviation for each missing type (robustness indicator)
        robustness_data = []
        
        for missing_type in missing_types:
            subset = summary_df[summary_df['missing_type'] == missing_type]
            
            # Check if columns exist
            gnn_robustness = subset['gnn_test_f1_std'].mean() if 'gnn_test_f1_std' in subset.columns else 0
            rnn_robustness = subset['rnn_test_f1_std'].mean() if 'rnn_test_f1_std' in subset.columns else 0
            
            robustness_data.append([gnn_robustness, rnn_robustness])
        
        # Plot bar chart
        x = np.arange(len(missing_types))
        width = 0.35
        
        gnn_values = [data[0] for data in robustness_data]
        rnn_values = [data[1] for data in robustness_data]
        
        plt.bar(x - width/2, gnn_values, width, label='GNN', alpha=0.8)
        plt.bar(x + width/2, rnn_values, width, label='RNN', alpha=0.8)
        
        plt.xlabel('Missing type', fontsize=12)
        plt.ylabel('F1 score standard deviation (robustness indicator)', fontsize=12)
        plt.title('Model robustness comparison\n(smaller standard deviation means more robust)', fontsize=14, fontweight='bold')
        plt.xticks(x, missing_types)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'robustness_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Model robustness comparison plot saved: {plot_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='GNN vs. RNN fault diagnosis comparison under partial symptoms')
    
    # Graph configuration parameters
    parser.add_argument('--graph_type', type=str, default='augmented_k_ary_n_cube',
                       help='Graph type')
    parser.add_argument('--n', type=int, default=5, help='Graph scale parameter')
    parser.add_argument('--k', type=int, default=None, help='k-ary cube parameter')
    
    # Partial symptom parameters
    parser.add_argument('--max_missing_ratio', type=float, default=0.5,
                       help='Maximum missing ratio')
    parser.add_argument('--ratio_step', type=float, default=0.05,
                       help='Missing ratio step')
    parser.add_argument('--missing_type', type=str, default='node_disable',
                       choices=['node_disable'],
                       help='Missing type')
    
    # Experiment parameters
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of runs for each missing ratio')
    parser.add_argument('--num_graphs', type=int, default=100,
                       help='Number of graphs generated for each run')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel processes')
    
    # Dataset parameters
    parser.add_argument('--intermittent_prob', type=float, default=0.5,
                       help='Intermittent fault probability')
    parser.add_argument('--num_rounds', type=int, default=10,
                       help='Number of test rounds')
    parser.add_argument('--force_regenerate', action='store_true',
                       help='Force regenerate dataset')
    parser.add_argument('--dataset_base_dir', type=str, default='datasets',
                       help='Dataset base directory')
    
    # GNN parameters
    parser.add_argument('--gnn_hidden_dim', type=int, default=64, help='GNN hidden layer dimension')
    parser.add_argument('--gnn_num_layers', type=int, default=2, help='GNN number of layers')
    parser.add_argument('--gnn_heads', type=int, default=8, help='GNN number of attention heads')
    parser.add_argument('--gnn_batch_size', type=int, default=16, help='GNN batch size')
    
    # RNN parameters
    parser.add_argument('--rnn_hidden_dims', type=int, nargs='+', default=[64, 32], 
                       help='RNN hidden layer dimension')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results_partial_symptom',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Determine missing types
    missing_types = [args.missing_type]
    
    logger.info("=" * 80)
    logger.info("RQ4: Fault diagnosis comparison experiment under partial symptoms")
    logger.info("=" * 80)
    logger.info(f"Graph configuration: {args.graph_type}, n={args.n}")
    logger.info(f"Missing type: {missing_types}")
    logger.info(f"Missing ratio: 0% - {args.max_missing_ratio*100}% (step: {args.ratio_step*100}%)")
    logger.info(f"Number of runs: {args.num_runs}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    # Create experiment instance
    comparison = PartialSymptomComparison(
        graph_type=args.graph_type,
        n=args.n,
        k=args.k,
        max_missing_ratio=args.max_missing_ratio,
        ratio_step=args.ratio_step,
        num_runs=args.num_runs,
        num_graphs=args.num_graphs,
        epochs=args.epochs
    )
    
    # Run experiment
    start_time = time.time()
    results = comparison.run_partial_symptom_experiments(args, missing_types, args.output_dir)
    total_time = time.time() - start_time
    
    logger.info("=" * 80)
    logger.info("Experiment completed!")
    logger.info(f"Total time: {total_time:.1f} seconds")
    logger.info(f"Number of experiments: {len(results['all_results'])}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main() 