"""
Models package for Fault Diagnosis with GNN.

This package contains neural network models for fault diagnosis:
- FaultGNN: Graph Attention Network for PMC model
- RNNIFDCom_PMC: RNN-based Intermittent Fault Diagnosis Communication for PMC model
"""

from .FaultGNN import FaultGNN
from .RNNIFDCom_PMC import RNNIFDCom_PMC

__all__ = ['FaultGNN', 'RNNIFDCom_PMC'] 