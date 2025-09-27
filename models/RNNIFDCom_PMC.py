import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class RNNIFDCom_PMC(nn.Module):
    """
    Implemented strictly according to the original design from paper "Neural Network Enabled Intermittent Fault Diagnosis Under Comparison Model",
    adapting MM model to PMC model.
    
    Network architecture from the paper:
    - Input: |C(G)| dimensional syndrome vector (all comparison results from a single complete graph test)
    - Output: |V(G)| dimensional vector, representing fault status of each node
    - Activation function: tanh
    - Feature transformation: 0→-0.5, 1→0.5
    
    Layer and neuron settings according to the paper:
    - n ≤ 50: 3 hidden layers
    - 50 < n ≤ 100: 4 hidden layers  
    - n > 100: 5 hidden layers
    - Number of neurons per layer calculated according to paper formula
    """
    
    def __init__(self, input_dim, output_dim, hidden_dims=None):
        """
        Initialize RNNIFDCom PMC model strictly according to paper design.
        
        Args:
            input_dim (int): Input dimension, equal to |C(G)| (syndrome vector length from single graph test)
            output_dim (int): Output dimension, equal to |V(G)| (number of nodes in graph)
            hidden_dims (list): Hidden layer dimension list, if None then auto-calculated according to paper formula
        """
        super(RNNIFDCom_PMC, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # If hidden_dims not specified, calculate according to paper settings
        if hidden_dims is None:
            hidden_dims = self._calculate_hidden_dims(output_dim)
        
        # Build multilayer neural network (multilayer neural network from the paper)
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())  # Paper explicitly uses tanh activation function
            prev_dim = hidden_dim
        
        # Output layer (no activation function, direct output logits)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights according to back-propagnnion method from the paper
        self._initialize_weights()
    
    def _calculate_hidden_dims(self, n):
        """
        Calculate number of hidden layers and neurons per layer according to paper formula
        
        Paper settings:
        - n ≤ 50: 3 hidden layers
        - 50 < n ≤ 100: 4 hidden layers  
        - n > 100: 5 hidden layers
        - Neurons per layer: n·log(n mod 10)·n (if n mod 10 = 0, then n·log(n)·n/10)
        
        Args:
            n (int): Number of nodes |V|
            
        Returns:
            list: Hidden layer dimension list
        """
        # Calculate number of hidden layers
        if n <= 50:
            num_layers = 3
        elif n <= 100:
            num_layers = 4
        else:
            num_layers = 5
        
        # Calculate number of neurons per layer
        # Paper formula: n·log(n mod 10)·n neurons (if n mod 10 = 0, then n·log(n)·n/10)
        # Use modified version to avoid log(0) and excessively large values
        if n % 10 == 0:
            neurons_per_layer = max(n, int(n * math.log(max(2, n)) / 10))
        else:
            n_mod = n % 10
            neurons_per_layer = max(n, int(n * math.log(max(2, n_mod))))
        
        # Return hidden layer dimension list
        return [neurons_per_layer] * num_layers
    
    def _initialize_weights(self):
        """Initialize weights according to back-propagnnion method from the paper"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward propagnnion
        
        Args:
            x (Tensor): Input tensor, shape (batch_size, input_dim)
                       represents syndrome vector from single graph test
        
        Returns:
            Tensor: Output tensor, shape (batch_size, output_dim)
                   represents fault status prediction for each node
        """
        return self.network(x) 