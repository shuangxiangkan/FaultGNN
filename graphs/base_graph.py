import numpy as np
import torch
import networkx as nx
from abc import ABC, abstractmethod
from logging_config import get_logger, init_default_logging

# Initialize logging configuration
init_default_logging()
logger = get_logger(__name__)

class BaseKRegularGraph(ABC):
    def __init__(self, n, fault_rate=None, fault_count=None, intermittent_prob=0.5, seed=None):
        """
        Initialize base k-regular k-connected graph.

        Args:
            n (int): Graph scale parameter (has different meanings for different graph types).
            fault_rate (float, optional): Ratio of fault nodes to total nodes in the graph (0.0-1.0).
                - If specified, set fault node count according to this ratio
                - If None, use random number within theoretical diagnosability range
            fault_count (int, optional): Specific number of fault nodes.
                - If specified, set exact number of fault nodes
                - Mutually exclusive with fault_rate, fault_count has higher priority
            intermittent_prob (float): Probability that fault nodes behave as faulty in each test round (intermittent fault probability), default 0.5.
            seed (int): Random seed for reproducibility.
        """
        self.n = n
        self.fault_rate = fault_rate
        self.fault_count = fault_count
        self.intermittent_prob = intermittent_prob
        self.rng = np.random.default_rng(seed)
        
        # fault_count and fault_rate mutual exclusion check
        if fault_count is not None and fault_rate is not None:
            raise ValueError("fault_count and fault_rate cannot be specified simultaneously, please choose one")
        
        # Subclass needs to set self.k value
        self.k = self._get_k_value()
        
        logger.info(f"Graph type: {self.get_graph_type()}, parameter n: {n}, automatically determined regularity k: {self.k}")
        
        # Generate graph
        self.G = self._generate_graph()
        self.num_vertices = self.G.number_of_nodes()
        self.vertices = list(self.G.nodes())
        self.edges = list(self.G.edges())
        
        # Create node to index mapping
        self.node_to_idx = {node: i for i, node in enumerate(self.vertices)}
        
        # Calculate theoretical intermittent fault diagnosability
        self.theoretical_diagnosability = self._calculate_theoretical_diagnosability()
        
        # Assign basic fault states and fault probabilities to nodes
        self.fault_states = None
        self.fault_probs = None
        self.assign_fault_probs()
        self.assign_faults()
    
    @abstractmethod
    def _get_k_value(self):
        """Subclass must implement: determine k value based on graph type and n"""
        pass
    
    @abstractmethod
    def _generate_graph(self):
        """Subclass must implement: generate graph according to specified type"""
        pass
    
    @abstractmethod 
    def get_graph_type(self):
        """Subclass must implement: return graph type name"""
        pass
    
    @abstractmethod
    def _calculate_theoretical_diagnosability(self):
        """Calculate theoretical intermittent fault diagnosability based on formulas in the paper. Each graph type needs to implement its own formula."""
        pass
    
    def assign_fault_probs(self):
        """Assign fault probability to each node (simulating intermittent faults)."""
        # Use intermittent_prob as baseline, add small random fluctuation to each node
        self.fault_probs = self.rng.uniform(
            max(0, self.intermittent_prob - 0.05),
            min(1, self.intermittent_prob + 0.05),
            size=self.num_vertices
        )
        return self.fault_probs
    
    def assign_faults(self):
        """
        Assign fault states (0: no fault, 1: fault).
        
        According to conclusions in the paper, intermittent fault diagnosability of k-regular k-connected graph is k-(g-1)/2-2,
        where g is the maximum number of common neighbors between any two vertices. Number of fault nodes must be less than or equal to this value
        to be correctly diagnosed.
        
        Three modes:
        1. Specified fault_count: Set exact number of fault nodes
        2. Specified fault_rate: Set fault node count according to specified ratio
        3. Neither fault_rate nor fault_count specified: Randomly select fault node count within theoretical diagnosability range
        """
        self.fault_states = np.zeros(self.num_vertices, dtype=int)
        
        if self.fault_count is not None:
            # Mode 1: Use specified fault node count
            actual_fault_count = self.fault_count
            logger.info(f"Mode: Specified fault node count, setting {actual_fault_count} fault nodes "
                  f"(total nodes: {self.num_vertices}, "
                  f"theoretical diagnosability: {self.theoretical_diagnosability})")
            
            # Validate fault node count validity
            if actual_fault_count < 0:
                raise ValueError(f"Fault node count cannot be negnnive: {actual_fault_count}")
            if actual_fault_count > self.num_vertices:
                raise ValueError(f"Fault node count {actual_fault_count} exceeds total node count {self.num_vertices}")
            
            # Give warning if exceeding theoretical diagnosability
            if actual_fault_count > self.theoretical_diagnosability:
                logger.warning(f"Fault node count {actual_fault_count} exceeds theoretical diagnosability {self.theoretical_diagnosability}, "
                      f"may lead to incorrect diagnosis")
        elif self.fault_rate is not None:
            # Mode 2: Use specified fault node ratio
            actual_fault_count = int(self.num_vertices * self.fault_rate)
            logger.info(f"Mode: Specified ratio, setting {actual_fault_count} fault nodes "
                  f"(ratio: {self.fault_rate}, total nodes: {self.num_vertices}, "
                  f"theoretical diagnosability: {self.theoretical_diagnosability})")
            
            # Give warning if exceeding theoretical diagnosability
            if actual_fault_count > self.theoretical_diagnosability:
                logger.warning(f"Fault node count {actual_fault_count} exceeds theoretical diagnosability {self.theoretical_diagnosability}, "
                      f"may lead to incorrect diagnosis")
        else:
            # Mode 3: Use theoretical diagnosability constraint, randomly select between 1 and theoretical diagnosability
            if self.theoretical_diagnosability > 0:
                max_fault_count = min(self.theoretical_diagnosability, self.num_vertices)
                actual_fault_count = self.rng.integers(1, max_fault_count + 1)
                logger.info(f"Mode: Theoretical diagnosability constraint, setting {actual_fault_count} fault nodes "
                      f"(theoretical diagnosability: {self.theoretical_diagnosability}, range: 1-{max_fault_count}, "
                      f"total nodes: {self.num_vertices})")
            else:
                actual_fault_count = 0
                logger.info(f"Mode: Theoretical diagnosability constraint, theoretical diagnosability is 0, cannot set fault nodes")
        
        # Randomly select specified number of nodes as fault nodes
        if actual_fault_count > 0:
            fault_indices = self.rng.choice(self.num_vertices, size=actual_fault_count, replace=False)
            self.fault_states[fault_indices] = 1
        
        return self.fault_states
    
    def generate_syndrome_single_round(self):
        """Generate single round test results (PMC model + intermittent faults)."""
        syndrome = np.zeros((self.num_vertices, self.num_vertices), dtype=int)
        
        for u, v in self.edges:
            # Get indices corresponding to nodes
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            
            # For intermittent faults, fault nodes have probability to behave as normal
            # Only when node's basic state is fault(1), consider intermittent behavior
            u_faulty = self.fault_states[u_idx] == 1 and self.rng.random() < self.fault_probs[u_idx]
            v_faulty = self.fault_states[v_idx] == 1 and self.rng.random() < self.fault_probs[v_idx]
            
            # Generate test results according to PMC model
            if not u_faulty:  # u has no fault
                syndrome[u_idx, v_idx] = 1 if v_faulty else 0
            else:  # u has fault
                syndrome[u_idx, v_idx] = self.rng.choice([0, 1])
            
            if not v_faulty:  # v has no fault
                syndrome[v_idx, u_idx] = 1 if u_faulty else 0
            else:  # v has fault
                syndrome[v_idx, u_idx] = self.rng.choice([0, 1])
        
        return syndrome
    
    def generate_multi_round_syndrome(self, num_rounds=10):
        """Generate multi-round test results."""
        syndromes = []
        for _ in range(num_rounds):
            syndromes.append(self.generate_syndrome_single_round())
        return syndromes
    
    def aggregnne_syndromes_to_proportions(self, syndromes):
        """
        Aggregnne multiple single syndrome matrices into proportion matrix.
        
        Args:
            syndromes: List of multiple single syndrome matrices
            
        Returns:
            np.ndarray: Aggregnned proportion matrix, each element represents proportion of 1s at corresponding position
        """
        if not syndromes:
            raise ValueError("Syndrome list cannot be empty")
        
        # Stack all syndrome matrices
        syndrome_stack = np.stack(syndromes, axis=0)  # shape: (num_rounds, num_vertices, num_vertices)
        
        # Calculate proportion of 1s at each position
        proportion_matrix = np.mean(syndrome_stack, axis=0)  # shape: (num_vertices, num_vertices)
        
        return proportion_matrix
    
    def get_node_features_from_multi_round(self, syndromes):
        """
        Generate node features from multi-round test results.
        
        Features are proportions of test results in multiple rounds that are 1 for each node's neighbors.
        For non-k-regular graphs (e.g., star graph), use zero padding to ensure consistent feature vector length.
        Add small noise to full 0 feature vectors to differentiate different fault-free nodes.
        """
        # First aggregnne multi-round symptoms into proportion matrix
        proportion_matrix = self.aggregnne_syndromes_to_proportions(syndromes)
        
        features = []
        
        # Find the maximum number of neighbors in the graph (for non-regular graphs, e.g., star graph)
        max_neighbors = max(len(list(self.G.neighbors(u))) for u in self.vertices)
        
        for u in self.vertices:
            u_idx = self.node_to_idx[u]
            neighbors = list(self.G.neighbors(u))
            # Initialize feature vector
            node_feature = []
            
            # Use proportion matrix values for each neighbor
            for v in neighbors:
                v_idx = self.node_to_idx[v]
                proportion_1 = proportion_matrix[u_idx, v_idx]
                node_feature.append(proportion_1)
            
            # If neighbor count is less than maximum neighbor count, pad with zeros
            padding_length = max_neighbors - len(neighbors)
            if padding_length > 0:
                node_feature.extend([0.0] * padding_length)
            
            # Add small noise, especially when feature is all zeros
            is_all_zero = all(val == 0.0 for val in node_feature)
            for i in range(len(node_feature)):
                # Add large noise to full 0 features, small noise to other features
                noise_scale = 0.02 if is_all_zero else 0.005
                noise = self.rng.normal(0, noise_scale)
                # Ensure added noise is within [0,1] range
                node_feature[i] = max(0.0, min(1.0, node_feature[i] + noise))
            
            features.append(node_feature)
        
        # Convert to PyTorch tensor
        features = torch.tensor(features, dtype=torch.float)
        return features
    
    def get_edge_index(self):
        """Return edge index (PyTorch Geometric format)."""
        # Use node indices instead of node labels
        edges = [(self.node_to_idx[u], self.node_to_idx[v]) for u, v in self.edges]
        edges.extend([(self.node_to_idx[v], self.node_to_idx[u]) for u, v in self.edges])  # Add reverse edges
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def get_data(self, num_rounds=10):
        """Return graph data: node features, edge index, and fault states."""
        syndromes = self.generate_multi_round_syndrome(num_rounds)
        x = self.get_node_features_from_multi_round(syndromes)
        edge_index = self.get_edge_index()
        y = torch.tensor(self.fault_states, dtype=torch.long)
        return x, edge_index, None, y
    
    def generate_rnn_syndrome_vector(self):
        """
        Generate syndrome vector for RNNIFDCom.
        
        Strictly follows paper design:
        - Input: Comprehensive syndrome vector of a complete graph test
        - Feature transformation: 0 → -0.5, 1 → 0.5
        - Output: Fault state of each node
        
        Returns:
            np.ndarray: Syndrome vector, converted to -0.5/0.5 format as per paper requirements
        """
        # Generate single graph test results
        syndrome_matrix = self.generate_syndrome_single_round()
        
        # Build all "comparison" test results
        # In PMC adaptation, we treat test results of neighboring nodes as comparison results
        all_comparisons = []
        
        # For each edge, extract bidirectional test results
        for u, v in self.edges:
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            
            # In MM model, this is equivalent to comparing u and v states
            # In PMC adaptation, we use XOR of bidirectional test results as "comparison" results
            test_u_to_v = syndrome_matrix[u_idx, v_idx]
            test_v_to_u = syndrome_matrix[v_idx, u_idx]
            
            # Simulate MM model comparison results: If two test results are different, comparison result is 1
            comparison_result = int(test_u_to_v != test_v_to_u)
            all_comparisons.append(comparison_result)
        
        # Convert according to paper requirements: 0 → -0.5, 1 → 0.5
        converted_syndrome = [(x - 0.5) for x in all_comparisons]
        
        return np.array(converted_syndrome, dtype=np.float32)

    def generate_syndrome_single_round_sparse(self):
        """
        Generate single test sparse format syndrome data
        Only store test results of actual edges, not complete matrix
        
        Returns:
            list: [(u, v, test_result), ...] Only store test results of actual edges
        """
        sparse_syndrome = []
        
        for u, v in self.edges:
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            
            # Generate u test v results
            test_u_to_v = self._test_node_pair(u_idx, v_idx)
            # Generate v test u results  
            test_v_to_u = self._test_node_pair(v_idx, u_idx)
            
            # Store bidirectional test results
            sparse_syndrome.append((u, v, test_u_to_v, test_v_to_u))
        
        return sparse_syndrome

    def sparse_to_dense_syndrome(self, sparse_syndrome):
        """
        Convert sparse format to complete matrix format (for compatibility with existing code)
        
        Args:
            sparse_syndrome: Sparse format syndrome data
            
        Returns:
            np.ndarray: Complete syndrome matrix
        """
        syndrome_matrix = np.zeros((self.num_vertices, self.num_vertices), dtype=np.uint8)
        
        for u, v, test_u_to_v, test_v_to_u in sparse_syndrome:
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            syndrome_matrix[u_idx, v_idx] = test_u_to_v
            syndrome_matrix[v_idx, u_idx] = test_v_to_u
        
        return syndrome_matrix

    def get_node_features_from_sparse_syndromes(self, sparse_syndromes, disabled_nodes=None,
                                                global_max_neighbors=None):
        """
        Generate node features from sparse format multi-round syndrome data - optimized version
        
        Args:
            sparse_syndromes: List of multi-round sparse syndrome data
            disabled_nodes: Set of disabled nodes (these nodes do not participate in symptom generation)
            global_max_neighbors: If provided, pad feature vectors to this length
                (required for random graphs like WS where max_degree varies per instance)
            
        Returns:
            torch.tensor: Node feature matrix
        """
        num_rounds = len(sparse_syndromes)
        disabled_nodes = disabled_nodes or set()
        
        edge_stats = {}
        
        for sparse_syndrome in sparse_syndromes:
            for edge_u, edge_v, test_u_to_v, test_v_to_u in sparse_syndrome:
                if edge_u in disabled_nodes or edge_v in disabled_nodes:
                    continue
                    
                key_uv = (edge_u, edge_v)
                if key_uv not in edge_stats:
                    edge_stats[key_uv] = 0
                edge_stats[key_uv] += test_u_to_v
                
                key_vu = (edge_v, edge_u)
                if key_vu not in edge_stats:
                    edge_stats[key_vu] = 0
                edge_stats[key_vu] += test_v_to_u
        
        local_max = max(len(list(self.G.neighbors(u))) for u in self.vertices)
        max_neighbors = max(local_max, global_max_neighbors or 0)
        
        features = []
        
        for u in self.vertices:
            neighbors = list(self.G.neighbors(u))
            node_feature = []
            
            for v in neighbors:
                if v in disabled_nodes:
                    node_feature.append(0.0)
                else:
                    key = (u, v)
                    count_1 = edge_stats.get(key, 0)
                    proportion_1 = count_1 / num_rounds if num_rounds > 0 else 0.0
                    node_feature.append(proportion_1)
            
            while len(node_feature) < max_neighbors:
                node_feature.append(0.0)
            
            if all(f == 0.0 for f in node_feature):
                for i in range(len(node_feature)):
                    node_feature[i] = self.rng.normal(0, 0.01)
            
            features.append(node_feature)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def get_node_features_with_partial_symptoms(self, sparse_syndromes, missing_ratio=0.0, missing_type='node_disable'):
        """
        Generate node features with partial symptoms, for RQ4 experiment
        
        Args:
            sparse_syndromes: List of multi-round sparse syndrome data
            missing_ratio: Missing proportion (0.0-1.0)
            missing_type: Missing type ('node_disable')
            
        Returns:
            tuple: (Node feature matrix, Disabled nodes set)
        """
        if missing_type == 'node_disable':
            # Disable a certain proportion of nodes, skip these nodes when generating symptoms
            disabled_nodes = self._select_disabled_nodes(missing_ratio)
            features = self.get_node_features_from_sparse_syndromes(sparse_syndromes, disabled_nodes)
            return features, disabled_nodes
        else:
            raise ValueError(f"Unknown missing type: {missing_type}")
    
    def _select_disabled_nodes(self, missing_ratio):
        """
        Select nodes to disable
        
        Args:
            missing_ratio: Proportion of nodes to disable
            
        Returns:
            set: Disabled nodes set
        """
        if missing_ratio <= 0:
            return set()
        
        num_to_disable = int(len(self.vertices) * missing_ratio)
        if num_to_disable > 0:
            disabled_indices = self.rng.choice(len(self.vertices), size=num_to_disable, replace=False)
            disabled_nodes = {self.vertices[i] for i in disabled_indices}
            return disabled_nodes
        
        return set()
    
    def generate_sparse_syndromes_with_disabled_nodes(self, num_rounds, disabled_nodes=None):
        """
        Generate sparse syndrome data with disabled nodes
        
        Args:
            num_rounds: Number of rounds to generate
            disabled_nodes: Set of disabled nodes
            
        Returns:
            list: List of sparse syndrome data
        """
        disabled_nodes = disabled_nodes or set()
        syndromes = []
        
        for _ in range(num_rounds):
            sparse_syndrome = []
            
            for u, v in self.edges:
                # Skip edges involving disabled nodes
                if u in disabled_nodes or v in disabled_nodes:
                    continue
                
                u_idx = self.node_to_idx[u]
                v_idx = self.node_to_idx[v]
                
                # Generate test results
                test_u_to_v = self._test_node_pair(u_idx, v_idx)
                test_v_to_u = self._test_node_pair(v_idx, u_idx)
                
                sparse_syndrome.append((u, v, test_u_to_v, test_v_to_u))
            
            syndromes.append(sparse_syndrome)
        
        return syndromes
    
    def _test_node_pair(self, tester_idx: int, tested_idx: int) -> int:
        """
        Execute single node pair test (PMC model + intermittent faults)
        
        Args:
            tester_idx: Index of testing node
            tested_idx: Index of tested node
            
        Returns:
            int: Test result (0 or 1)
        """
        # For intermittent faults, fault nodes have probability to behave as normal
        tester_faulty = (self.fault_states[tester_idx] == 1 and 
                        self.rng.random() < self.fault_probs[tester_idx])
        tested_faulty = (self.fault_states[tested_idx] == 1 and 
                        self.rng.random() < self.fault_probs[tested_idx])
        
        # Generate test results according to PMC model
        if not tester_faulty:  # Testing node has no fault
            return 1 if tested_faulty else 0
        else:  # Testing node has fault
            return self.rng.choice([0, 1]) 