from .bc_network import BCNetwork
from .augmented_k_ary_n_cube import AugmentedKAryNCube
from .watts_strogatz import WattsStrogatzGraph

class GraphFactory:
    """Graph factory class for creating corresponding graph instances based on graph type"""
    
    # Move graph class dictionary to class level to avoid repeated creation on each call
    _simple_graph_classes = {
        'bc': BCNetwork
    }
    
    _k_param_graph_classes = {
        'augmented_k_ary_n_cube': AugmentedKAryNCube
    }
    
    # Irregular graphs: Watts-Strogatz (n nodes, k neighbors, p rewiring prob)
    _ws_param_graph_classes = {
        'watts_strogatz': WattsStrogatzGraph
    }
    
    _all_types = (list(_simple_graph_classes.keys()) +
                  list(_k_param_graph_classes.keys()) +
                  list(_ws_param_graph_classes.keys()))
    
    @staticmethod
    def create_graph(graph_type, n, k=None, p=None, fault_rate=None, fault_count=None,
                    intermittent_prob=0.5, seed=None):
        """
        Create corresponding graph instance based on graph type
        
        Args:
            graph_type (str): Graph type
            n (int): Graph scale parameter (n_ws=node count for watts_strogatz)
            k (int, optional): k_ws for watts_strogatz (neighbors), base for k-ary cube
            p (float, optional): p_ws for watts_strogatz (rewiring prob, default 0.1)
            fault_rate (float, optional): Fault node ratio
            fault_count (int, optional): Number of fault nodes
            intermittent_prob (float): Intermittent fault probability
            seed (int): Random seed
            
        Returns:
            BaseKRegularGraph: Graph instance of corresponding type
        """
        if graph_type not in GraphFactory._all_types:
            valid_types = ', '.join(GraphFactory._all_types)
            raise ValueError(f"Unsupported graph type: {graph_type}. Supported types: {valid_types}")
        
        # Create corresponding graph instance
        if graph_type in GraphFactory._simple_graph_classes:
            graph_class = GraphFactory._simple_graph_classes[graph_type]
            return graph_class(n, fault_rate, fault_count, intermittent_prob, seed)
        elif graph_type in GraphFactory._k_param_graph_classes:
            graph_class = GraphFactory._k_param_graph_classes[graph_type]
            if k is None:
                if graph_type == 'k_ary_n_cube':
                    k = 2
                elif graph_type == 'augmented_k_ary_n_cube':
                    k = 3
            return graph_class(n, k, fault_rate, fault_count, intermittent_prob, seed)
        elif graph_type in GraphFactory._ws_param_graph_classes:
            graph_class = GraphFactory._ws_param_graph_classes[graph_type]
            n_ws = n
            k_ws = k if k is not None else 4  # default: 4 nearest neighbors
            p_ws = p if p is not None else 0.1  # default: small-world regime
            return graph_class(n_ws, k_ws, p_ws, fault_rate, fault_count, intermittent_prob, seed)
    
    @staticmethod
    def get_supported_types():
        """Return list of supported graph types"""
        return GraphFactory._all_types.copy() 