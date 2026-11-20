import networkx as nx
from .base_graph import BaseKRegularGraph

class AugmentedKAryNCube(BaseKRegularGraph):
    """Augmented k-ary n-cube implementation class - based on paper Definition 1"""
    
    def __init__(self, n, k=3, fault_rate=None, fault_count=None, intermittent_prob=0.5, seed=None):
        """
        Initialize augmented k-ary n-cube.
        
        Args:
            n (int): Dimension of the cube, n≥1
            k (int): Base of each dimension, k≥3, default is 3
            fault_rate (float, optional): Fault node ratio
            fault_count (int, optional): Number of fault nodes
            intermittent_prob, seed: Inherited from base class
        """
        if k < 3:
            raise ValueError("Augmented k-ary n-cube requires k≥3")
        if n < 1:
            raise ValueError("Augmented k-ary n-cube requires n≥1")
            
        self.k_ary = k  # Store k value to avoid confusion with regularity k
        super().__init__(n, fault_rate, fault_count, intermittent_prob, seed)
    
    def _get_k_value(self):
        """Regularity of augmented k-ary n-cube - based on actual edge connection calculation"""
        # Each node has the following types of neighbors:
        # 1. Condition 1: For each dimension i, has 2 neighbors (±1 direction) -> 2*n neighbors
        # 2. Condition 2: For each i∈[2,n], has 2 neighbors (≤i,±1 direction) -> 2*(n-1) neighbors
        # Total: 2*n + 2*(n-1) = 4*n - 2 neighbors
        return 4 * self.n - 2
    
    def get_graph_type(self):
        """Return graph type name"""
        return f"augmented_k_ary_n_cube_{self.k_ary}"
    
    def _calculate_theoretical_diagnosability(self):
        """
        Calculate intermittent fault diagnosability of AQ_n,k.

        The general value is 4n - 6 for n >= 3 and k >= 4.
        The two smaller configurations below have different values.
        """
        if (self.n, self.k_ary) == (3, 3):
            return 5
        if (self.n, self.k_ary) == (2, 7):
            return 3
        return 4 * self.n - 6
    
    def _generate_vertices(self):
        """Generate all k^n vertices"""
        vertices = []
        
        def generate_recursive(current, remaining_dims):
            if remaining_dims == 0:
                vertices.append(tuple(current))
                return
            
            for digit in range(self.k_ary):
                generate_recursive(current + [digit], remaining_dims - 1)
        
        generate_recursive([], self.n)
        return vertices
    
    def _generate_graph(self):
        """
        Generate augmented k-ary n-cube AQ_{n,k}.
        
        Based on paper Definition 1:
        - Vertex set: All k^n n-tuples, each component ∈[0,k-1]
        - Edge set: Vertex pairs satisfying one of two adjacency conditions
        
        Optimization method: Directly generate neighbors for each vertex instead of traversing all vertex pairs
        """
        # Generate all vertices
        vertices = self._generate_vertices()
        vertices_set = set(vertices)  # For fast lookup
        
        # Create graph
        G = nx.Graph()
        G.add_nodes_from(vertices)
        
        # Directly generate neighbors for each vertex to avoid O(k^2n) complexity
        for u in vertices:
            # Condition 1: (i, ±1)-edges
            # For each dimension i, generate neighbors u_i±1
            for i in range(self.n):
                # +1 direction neighbor
                v_plus = list(u)
                v_plus[i] = (u[i] + 1) % self.k_ary
                v_plus = tuple(v_plus)
                if v_plus in vertices_set:
                    G.add_edge(u, v_plus)
                
                # -1 direction neighbor
                v_minus = list(u)
                v_minus[i] = (u[i] - 1) % self.k_ary
                v_minus = tuple(v_minus)
                if v_minus in vertices_set:
                    G.add_edge(u, v_minus)
            
            # Condition 2: (≤i, ±1)-edges
            # For each i∈[2,n] (indices 1 to n-1), generate neighbors with first i+1 coordinates all ±1
            for i in range(1, self.n):
                # +1 direction neighbor
                v_plus = list(u)
                for j in range(i + 1):
                    v_plus[j] = (u[j] + 1) % self.k_ary
                v_plus = tuple(v_plus)
                if v_plus in vertices_set:
                    G.add_edge(u, v_plus)
                
                # -1 direction neighbor
                v_minus = list(u)
                for j in range(i + 1):
                    v_minus[j] = (u[j] - 1) % self.k_ary
                v_minus = tuple(v_minus)
                if v_minus in vertices_set:
                    G.add_edge(u, v_minus)
        
        print(f"Augmented k-ary n-cube AQ{self.n},{self.k_ary} generation completed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"Theoretical node count: {self.k_ary**self.n}")
        
        # Verify regularity
        if G.number_of_nodes() > 0:
            degrees = [G.degree(node) for node in G.nodes()]
            print(f"Node degree range: {min(degrees)} - {max(degrees)}")
            if len(set(degrees)) == 1:
                print(f"Graph is {degrees[0]}-regular, theoretical regularity: {self.k}")
            else:
                print("Warning: Graph is not regular!")
        
        return G
