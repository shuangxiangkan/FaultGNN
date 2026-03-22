"""
Watts-Strogatz small-world graph for intermittent fault diagnosis.

Watts-Strogatz graphs are irregular (non-regular) networks that exhibit
small-world properties: high clustering coefficient and short average path length.
Unlike hypercubes and k-ary n-cubes, node degrees vary after the rewiring process.

Parameter naming (aligned with hypercube for comparable experiments):
- Hypercube: n = dimension, nodes = 2^n, degree = n
- Watts-Strogatz: k_ws = scale param, n_ws = 2^k_ws (like hypercube), p_ws = 0.1 fixed

Reference: Watts, D.J. & Strogatz, S.H. (1998). "Collective dynamics of
small-world networks." Nature, 393, 440-442.
"""

import networkx as nx
from .base_graph import BaseKRegularGraph


class WattsStrogatzGraph(BaseKRegularGraph):
    """
    Watts-Strogatz small-world graph (irregular topology).

    Parameters (named to distinguish from hypercube):
        n_ws: Total number of nodes (unlike hypercube where n = dimension)
        k_ws: Each node connected to k_ws nearest neighbors in ring (even, k_ws < n_ws)
        p_ws: Rewiring probability (0 = regular ring, 1 = random). Typical: 0.1 for small-world
        fault_rate, fault_count, intermittent_prob, seed: Inherited from base class
    """

    def __init__(self, n_ws, k_ws=4, p_ws=0.1, fault_rate=None, fault_count=None,
                 intermittent_prob=0.5, seed=None):
        if k_ws >= n_ws:
            raise ValueError(f"Watts-Strogatz requires k_ws < n_ws, got k_ws={k_ws}, n_ws={n_ws}")
        if k_ws % 2 != 0:
            raise ValueError(f"Watts-Strogatz requires even k_ws for ring construction, got k_ws={k_ws}")
        if not 0 <= p_ws <= 1:
            raise ValueError(f"Rewiring probability p_ws must be in [0, 1], got p_ws={p_ws}")

        self.n_ws = n_ws
        self.k_ws = k_ws
        self.p_ws = p_ws
        # Base class expects n (scale param); for WS, n_ws is node count
        super().__init__(n_ws, fault_rate, fault_count, intermittent_prob, seed)

    def _get_k_value(self):
        """
        Return nominal degree for compatibility. Watts-Strogatz is irregular;
        we use k_ws (initial ring degree) as the nominal value for logging.
        """
        return self.k_ws

    def get_graph_type(self):
        return f"watts_strogatz_k{self.k_ws}_p{self.p_ws:.2f}"

    def _calculate_theoretical_diagnosability(self):
        """
        Watts-Strogatz has no known closed-form intermittent fault diagnosability.
        Use conservative heuristic: min_degree - 1 (each node needs at least one
        fault-free neighbor to be correctly tested). Floor at 1 for validity.
        """
        degrees = [self.G.degree(v) for v in self.vertices]
        min_degree = min(degrees)
        return max(1, min_degree - 1)

    def _generate_graph(self):
        """Generate connected Watts-Strogatz small-world graph."""
        ws_seed = int(self.rng.integers(0, 2**31 - 1)) if self.rng is not None else None
        G = nx.connected_watts_strogatz_graph(
            self.n_ws, self.k_ws, self.p_ws,
            tries=100, seed=ws_seed
        )
        return G
