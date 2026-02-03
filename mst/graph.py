import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        self.mst = None
        num_nodes = self.adj_mat.shape[0]
        if num_nodes == 0:
            self.mst = np.zeros((0, 0))
            return

        mst = np.zeros_like(self.adj_mat)
        visited = set([0])
        edge_heap = []

        for neighbor in range(num_nodes):
            weight = self.adj_mat[0, neighbor]
            if neighbor != 0 and weight > 0:
                heapq.heappush(edge_heap, (weight, 0, neighbor))

        while edge_heap and len(visited) < num_nodes:
            weight, u, v = heapq.heappop(edge_heap)
            if v in visited:
                continue
            mst[u, v] = weight
            mst[v, u] = weight
            visited.add(v)

            for neighbor in range(num_nodes):
                if neighbor in visited:
                    continue
                next_weight = self.adj_mat[v, neighbor]
                if next_weight > 0:
                    heapq.heappush(edge_heap, (next_weight, v, neighbor))

        if len(visited) != num_nodes:
            raise ValueError('Input graph must be connected')

        self.mst = mst
