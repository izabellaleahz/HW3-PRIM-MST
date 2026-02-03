import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    assert adj_mat.shape == mst.shape, 'MST shape must match input adjacency matrix'
    assert np.allclose(mst, mst.T), 'MST must be symmetric'
    assert np.allclose(np.diag(mst), 0), 'MST must have zero diagonal'

    # MST should only use edges that exist in original graph
    invalid_edges = (mst > 0) & (adj_mat == 0)
    assert not np.any(invalid_edges), 'MST includes edges not in original graph'

    # MST should have exactly n-1 edges for a connected graph
    num_nodes = mst.shape[0]
    edge_count = np.sum(np.triu(mst > 0, k=1))
    assert edge_count == num_nodes - 1, 'MST must have exactly n-1 edges'

    # MST should be connected
    if num_nodes > 0:
        visited = set([0])
        stack = [0]
        while stack:
            node = stack.pop()
            neighbors = np.where(mst[node] > 0)[0]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        assert len(visited) == num_nodes, 'MST must be connected'

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """
    adj_mat = np.array([
        [0, 3, 0, 2, 0],
        [3, 0, 1, 0, 4],
        [0, 1, 0, 5, 6],
        [2, 0, 5, 0, 7],
        [0, 4, 6, 7, 0],
    ], dtype=float)
    g = Graph(adj_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 10)


def test_mst_student_triangle():
    """
    
    Additional unit test for MST construction on a triangle graph.
    
    """
    adj_mat = np.array([
        [0, 2, 3],
        [2, 0, 1],
        [3, 1, 0],
    ], dtype=float)
    g = Graph(adj_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 3)
