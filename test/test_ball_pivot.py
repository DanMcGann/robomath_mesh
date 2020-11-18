"""
Unit tests for the ball pivot algorithm.
"""
import unittest
import numpy as np
import sys
import os
from scipy.spatial import KDTree

from mesh_algos.ball_pivot import BallPivot, Edge


class TestBallPivotFunctions(unittest.TestCase):

    """
    Test Pivoting
    """

    def test_pivot_simple(self):
        bp = BallPivot(1)
        cloud = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        e = Edge(cloud[0], cloud[1])
        kdtree = KDTree(cloud)
        _, nearest_neighbors = kdtree.query(cloud, k=10)
        nearest_neighbors = nearest_neighbors[:, 1:]
        np.testing.assert_equal(bp.pivot(cloud, nearest_neighbors, e), cloud[2])

    def test_pivot_no_solution(self):
        bp = BallPivot(1)
        cloud = np.array([[0, 0, 0], [1, 0, 0], [0, 3, 0]])
        e = Edge(cloud[0], cloud[1])
        kdtree = KDTree(cloud)
        _, nearest_neighbors = kdtree.query(cloud, k=10)
        nearest_neighbors = nearest_neighbors[:, 1:]
        self.assertIsNone(bp.pivot(cloud, nearest_neighbors, e))

    def test_pivot_two_solutions(self):
        bp = BallPivot(1)
        cloud = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 0]])
        e = Edge(cloud[0], cloud[1])
        kdtree = KDTree(cloud)
        _, nearest_neighbors = kdtree.query(cloud, k=10)
        nearest_neighbors = nearest_neighbors[:, 1:]
        np.testing.assert_equal(bp.pivot(cloud, nearest_neighbors, e), cloud[3])
