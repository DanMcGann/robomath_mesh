"""
Implementation of the ball pivot Algorithm from...
https://ieeexplore.ieee.org/document/817351
https://pdfs.semanticscholar.org/8ec0/d70299f83ccb98ad593a1b581deb018cbfe2.pdf
"""
import numpy as np
from queue import Queue
from typing import NamedTuple
from mesh_algos.mesh_algo import MeshAlgo, Point
from scipy.spatial import KDTree

"""
Data Definitions
"""
# Represents an edge in the mesh used to pivot around the the algorithm
# Each point the index of a point in the cloud
Edge = NamedTuple("Edge", [("p1", np.ndarray), ("p2", np.ndarray)])



# NeareastNeighborMatrix
# A ndarray where the ith row lists the indicies of the nearest neighbors of the ith point in the cloud
# Note: If there are not k nearest neighbors the row will be padded with values equal to len(cloud)


class BallPivot(MeshAlgo):
    """
    Algorithm for generating a mesh from a point cloud by rolling a ball along
    the exterior surface of the cloud.
    """

    def __init__(self, ball_radius: float):
        """
        Constructor:
            - ball_radius: The radius (m) of the ball that will be used to pivot over the cloud
        """
        self._rad = ball_radius

    def generate_mesh(self, cloud: np.ndarray) -> int:
        """ Required from @MeshAlgo"""
        # The mesh that we are generating
        mesh = 1  # TODO add type
        # Queue of Edges
        active_edges = Queue()
        # Set of indices of points already added to the mesh
        used_points = set()
        # Flag indicating whether or not the algorithm has completed
        complete = False

        # Pre-Calculate the Knn for the cloud using a KD tree
        kdtree = KDTree(cloud)
        _, nearest_neighbors = kdtree.query(cloud, k=10)
        # Remove self points from nn
        nearest_neighbors = nearest_neighbors[:,1:]

        while not complete:
            while not active_edges.empty():
                edge = active_edges.get()
                third_point = self.pivot(cloud, nearest_neighbors, edge)
                # TODO look into whether or not we have to worry about the "front"
                if third_point and third_point not in used_points:
                    self.add_triangle(mesh, cloud, edge, third_point)
                else:
                    pass
                    # self.mark_as_boundary(edge) TODO Figure out how this is necessary

            new_seed = self.find_seed_triangle(cloud, used_points)
            if new_seed:
                p1, p2, p3 = new_seed
                self.add_triangle(mesh, cloud, Edge(p1, p2), p3)
                active_edges.put(Edge(p1, p2))
                active_edges.put(Edge(p1, p3))
                active_edges.put(Edge(p2, p3))
            else:
                complete = True

        return mesh

    def pivot(
        self, cloud: np.ndarray, nearest_neighbors: np.ndarray, edge: Edge
    ) -> int:
        """
        Pivot Operation. Pivots the "ball" around the given edge
        and returns the first point that contacts the ball such that the ball contains
        no other points.
        Params:
            cloud: The point cloud
            nearest_neighbors: A neareast neighbors matrix
            edge: The edge around which to pivot
        Returns:
            The index of the first good point if it exists or None
        """
        nn = nearest_neighbors[]
        return np.array([1, 2, 3])  # TODO

    def add_triangle(
        self, mesh: int, cloud: np.ndarray, edge: Edge, point: int
    ) -> None:
        """
        Adds a triangle constructed from Edge and point to the mesh

        Params:
            mesh: The generated mesh so far
            cloud
            edge: Edge containing the indices of two points of the triangle
                    Assumed to already be in the mesh unless the mesh is otherwise empty
            point: The index of the third point of the triangle

        Returns:
            nothing
        """
        pass  # TODO

    def find_seed_triangle(self, cloud: np.ndarray, used_points: set):
        """
        Finds a seed triangle from the cloud consisting of points that are not yet used

        Params:
            cloud: The entire point cloud
            used_points: The indices of the points that have already been added to the mesh

        Returns:
            (p1,p2,p3) The points of the triangle OR None
        """
        return (1, 2, 3)  # TODO
