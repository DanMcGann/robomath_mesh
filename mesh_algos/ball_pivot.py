"""
Implementation of the ball pivot Algorithm from...
https://ieeexplore.ieee.org/document/817351
"""
import numpy as np
from queue import Queue
from collections import Set
from typing import NamedTuple
from mesh_algos.mesh_algo import MeshAlgo

"""
Data Definitions
"""
# Represents an edge in the mesh used to pivot around the the algorithm
# Each point is a [x,y,z] vector
Edge = NamedTuple("Edge", [("p1", np.ndarray), ("p2", np.ndarray)])

# A point in the cloud of form [x,y,z]
Point = np.ndarray


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
        # Set of Points
        used_points = Set()
        # Flag indicating whether or not the algorithm has completed
        complete = False

        while not complete:
            while not active_edges.empty():
                edge = active_edges.get()
                third_point = self.pivot(cloud, edge)
                # TODO look into whether or not we have to worry about the "front"
                if third_point and third_point not in used_points:
                    self.add_triangle(mesh, edge, third_point)
                else:
                   # self.mark_as_boundary(edge) TODO Figure out how this is necessary

            new_seed = self.find_seed_triangle(cloud, used_points)
            if new_seed:
                p1, p2, p3 = new_seed
                self.add_triangle(mesh, Edge(p1, p2), p3)
                active_edges.put(Edge(p1, p2))
                active_edges.put(Edge(p1, p3))
                active_edges.put(Edge(p2, p3))
            else:
                complete = True

        return mesh

    def pivot(self, cloud: np.ndarray, edge: Edge) -> Point:
        """
        Pivot Operation. Pivots the "ball" around the given edge
        and returns the first point that contacts the ball such that the ball contains
        no other points.
        Params:
            cloud: The point cloud
            edge: The edge around which to pivot
        Returns:
            The first good point if it exists or None
        """
        return np.array([1, 2, 3])  # TODO

    def add_triangle(self, mesh: int, edge: Edge, point: Point) -> None:
        """
        Adds a triangle constructed from Edge and point to the mesh

        Params:
            mesh: The generated mesh so far
            edge: Edge containing two points of the triangle
                    Assumed to already be in the mesh unless the mesh is otherwise empty
            point: The third point of the triangle

        Returns:
            nothing
        """
        pass  # TODO

    def find_seed_triangle(self, cloud: np.ndarray, used_points: Set):
        """
        Finds a seed triangle from the cloud consisting of points that are not yet used

        Params:
            cloud: The entire point cloud
            used_points: The points that have already been added to the mesh

        Returns:
            (p1,p2,p3) The points of the triangle OR None
        """
        return (1, 2, 3)  # TODO


