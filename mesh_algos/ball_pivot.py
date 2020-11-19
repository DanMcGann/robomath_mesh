"""
Implementation of the ball pivot Algorithm from...
https://ieeexplore.ieee.org/document/817351
https://pdfs.semanticscholar.org/8ec0/d70299f83ccb98ad593a1b581deb018cbfe2.pdf
"""
import numpy as np
from queue import Queue
from typing import NamedTuple, Tuple
from mesh_algos.mesh_algo import MeshAlgo, Point, Mesh
from mesh_algos.utils import euclidean, midpoint
from scipy.spatial import KDTree

"""
Data Definitions
"""

""" EDGE
Represents an edge + metadata in the mesh used to pivot around the the algorithm
Each p1,p2 the index of a point in the cloud
The ball center is the center of the ball used to create the triangle of which this edge is apart
And the triangle_normal is the normal for the triangle of which this edge is apart
"""
Edge = NamedTuple(
    "Edge",
    [
        ("p1", int),
        ("p2", int),
        ("ball_center", np.ndarray),
        ("triangle_normal", np.ndarray),
    ],
)

""" NeareastNeighborMatrix
A ndarray where the ith row lists the indicies of the nearest neighbors of the ith point in the cloud
Note: If there are not k nearest neighbors the row will be padded with values equal to len(cloud)
"""


"""
Helper Functions
"""


def calc_circumcircle(A: Point, B: Point, C: Point) -> Tuple[Point, float]:
    """
    Calculate the Circumcircle of the three points
    returns the circumcenter of the circle (np.ndarray[x,y,z]), and the circum radius (float)
    """
    a, b, c = euclidean(B, C), euclidean(A, C), euclidean(A, B)
    circumrad = np.sqrt(
        ((a * b * c) ** 2) / ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))
    )
    ha = (a ** 2) * (b ** 2 + c ** 2 - a ** 2)
    hb = (b ** 2) * (c ** 2 + a ** 2 - b ** 2)
    hc = (c ** 2) * (a ** 2 + b ** 2 - c ** 2)
    circumcenter = (ha * A + hb * B + hc * C) / (ha + hb + hc)
    return circumcenter, circumrad


def calc_triangle_normal(
    prev_normal: np.ndarray, a: Point, b: Point, c: Point
) -> np.ndarray:
    """
    Calculate the normal for a triangle based its points and the normal of the previous triangle
    """
    # From two sides of the triangle we can get two potential normals
    pos_norm = np.cross(a - b, a - c)
    neg_norm = -pos_norm
    # return the normal that points in the same direction as the prev normal
    return pos_norm if (pos_norm @ prev_normal > 0) else neg_norm


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

    def generate_mesh(self, cloud: np.ndarray) -> Mesh:
        """ Required from @MeshAlgo"""
        # The mesh that we are generating
        mesh = Mesh()
        # Queue of Edges
        active_edges = Queue()
        # Set of indices of points already added to the mesh
        used_points = set()  # Set(int)
        # Set of frozensets each containing the indices of two points the front
        front_edges = set()  # Set(fronzenset(int, int))
        # Set of forzensets that are either boundary edges or interior edges
        inactive_edges = set()  # Set(frozenset(int, int))
        # Flag indicating whether or not the algorithm has completed
        complete = False

        # Pre-Calculate the Knn for the cloud using a KD tree
        kdtree = KDTree(cloud)
        _, nearest_neighbors = kdtree.query(cloud, k=10)
        # Remove self points from nn
        nearest_neighbors = nearest_neighbors[:, 1:]

        while not complete:
            while not active_edges.empty():
                edge = active_edges.get()
                if edge not in inactive_edges:
                    third_point_idx = self.pivot(cloud, nearest_neighbors, edge)

                    if (third_point_idx) and (
                        (third_point_idx not in used_points)
                        or (self.in_front(third_point_idx, front_edges))
                    ):
                        self.add_triangle(
                            mesh, cloud, edge, third_point_idx, front_edges, used_points
                        )
                    else:
                        inactive_edges.add(edge)

            new_seed = self.find_seed_triangle(cloud, used_points)
            if new_seed:
                p1, p2, p3 = new_seed
                self.add_triangle(
                    mesh, cloud, Edge(p1, p2), p3, used_points, front_edges
                )
                # Add the edges to the queue
                active_edges.put(Edge(p1, p2))  # TODO ADD CENTER HERE
                active_edges.put(Edge(p1, p3))
                active_edges.put(Edge(p2, p3))
                # Add the edges to the front
                front_edges.add(Edge(p1, p2))
                front_edges.add(Edge(p1, p3))
                front_edges.add(Edge(p2, p3))
            else:
                complete = True

        return mesh

    def in_front(self, pi: int, front: set):
        """
        Returns true iff the point index is contained in any of the edges of the front
        """
        return pi in frozenset().union(*list(front))

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
        # Get the indices of the nearest neighbors
        nn = nearest_neighbors[edge.p1] + nearest_neighbors[edge.p2]
        # Remove indies that are too large (filler in the nn matrix)
        nn = [n for n in nn if n == len(cloud)]

        C = cloud[edge.p1]  # Point C in the triangle
        B = cloud[edge.p2]  # Point B in the triangle

        # For each nn point calculate the ball if it exists
        bcent_point_pairs = []
        for pi in nn:
            A = cloud[pi]
            # Calculate the circum radius To determine if the point can form a triangle
            circumcenter, circumrad = calc_circumcircle(A, B, C)
            if circumrad <= self._rad:
                # Calculate the ball center
                n = np.array([0, 0, 1])  # TODO figure out how to get normal vector
                ball_center = (
                    circumcenter + np.sqrt(self._rad ** 2 - circumrad ** 2) @ n
                )

                bcent_point_pairs.append([pi, ball_center])

        # Lets sort all of the potential balls by their position along the gamma trajectory
        midpt = midpoint(B, C)
        # First look at centers such that (c-m) points in the same direction as the previous normal
        upper_pairs = [
            pair
            for pair in bcent_point_pairs
            if ((pair[1] - midpt) @ edge.prev_normal) > 0
        ]
        # Then sort them based on the distance from the center to prev point
        upper_pairs.sort(key=lambda p, prev=edge.ball_center: euclidean(p[1], prev))
        # Next look at the centers such that the (c-m) vector points in the opposite direction as the prev normal
        lower_pairs = [
            pair
            for pair in bcent_point_pairs
            if ((pair[1] - midpt) @ edge.prev_normal) <= 0
        ]
        # Sort these lower pairs off of the reverse distance
        lower_pairs.sort(
            key=lambda p, prev=edge.ball_center: euclidean(p[1], prev), reverse=True
        )

        # Iterate over the ordered pairs, return the first one that doesnt violate any condtions
        for pi, center in upper_pairs + lower_pairs:
            distances = [
                euclidean(cloud[n], center)
                for n in nn
                if (n != edge.p1) and (n != edge.p2) and (n != pi)
            ]
            if np.all(np.array(distances) > self._rad):
                return pi
        return None

    def add_triangle(
        self,
        mesh: int,
        cloud: np.ndarray,
        edge: Edge,
        point_index: int,
        front: set,
        used_pts: set,
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
