"""
Implementation of the ball pivot Algorithm from...
https://ieeexplore.ieee.org/document/817351
https://pdfs.semanticscholar.org/8ec0/d70299f83ccb98ad593a1b581deb018cbfe2.pdf
"""
from queue import Queue
from typing import Iterable, NamedTuple, Tuple
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

from mesh_algos.mesh_algo import Mesh, MeshAlgo, Point
from mesh_algos.utils import euclidean, set_axes_equal

"""
########     ###    ########    ###        ########  ######## ########
##     ##   ## ##      ##      ## ##       ##     ## ##       ##
##     ##  ##   ##     ##     ##   ##      ##     ## ##       ##
##     ## ##     ##    ##    ##     ##     ##     ## ######   ######
##     ## #########    ##    #########     ##     ## ##       ##
##     ## ##     ##    ##    ##     ##     ##     ## ##       ##
########  ##     ##    ##    ##     ##     ########  ######## ##
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
##     ## ######## ##       ########  ######## ########   ######
##     ## ##       ##       ##     ## ##       ##     ## ##    ##
##     ## ##       ##       ##     ## ##       ##     ## ##
######### ######   ##       ########  ######   ########   ######
##     ## ##       ##       ##        ##       ##   ##         ##
##     ## ##       ##       ##        ##       ##    ##  ##    ##
##     ## ######## ######## ##        ######## ##     ##  ######
"""


def plot_pivot(title, a, b, c, center, radius, nn):
    """
    Plots the geometry of a pivot operation. Displays the ball post pivot
    the points used in the triangle and the neighbors
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    u, v = np.mgrid[0 : 2 * np.pi : 20 * 1j, 0 : np.pi : 10 * 1j]
    sphere_x = center[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = center[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = center[2] + radius * np.cos(v)
    ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="black", alpha=0.5)

    ax.scatter(nn.T[0], nn.T[1], nn.T[2], c="b", alpha=0.5)
    tri = np.stack([b, c])
    ax.scatter(tri.T[0], tri.T[1], tri.T[2], c="orange", alpha=1)
    ax.scatter([a[0]], [a[1]], [a[2]], c="pink", s=30, alpha=1)
    set_axes_equal(ax)
    plt.title(title)
    plt.show()


def calc_circumcircle(A: Point, B: Point, C: Point) -> Tuple[Point, float]:
    """
    Calculate the Circumcircle of the three points
    returns the circumcenter of the circle (np.ndarray[x,y,z]), and the circum radius (float)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
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
    cross = np.cross(a - b, a - c)
    pos_norm = cross / np.linalg.norm(cross)
    neg_norm = -cross / np.linalg.norm(cross)
    # return the normal that points in the same direction as the prev normal
    return pos_norm if (pos_norm @ prev_normal > 0) else neg_norm


"""
########     ###    ##       ##       ########  #### ##     ##  #######  ########
##     ##   ## ##   ##       ##       ##     ##  ##  ##     ## ##     ##    ##
##     ##  ##   ##  ##       ##       ##     ##  ##  ##     ## ##     ##    ##
########  ##     ## ##       ##       ########   ##  ##     ## ##     ##    ##
##     ## ######### ##       ##       ##         ##   ##   ##  ##     ##    ##
##     ## ##     ## ##       ##       ##         ##    ## ##   ##     ##    ##
########  ##     ## ######## ######## ##        ####    ###     #######     ##
"""


class BallPivot(MeshAlgo):
    """
    Algorithm for generating a mesh from a point cloud by rolling a ball along
    the exterior surface of the cloud.
    """

    def __init__(self, ball_radius: float, vis: bool = False):
        """
        Constructor:
            - ball_radius: The radius (m) of the ball that will be used to pivot over the cloud
        """
        self.mesh = Mesh()
        self.rad = ball_radius
        self.cloud = np.array([])
        self.active_edges = Queue()
        self.used_points = set()
        self.front_edges = set()
        self.inactive_edges = set()
        self.nearest_neighbors = np.array([])
        self.vis = vis

    def generate_mesh(self, cloud: np.ndarray) -> Mesh:
        """ Required from @MeshAlgo"""
        # RESET THE STATE VARIABLES
        self.cloud = cloud
        # Queue of Edges
        self.active_edges = Queue()
        # Set of indices of points already added to the mesh
        self.used_points = set()  # Set(int)
        # Set of frozensets each containing the indices of two points the front
        self.front_edges = set()  # Set(fronzenset(int, int))
        # Set of forzensets that are either boundary edges or interior edges
        self.inactive_edges = set()  # Set(frozenset(int, int))
        # Flag indicating whether or not the algorithm has completed
        complete = False

        # Pre-Calculate the Knn for the cloud using a KD tree
        nbrs = NearestNeighbors(
            n_neighbors=20, radius=self.rad * 2, algorithm="ball_tree"
        ).fit(self.cloud)
        _, nearest_neighbors = nbrs.kneighbors(self.cloud)
        # Remove self points from nn
        self.nearest_neighbors = nearest_neighbors[:, 1:]

        while not complete:
            while not self.active_edges.empty():
                print(len(self.used_points))
                edge = self.active_edges.get()
                if frozenset((edge.p1, edge.p2)) not in self.inactive_edges:
                    pivot_result = self.pivot(edge)
                    if (pivot_result) and (
                        (pivot_result[0] not in self.used_points)
                        or (self.in_front(pivot_result[0]))
                    ):
                        pi, center, normal = pivot_result
                        self.add_triangle(edge, pi, center, normal)
                    else:
                        self.inactive_edges.add(frozenset((edge.p1, edge.p2)))

            new_seed = self.find_seed_triangle()
            if new_seed:
                print(len(self.used_points))
                ((p1, p2, p3), ball_center, normal) = new_seed
                self.active_edges.put(Edge(p1, p2, ball_center, normal))
                self.front_edges.add(frozenset((p1, p2)))
                self.used_points.add(p1)
                self.used_points.add(p2)
                self.add_triangle(
                    Edge(p1, p2, ball_center, normal), p3, ball_center, normal
                )
            else:
                complete = True

        return self.mesh

    def in_front(self, pi: int):
        """
        Returns true iff the point index is contained in any of the edges of the front
        """
        return pi in frozenset().union(*list(self.front_edges))

    def get_nn(self, pts: Iterable[int]):
        """
        Takes a iterable of point indices, and returns the indices of their nearest neighbors
        """
        input_pts = np.array(pts)
        matrix_of_neighbors = self.nearest_neighbors[input_pts]
        nn = np.unique(matrix_of_neighbors.flatten())
        nn = np.setdiff1d(nn, input_pts)
        return nn[nn < self.cloud.shape[0]]

    def check_ball_free_of_points(self, p1: int, p2: int, p3: int, center: Point):
        """
        Checks that no point lies within a given ball.
        For efficiency lets check only the points close to the triangle of the ball
        Params:
            p1,p2,p3 the indicies of the points used to calculate the ball center
            center: the center of the ball
        returns true iff the ball is free of points
        """
        nn = self.get_nn((p1, p2, p3))
        distances = cdist(np.array([center]), self.cloud[nn])[0]
        return np.all(distances > self.rad)

    def pivot(self, edge: Edge) -> int:
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
        nn = self.get_nn((edge.p1, edge.p2))

        C = self.cloud[edge.p1]  # Point C in the triangle
        B = self.cloud[edge.p2]  # Point B in the triangle

        # For each nn point calculate the ball if it exists
        bcent_point_pairs = []
        for pi in nn:
            A = self.cloud[pi]
            # Calculate the circum radius To determine if the point can form a triangle
            circumcenter, circumrad = calc_circumcircle(A, B, C)
            if circumrad <= self.rad:
                # Calculate the ball center
                n = calc_triangle_normal(edge.triangle_normal, A, B, C)
                ball_center = circumcenter + np.sqrt(self.rad ** 2 - circumrad ** 2) * n

                bcent_point_pairs.append([pi, ball_center, n])

        # Sort the points in order of distance along he trajectory
        bcent_point_pairs.sort(
            key=lambda bcp, prev=edge.ball_center: euclidean(bcp[1], prev)
        )

        for pi, center, normal in bcent_point_pairs:
            # Check the conditions of for the pivot
            nn = self.get_nn((pi, edge.p1, edge.p2))
            free_of_points = self.check_ball_free_of_points(
                edge.p1, edge.p2, pi, center
            )
            not_the_prev_center = not np.allclose(center, edge.ball_center)
            good_pt = free_of_points and not_the_prev_center
            if self.vis:
                title = "Good Pivot" if good_pt else "Invalid Pivot"
                plot_pivot(
                    title, self.cloud[pi], B, C, center, self.rad, self.cloud[nn]
                )

            if good_pt:
                return pi, center, normal
        return None

    def add_triangle(
        self, edge: Edge, point_index: int, center: np.ndarray, normal: np.ndarray
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
        # Add the triangle to the mesh
        self.mesh.add_triangle(
            (self.cloud[point_index], self.cloud[edge.p1], self.cloud[edge.p2])
        )
        # House keep the tracking information
        self.used_points.add(point_index)

        # For the two new edges check if they need to get added to the front
        for new_edge in [
            frozenset((point_index, edge.p1)),
            frozenset((point_index, edge.p2)),
        ]:

            if new_edge in self.front_edges:
                self.front_edges.remove(new_edge)
                self.inactive_edges.add(new_edge)
            else:
                (p1, p2) = new_edge
                self.front_edges.add(new_edge)
                self.active_edges.put(Edge(p1, p2, center, normal))
        # Remove the previous front edge from the front
        self.front_edges.remove(frozenset((edge.p1, edge.p2)))
        self.inactive_edges.add(frozenset((edge.p1, edge.p2)))

    def find_seed_triangle(self) -> Tuple[Tuple[int, int, int], Point, float]:
        """
        Finds a seed triangle from the cloud consisting of points that are not yet used

        Params:
            cloud: The entire point cloud
            used_points: The indices of the points that have already been added to the mesh

        Returns:
            (p1,p2,p3) The points of the triangle OR None
        """
        unused_pts = list(set(list(range(len(self.cloud)))) - self.used_points)
        random.shuffle(unused_pts)
        # Check all of our unsed points
        for p in unused_pts:
            # Look at their two closest neighbors
            n1, n2 = self.nearest_neighbors[p][0:2]
            if n1 < len(self.cloud) and n2 < len(self.cloud):
                circumcenter, circumrad = calc_circumcircle(
                    self.cloud[p], self.cloud[n1], self.cloud[n2]
                )
                if circumrad < self.rad:
                    centroid = np.mean(self.cloud, axis=0)
                    cent_to_mid = circumcenter - centroid
                    normal = calc_triangle_normal(
                        cent_to_mid, self.cloud[p], self.cloud[n1], self.cloud[n2]
                    )
                    ball_center = (
                        circumcenter + np.sqrt(self.rad ** 2 - circumrad ** 2) * normal
                    )

                    if self.check_ball_free_of_points(p, n1, n2, ball_center):
                        return ((p, n1, n2), ball_center, normal)

        return None
