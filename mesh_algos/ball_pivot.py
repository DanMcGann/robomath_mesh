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
from scipy.spatial import cKDTree as KDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

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


def set_axes_equal(ax):
    # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_pivot(title, a, b, c, center, radius, nn):
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
        self.mesh = Mesh()
        self._rad = ball_radius
        self.cloud = np.array([])
        self.active_edges = Queue()
        self.used_points = set()
        self.front_edges = set()
        self.inactive_edges = set()
        self.nearest_neighbors = np.array([])

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
            n_neighbors=20, radius=self._rad * 2, algorithm="ball_tree"
        ).fit(self.cloud)
        _, nearest_neighbors = nbrs.kneighbors(self.cloud)
        # Remove self points from nn
        self.nearest_neighbors = nearest_neighbors[:, 1:]

        while not complete:
            while not self.active_edges.empty():
                edge = self.active_edges.get()
                print("Edge: {}".format(edge))
                if frozenset((edge.p1, edge.p2)) not in self.inactive_edges:
                    pivot_result = self.pivot(edge)
                    if (pivot_result) and (
                        (pivot_result[0] not in self.used_points)
                        or (self.in_front(pivot_result[0]))
                    ):
                        print("Found Pivot: {}".format(pivot_result))
                        pi, center, normal = pivot_result
                        self.add_triangle(edge, pi, center, normal)
                    else:
                        print(
                            "Found Pivot {} is in the used points or the not in the front edges".format(
                                pivot_result[0] if pivot_result else None
                            )
                        )
                        self.inactive_edges.add(frozenset((edge.p1, edge.p2)))
                print()
                # complete = True

            new_seed = self.find_seed_triangle()
            if new_seed:
                (p1, p2, p3), circumcenter, circumrad = new_seed
                print("NEW SEED")
                # Calc the Normal have it point away from the centroid
                centroid = np.mean(cloud, axis=0)
                cent_to_mid = circumcenter - centroid
                normal = calc_triangle_normal(
                    cent_to_mid, cloud[p1], cloud[p2], cloud[p3]
                )
                ball_center = (
                    circumcenter + np.sqrt(self._rad ** 2 - circumrad ** 2) * normal
                )

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

    def check_ball_free_of_points(self, p1: int, p2: int, p3: int, center: np.ndarray):
        nn = list(
            set(
                np.concatenate(
                    (
                        self.nearest_neighbors[p1],
                        self.nearest_neighbors[p2],
                        self.nearest_neighbors[p3],
                    )
                )
            )
        )
        nn = [n for n in nn if n < len(self.cloud)]

        """ PLOT NEAREST NEIGHBORS AND NOT NEAREST NEIGHBORS """
        """
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        not_neighbors = list(set(list(range(len(self.cloud)))) - set(nn))
        nn_pts = self.cloud[nn]
        not_nn_pts = self.cloud[not_neighbors]
        ax.scatter(nn_pts.T[0], nn_pts.T[1], nn_pts.T[2], c="red")
        ax.scatter(not_nn_pts.T[0], not_nn_pts.T[1], not_nn_pts.T[2], c="blue")
        triangle = np.array([p1, p2, p3, p1])
        triangle = self.cloud[triangle]
        ax.plot(triangle.T[0], triangle.T[1], triangle.T[2], c="pink")
        set_axes_equal(ax)
        plt.show()
        """
        distances = [
            euclidean(self.cloud[n], center)
            for n in nn
            if (n != p1) and (n != p2) and (n != p3)
        ]

        return np.all(np.array(distances) > self._rad)

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
        print("\nSTART OF PIVOT:")
        # Get the indices of the nearest neighbors
        nn = np.concatenate(
            (self.nearest_neighbors[edge.p1], self.nearest_neighbors[edge.p2])
        )
        # Remove indies that are too large (filler in the nn matrix)
        nn = [n for n in nn if n < len(self.cloud)]

        C = self.cloud[edge.p1]  # Point C in the triangle
        B = self.cloud[edge.p2]  # Point B in the triangle

        # For each nn point calculate the ball if it exists
        bcent_point_pairs = []
        for pi in nn:
            A = self.cloud[pi]
            # Calculate the circum radius To determine if the point can form a triangle
            circumcenter, circumrad = calc_circumcircle(A, B, C)
            if circumrad <= self._rad:
                # Calculate the ball center
                n = calc_triangle_normal(edge.triangle_normal, A, B, C)
                ball_center = (
                    circumcenter + np.sqrt(self._rad ** 2 - circumrad ** 2) * n
                )

                bcent_point_pairs.append([pi, ball_center, n])

        print("Number of pairs: {}".format(len(bcent_point_pairs)))
        bcent_point_pairs.sort(
            key=lambda bc, prev=edge.ball_center: euclidean(bc[1], prev)
        )
        midpt = midpoint(B, C)
        # Iterate over the ordered pairs, return the first one that doesnt violate any condtions

        for pi, center, normal in bcent_point_pairs:
            nn = list(
                set(
                    np.concatenate(
                        (
                            self.nearest_neighbors[pi],
                            self.nearest_neighbors[edge.p1],
                            self.nearest_neighbors[edge.p2],
                        )
                    )
                )
            )
            nn = [n for n in nn if n < len(self.cloud)]
            free_of_points = self.check_ball_free_of_points(
                edge.p1, edge.p2, pi, center
            )
            not_the_prev_center = not np.allclose(center, edge.ball_center)
            good_pt = free_of_points and not_the_prev_center
            """
            plot_pivot(
                "good pt" if good_pt else "BAD",
                self.cloud[pi],
                B,
                C,
                center,
                self._rad,
                self.cloud[nn],
            )
            """
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
        # self.mesh.save_mesh("mesh_series/{}.ply".format(len(self.used_points)))
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
        # Check all of our unsed points
        for p in unused_pts:
            # Look at their two closest neighbors
            n1, n2 = self.nearest_neighbors[p][0:2]
            if n1 < len(self.cloud) and n2 < len(self.cloud):
                center, rad = calc_circumcircle(
                    self.cloud[p], self.cloud[n1], self.cloud[n2]
                )
                if rad:
                    return ((p, n1, n2), center, rad)

        return None
