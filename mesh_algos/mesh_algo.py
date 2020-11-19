"""
This file contains a base class for all the mesh generation algorithms.



"""
from abc import ABC
import pymesh

import numpy as np
import pcl
import typing
from collections import OrderedDict

"""
Data Definitions
"""

# A point in the cloud of form [x,y,z]
Point = np.ndarray

# A triangle in a Mesh
Triangle = typing.Tuple[Point, Point, Point]


class Mesh:
    """
    Class that defines a mesh.
    Pymesh handles the unparsing to files. We just need to provide
        - List of vertices
        - List of faces defined by three indices corresponding to the vertices of the face
    """

    def __init__(self):
        # Point -> idx
        self.vertices = OrderedDict()
        # List of np.ndarray[idx, idx, idx]
        self.faces = []

    def add_triangle(self, triangle: Triangle):
        """
        Adds a Triangle to the Mesh
        """
        vert_indices = []
        for p in triangle:
            if p not in self.vertices:
                self.vertices[p] = len(self.vertices)
            vert_indices.append(self.vertices[p])
        self.faces.append(np.array(vert_indices))

    def save_mesh(self, out_file: str):
        """
        Saves a generated mesh to file
        """
        vertices = np.stack(self.vertices.keys())
        faces = np.stack(self.faces)
        pymesh.save_mesh_raw(out_file, vertices, faces)


"""
Helper functions 
"""


def load_cloud_from_pcd(file_name: str) -> np.ndarray:
    """
    Loads a point cloud from a pcd file
    Returns the cloud as a numpy array of shape (N, 3) each row containing one [x,y,z] point
    """
    return pcl.load(file_name).to_array()


def load_cloud_from_xyz(file_name: str) -> np.ndarray:
    """
    Loads a point cloud from a xyz file
    Returns the cloud as a numpy array of shape (N, 3) each row containing one [x,y,z] point
    """
    return pcl.load_XYZI(
        file_name
    ).to_array()  # TODO Will this work with a .xyz or just with a .xyzi


"""
Interface for Algorithms
"""


class MeshAlgo(ABC):
    """
    Base class defining the required functionality for all mesh generation algorithms.
    Also provides common functionality for reading point clouds, and saving the generated meshes.
    """

    def generate_mesh(self, cloud: np.ndarray) -> Mesh:
        """
        Generates a mesh from a pointcloid of x,y,z points

        Params:
            cloud: The point cloud. As a numpy array of shape (N, 3)
                   each row containing one [x,y,z] point

        Returns:
            mesh: TODO FIGURE OUT DATA STRUCTURE FOR MESHES? PLY?
        """
        raise NotImplementedError()
