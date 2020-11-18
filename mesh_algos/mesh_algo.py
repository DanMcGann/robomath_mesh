"""
This file contains a base class for all the mesh generation algorithms.



"""
from abc import ABC
from pymesh import save_mesh_raw

import numpy as np
import pcl
import typing

"""
Data Definitions
"""

# A point in the cloud of form [x,y,z]
Point = np.ndarray

# A triangle in a Mesh
Triangle = typing.Tuple[Point, Point, Point]


class Mesh:
    """
    Class that defines a mesh
    """

    # Need: List of Vertices (Points)
    # Need: List of Faces (indices of points that make up each face)
    def add_triangle(self, t: Triangle):
        """
        Adds a Triangle to the Mesh
        """
        pass

    def save_mesh(self, out_file: str):
        """
        Saves a generated mesh to file
        """
        vertices = None
        faces = None
        save_mesh_raw(out_file, vertices, faces)


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
