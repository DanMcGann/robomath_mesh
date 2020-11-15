"""
This file contains a base class for all the mesh generation algorithms.



"""
from abc import ABC

import numpy as np
import pcl


class MeshAlgo(ABC):
    """
    Base class defining the required functionality for all mesh generation algorithms.
    Also provides common functionality for reading point clouds, and saving the generated meshes.
    """

    def generate_mesh(self, cloud: np.ndarray) -> int:
        """
        Generates a mesh from a pointcloid of x,y,z points

        Params:
            cloud: The point cloud. As a numpy array of shape (N, 3)
                   each row containing one [x,y,z] point

        Returns:
            mesh: TODO FIGURE OUT DATA STRUCTURE FOR MESHES? PLY?
        """
        raise NotImplementedError()

    def load_from_pcd(self, file_name: str) -> np.ndarray:
        """
        Loads a point cloud from a pcd file
        Returns the cloud as a numpy array of shape (N, 3) each row containing one [x,y,z] point
        """
        return pcl.load(file_name).to_array()

    def load_from_xyz(self, file_name: str) -> np.ndarray:
        """
        Loads a point cloud from a xyz file
        Returns the cloud as a numpy array of shape (N, 3) each row containing one [x,y,z] point
        """
        return pcl.load_XYZI(
            file_name
        ).to_array()  # TODO Will this work with a .xyz or just with a .xyzi

    def save_mesh(
        self, out_file: str, mesh
    ):  # TODO figure out the datatype for out mesh
        """
        Saves a generated mesh to file
        """
        pass  # TODO
