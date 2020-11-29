# Final Project
Code for Robomath final project exploring mesh generation from pointclouds.


# Navigation
* `data/` - Location of Example Point Cloud Data
    * `objects/` - Small object point clouds
    * `terrain/` - Large outdoor point clouds
* `mesh_algos/` - Location of source code for Cloud -> Mesh Algorithms
* `scripts/` - Location of scripts to run and visualize point clouds and the computed meshes


# Dependencies
* python3
* python-pcl
    * Used for loading `pcd` and `xyz` point cloud files
    * `apt-get install python3-pcl` note: the pip version doesn't appear to work
* pymesh
    * used for unparsing meshes into `obj` or `ply` or `stl` files. 
    * [Instillaton Guide](https://pymesh.readthedocs.io/en/latest/installation.html)
    * Note: I had to change the `setup.py` and `thirdparty/build.py` hashbangs to `python3`
    * Instillation: Had to force `-j1` compilation flag in `setup.py`

# Formatting
* https://pypi.org/project/black/