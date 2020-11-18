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
    * `apt-get install python3-pcl` note: the pip version doesn't appear to work
* pymesh
    * `pip3 install pymesh`

# Formatting
* https://pypi.org/project/black/