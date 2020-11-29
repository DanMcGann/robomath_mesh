#! /usr/bin/env python3

from mesh_algos.ball_pivot import BallPivot
from mesh_algos.mesh_algo import load_cloud, load_cloud_from_xyz
import matplotlib.pyplot as plt


file = "/home/daniel/Development/1-year/robomath_16881/robomath_mesh/data/terrain/a100_dome_007.xyz"
cloud = load_cloud_from_xyz(file)
bp = BallPivot(0.3)
mesh = bp.generate_mesh(cloud)

mesh.save_mesh("mesh.ply")
