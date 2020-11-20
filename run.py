#! /usr/bin/env python3

from mesh_algos.mesh_algo import load_cloud_from_pcd
from mesh_algos.ball_pivot import BallPivot

file = (
    "/home/daniel/Development/1-year/robomath_16881/robomath_mesh/data/objects/cat.pcd"
)
cloud = load_cloud_from_pcd(file)
bp = BallPivot(10)
mesh = bp.generate_mesh(cloud)

mesh.save_mesh("mesh.ply")
