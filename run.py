#! /usr/bin/env python3

from mesh_algos.ball_pivot import BallPivot
from mesh_algos.mesh_algo import load_cloud
import matplotlib.pyplot as plt


file = "/home/daniel/Development/1-year/robomath_16881/robomath_mesh/data/objects/stanford/decimation/bunny_D01_L04.ply"
cloud = load_cloud(file)
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.scatter(cloud.T[0], cloud.T[1], cloud.T[2])
plt.show()

bp = BallPivot(0.1)
mesh = bp.generate_mesh(cloud)

mesh.save_mesh("mesh.ply")
