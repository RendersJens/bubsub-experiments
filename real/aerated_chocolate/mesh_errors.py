import sys
import numpy as np
import cupy as cp
import astra
import trimesh
import pylops
import dxchange
from matplotlib import pyplot as plt
from mesh_tomography.projection3D.project_mesh_gpu64 import project_mesh
from mesh_tomography.utils.recalculate_normals import recalculate_normals
from mesh_tomography.utils.subdivision import loop_subdivide
from tqdm import tqdm

int_type = np.int64
float_type = np.float64

attenuation = -0.0034
total_angles = 900

path = "/media/jens/Samsung_T5/journal-paper-mesh-reconstructions/real/luflee/"

# load sino and crop projections
sino = np.load(path + "data/sino.npy")[:, :, 500:-500]

# select half of the angles, the ones not used in reconstruction
sino = sino[1::2]
sino = np.asarray(sino, dtype=float_type)

angles = -np.linspace(0, np.pi, total_angles, endpoint=False)

# select half of the angles, the ones not used in reconstruction
angles = angles[1::2]

# geometry settings
geom_settings = (
    1, 1,
    sino.shape[1],
    sino.shape[2], 
    angles+np.pi/2,
    50000, # simulate parallel beam by placing source far away
    1
)

errors = []
for n_angles in list(range(2, 11, 2)) + list(range(15, 51, 5)) + list(range(60, 201, 10)):
    control_vertices = np.load(f"data/control_vertices_{n_angles}.npy")
    control_faces = np.load(f"data/control_faces_{n_angles}.npy")
    all_vertices, all_faces = loop_subdivide(control_vertices, control_faces)
    all_normals = recalculate_normals(all_vertices, all_faces)


    proj_diff = project_mesh(all_vertices, all_faces, all_normals, *geom_settings).get()
    proj_diff *= attenuation
    proj_diff -= sino
    proj_diff = proj_diff
    error = np.dot(proj_diff.ravel(), proj_diff.ravel())/proj_diff.size

    print(error)
    errors.append(error)

np.save("data/mean_squared_projection_errors_mesh.npy", errors)