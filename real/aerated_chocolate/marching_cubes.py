import numpy as np
import cupy as cp
import astra
import trimesh
import pylops
import dxchange
from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage.measure import marching_cubes

int_type = np.int64
float_type = np.float64
attenuation = -0.0034
path = "/media/jens/Samsung_T5/journal-paper-mesh-reconstructions/real/luflee/"
# sino = cp.asarray(np.load(path + "data/sino.npy"), dtype=float_type)

# # create astra optomo operator
# angles = -np.linspace(0, np.pi, sino.shape[0], endpoint=False)
# sino = sino[::5]
# angles = angles[::5]
# geom_settings = (
#     1, 1,
#     sino.shape[1],
#     sino.shape[2], 
#     angles+np.pi/2,
#     50000,
#     1
# )

# ordinary reconstruction
print("loading ordinary reconstruction")
rec = np.load("data/ordinary_reconstruction_25.npy")
# rec = rec[2:-2, 700:980, 550:900]
print("done")

plt.figure()
plt.imshow(rec[rec.shape[0]//2, :, :], cmap="gray")

plt.figure()
plt.imshow(rec[:, rec.shape[1]//2, :], cmap="gray")

plt.figure()
plt.imshow(rec[:, :, rec.shape[2]//2], cmap="gray")

plt.show()

vertices, faces, normals, values = marching_cubes(
    rec,
    level=-0.002,
    step_size=2)

mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_normals=normals)
trimesh.exchange.export.export_mesh(mesh, f"data/mc_25.stl")