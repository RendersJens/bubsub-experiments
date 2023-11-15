import numpy as np
import cupy as cp
import astra
import trimesh
import pylops
from matplotlib import pyplot as plt
from mesh_tomography.projection3D.project_mesh_gpu64 import project_mesh, grad_project_mesh
from mesh_tomography.reconstruction import quasi_newton, BB
from mesh_tomography.utils.recalculate_normals import recalculate_normals
from mesh_tomography.utils.subdivision import loop_subdivide
from mesh_tomography.utils.spheres import generate_sphere
from tqdm import tqdm

#attenuation = -0.0002791397898305019
attenuation = -0.00032*2 # times 2 because half resolution
#attenuation = -0.0004

int_type = np.int64
float_type = np.float64

n_angles = 300
angle_inds = np.round(np.linspace(0, 300, n_angles, endpoint=False)).astype(np.int64)
print(angle_inds)

print("loading sinogram")
sino = cp.load("data/sino.npy")[angle_inds, :, :].astype(float_type).copy()
print("done")

angles = np.linspace(0, np.pi, 300, endpoint=False, dtype=float_type)
angles = angles[angle_inds].copy()
geom_settings = (1, 1, sino.shape[1], sino.shape[2], angles, 50000, 1)


# create astra optomo operator
vol_geom = astra.create_vol_geom(sino.shape[2], sino.shape[2], sino.shape[1])
proj_geom = astra.create_proj_geom('cone', *geom_settings)
proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
W = astra.optomo.OpTomo(proj_id)

geom_settings = (1, 1, sino.shape[1], sino.shape[2], cp.asarray(angles)+np.pi/2, 50000, 1)

# ordinary reconstruction
print("loading volumetric reconstruction")
rec = np.load(f"data/vol_rec_{n_angles}.npy")
print("done")

vol_sino = (W @ rec.ravel()).reshape(sino.shape).swapaxes(0,1)


vertices = np.load(f"data/last_results_{n_angles}/iterate{58}.npy")
faces = np.load(f"data/last_results_{n_angles}/all_faces.npy")
normals = recalculate_normals(vertices, faces)
mesh_sino = project_mesh(vertices, faces, normals, *geom_settings).get()*attenuation

print("creating initial guess")
sphere = generate_sphere(1)
sphere_vertices = np.asarray(sphere.vertices, dtype=float_type)
sphere_faces = np.asarray(sphere.faces, dtype=int_type)
centers = np.load("data/centers.npy")
n_bubbles = len(centers)
centers[:, 0] -= rec.shape[0]//2
centers[:, 1] -= rec.shape[1]//2
centers[:, 2] -= rec.shape[2]//2
centers = centers[:, [2,1,0]]
centers[:, 2] *= -1
radius = 25
control_vertices = []
control_faces = []
for i in range(n_bubbles):
    center = centers[i]
    bubble = sphere_vertices*radius
    bubble[:, 0] += center[0]
    bubble[:, 1] += center[1]
    bubble[:, 2] += center[2]
    control_vertices.append(bubble)
    control_faces.append(sphere_faces+(len(sphere_vertices)*i))
control_vertices = np.vstack(control_vertices)
control_faces = np.vstack(control_faces)
vertices, faces = loop_subdivide(control_vertices, control_faces)
vertices, faces = loop_subdivide(vertices, faces)
normals = recalculate_normals(vertices, faces)
init_sino = project_mesh(vertices, faces, normals, *geom_settings).get()*attenuation

plt.figure()
plt.imshow(init_sino[:, sino.shape[1]//2, :], cmap="gray")
plt.imsave("images/init_sino.png", init_sino[:, sino.shape[1]//2, :], cmap="gray")

plt.figure()
plt.imshow(mesh_sino[:, sino.shape[1]//2, :], cmap="gray")
plt.imsave("images/mesh_sino.png", mesh_sino[:, sino.shape[1]//2, :], cmap="gray")

plt.figure()
plt.imshow(vol_sino[:, sino.shape[1]//2, :], cmap="gray")
plt.imsave("images/vol_sino.png", vol_sino[:, sino.shape[1]//2, :], cmap="gray")

plt.figure()
plt.imshow(sino[:, sino.shape[1]//2, :].get(), cmap="gray")
plt.imsave("images/sino.png", sino[:, sino.shape[1]//2, :].get(), cmap="gray")

plt.show()