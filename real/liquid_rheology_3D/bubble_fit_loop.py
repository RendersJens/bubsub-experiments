import sys
sys.path.append("../../..")

import numpy as np
import cupy as cp
import astra
import trimesh
import pylops
from matplotlib import pyplot as plt
from mesh_projector.projection3D.project_mesh_gpu64 import project_mesh, grad_project_mesh
from mesh_projector.solvers import quasi_newton, BB
from mesh_projector.utils.recalculate_normals import recalculate_normals
from parametric import loop_subdivide
from tqdm import tqdm

#attenuation = -0.0002791397898305019
attenuation = -0.00032*2 # times 2 because half resolution
#attenuation = -0.0004

int_type = np.int64
float_type = np.float64

print("loading sinogram")
sino = cp.load("data/sino.npy")[::5, :, :].astype(float_type).copy()
print("done")

angles = np.linspace(0, np.pi, 300, endpoint=False, dtype=float_type)
angles = angles[::5].copy()
geom_settings = (1, 1, sino.shape[1], sino.shape[2], angles, 50000, 1)


# create astra optomo operator
vol_geom = astra.create_vol_geom(sino.shape[2], sino.shape[2], sino.shape[1])
proj_geom = astra.create_proj_geom('cone', *geom_settings)
proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
W = astra.optomo.OpTomo(proj_id)

geom_settings = (1, 1, sino.shape[1], sino.shape[2], cp.asarray(angles)+np.pi/2, 50000, 1)

# ordinary reconstruction
print("loading ordinary reconstruction")
ordinary_rec = np.load("data/initial_rec.npy")
print("done")

rec_shape = ordinary_rec.shape
rec_size = ordinary_rec.size

# initial guess
sphere = trimesh.load("data/sphere_div1.stl")
sphere_vertices = np.asarray(sphere.vertices, dtype=float_type)
sphere_faces = np.asarray(sphere.faces, dtype=int_type)

centers = np.load("data/centers.npy")
n_bubbles = len(centers)
centers[:, 0] -= rec_shape[0]//2
centers[:, 1] -= rec_shape[1]//2
centers[:, 2] -= rec_shape[2]//2
centers = centers[:, [2,1,0]]
centers[:, 2] *= -1
radius = 25

print("creating initial guess")
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
print("done")

all_vertices, all_faces, S1 = loop_subdivide(control_vertices, control_faces, return_matrix=True)
all_vertices, all_faces, S2 = loop_subdivide(all_vertices, all_faces, return_matrix=True) 
S = pylops.MatrixMult(S2) @ pylops.MatrixMult(S1)

def grad_f(x):
    all_vertices = (S @ x).reshape((-1, 3))
    all_normals = recalculate_normals(all_vertices, all_faces)
    proj_diff = project_mesh(all_vertices, all_faces, all_normals, *geom_settings)
    proj_diff *= attenuation
    proj_diff -= sino
    print(cp.dot(proj_diff.ravel(), proj_diff.ravel()))
    grad = grad_project_mesh(all_vertices, all_faces, all_normals, proj_diff, *geom_settings)
    grad *= attenuation
    return S.T @ grad.ravel().get()


def f(x):
    all_vertices = (S @ x).reshape((-1, 3))
    all_normals = recalculate_normals(all_vertices, all_faces)
    proj_diff = project_mesh(all_vertices, all_faces, all_normals, *geom_settings)
    proj_diff *= attenuation
    proj_diff -= sino
    return 1/2*cp.dot(proj_diff.ravel(), proj_diff.ravel())


# plot during the iterations
plt.ion()
fig, ax = plt.subplots(1)
ax.imshow(ordinary_rec[rec_shape[0]//2, :, :], cmap="gray")
plt.pause(0.1)
iteration = 0
np.save(f"data/last_results_60/all_faces", all_faces)
def callback(current_solution):
    global iteration
    iteration += 1
    ax.clear()
    ax.imshow(ordinary_rec[rec_shape[0]//2, :, :], cmap="gray")
    fig.suptitle(str(iteration))
    vertices = (S @ current_solution).reshape((-1, 3))
    np.save(f"data/last_results_60/iterate{iteration}", vertices)
    faces = all_faces
    normals = recalculate_normals(vertices, faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_normals=normals)
    lines = trimesh.intersections.mesh_plane(mesh, np.array([0,0,1.0]), np.array([0,0,0.0]))
    for line in lines:
        ax.plot(*line[:, :2].T+rec_shape[1]//2, 'b')
    plt.pause(0.01)

rec = BB(
    grad_f,
    x0=control_vertices.ravel(),
    max_iter=20,
    verbose=True,
    callback=callback,
)[0]

rec = quasi_newton(
    f,
    grad_f,
    x0=rec,
    H_strat="BFGS",
    max_iter=8,
    verbose=True,
    callback=callback
)

rec = BB(
    grad_f,
    x0=rec,
    max_iter=20,
    verbose=True,
    callback=callback,
)[0]

rec = quasi_newton(
    f,
    grad_f,
    x0=rec,
    H_strat="BFGS",
    max_iter=8,
    verbose=True,
    callback=callback
)


all_vertices = (S @ rec).reshape((-1, 3))
all_normals = recalculate_normals(all_vertices, all_faces)
mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, face_normals=all_normals)
trimesh.exchange.export.export_mesh(mesh, "data/final_reconstruction_60.stl")