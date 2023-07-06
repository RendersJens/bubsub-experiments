import sys
import numpy as np
import cupy as cp
import astra
import trimesh
import pylops
import dxchange
from matplotlib import pyplot as plt
from tqdm import tqdm
from mesh_tomography.projection3D.project_mesh_gpu64 import project_mesh
from mesh_tomography.projection3D.project_mesh_gpu64 import grad_project_mesh
from mesh_tomography.reconstruction import quasi_newton, BB
from mesh_tomography.utils.recalculate_normals import recalculate_normals
from mesh_tomography.utils.subdivision import loop_subdivide
from mesh_tomography.utils.remesh import remesh
from cleaning import clean_bubble_mesh


subscan = int(sys.argv[1])

flux = 1e4
angle_skip = 4
attenuation = 1
phantom_width_cm = 3
phantom_width_vox = 500
phantom_height_vox = 620
voxel_size = phantom_width_cm/phantom_width_vox

int_type = np.int64
float_type = np.float64

print("loading sinogram")
sino_complete = dxchange.read_tiff_stack(f"../phantom/scan/subscan_{subscan}/proj_00000.tiff", range(100))
sino_container = dxchange.read_tiff_stack(f"../phantom/scan_container/subscan_{subscan}/proj_00000.tiff", range(100))

sino_complete = flux*np.exp(-sino_complete)
np.random.seed(0)
sino_complete = np.random.poisson(sino_complete)
sino_complete = -np.log(sino_complete/flux)

sino = sino_container - sino_complete
sino /= voxel_size
sino = cp.asarray(sino).astype(float_type)
del sino_complete
del sino_container
print("done")

angles = np.linspace(0, 2*np.pi, sino.shape[0])
sino = sino[::angle_skip].copy()
angles = angles[::angle_skip].copy()
geom_settings = (
    1, 1,
    phantom_height_vox,
    phantom_width_vox, 
    angles,
    phantom_width_vox*10,
    phantom_width_vox//2
)


# create astra optomo operator
vol_geom = astra.create_vol_geom(sino.shape[2], sino.shape[2], sino.shape[1])
proj_geom = astra.create_proj_geom('cone', *geom_settings)
proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
W = astra.optomo.OpTomo(proj_id)

geom_settings = (
    1, 1,
    phantom_height_vox,
    phantom_width_vox, 
    angles + np.pi/2,
    phantom_width_vox*10,
    phantom_width_vox//2
)

# ordinary reconstruction
print("loading ordinary reconstruction")
ordinary_rec = np.load(f"data/ordinary_reconstruction_{subscan}.npy")
ordinary_rec = np.flip(ordinary_rec, axis=0)
print("done")

rec_shape = ordinary_rec.shape
rec_size = ordinary_rec.size

# initial guess
print("loading initial guess")
control_vertices = np.load(f"data/control_vertices_{subscan + 1}.npy")
control_faces = np.load(f"data/control_faces_{subscan + 1}.npy")
control_vertices, control_faces = clean_bubble_mesh(
    control_vertices, control_faces)
print("done")

# vertices = control_vertices.reshape((-1, 3))
# faces = control_faces
# print("computing normals")
# normals = recalculate_normals(vertices, faces)
# print("done")

# print("projecting x0")
# sino_x0 = attenuation*project_mesh(vertices, faces, normals, *geom_settings)
# print("done")

# plt.figure()
# plt.title("sino initial guess")
# plt.imshow(sino_x0[:, :, sino_x0.shape[2]//2].get(), cmap="gray", aspect="auto")
# plt.colorbar()

# plt.figure()
# plt.title("sino initial guess")
# plt.imshow(sino_x0[:, sino_x0.shape[1]//2, :].get(), cmap="gray", aspect="auto")
# plt.colorbar()

# plt.figure()
# plt.title("sino initial guess")
# plt.imshow(sino_x0[sino_x0.shape[0]//2, :, :].get(), cmap="gray", aspect="auto")
# plt.colorbar()

# plt.figure()
# plt.title("sino")
# plt.imshow(sino[:, :, sino.shape[2]//2].get(), cmap="gray", aspect="auto")
# plt.colorbar()

# plt.figure()
# plt.title("sino")
# plt.imshow(sino[:, sino.shape[1]//2, :].get(), cmap="gray", aspect="auto")
# plt.colorbar()

# plt.figure()
# plt.title("sino")
# plt.imshow(sino[sino.shape[0]//2, :, :].get(), cmap="gray", aspect="auto")
# plt.colorbar()

# mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_normals=normals)
# print("computing slice of x0")
# lines = trimesh.intersections.mesh_plane(mesh, np.array([0,0,1.0]), np.array([0,0,0.0]))
# print("done")

# plt.figure()
# plt.imshow(ordinary_rec[rec_shape[0]//2, :, :], cmap="gray")
# for line in lines:
#     plt.plot(*line[:, :2].T+rec_shape[1]//2, 'b')

# plt.show()

all_vertices, all_faces, S1 = loop_subdivide(control_vertices, control_faces, return_matrix=True)
# all_vertices, all_faces, S2 = loop_subdivide(level1_vertices, level1_faces, return_matrix=True) 
# all_faces = np.ascontiguousarray(all_faces.astype(int_type))
# S = pylops.MatrixMult(S2) @ pylops.MatrixMult(S1)
S = S1

def grad_f(x):
    all_vertices = (S @ x).reshape((-1, 3))
    all_vertices = np.ascontiguousarray(all_vertices.astype(float_type))
    all_normals = recalculate_normals(all_vertices, all_faces)
    all_normals = np.ascontiguousarray(all_normals.astype(float_type))
    proj_diff = project_mesh(all_vertices, all_faces, all_normals, *geom_settings)
    if proj_diff.min() == proj_diff.max():
        raise
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
    return 1/2*cp.dot(proj_diff.ravel(), proj_diff.ravel()).get()


# plot during the iterations
plt.ion()
fig, ax = plt.subplots(1)
ax.imshow(ordinary_rec[:, rec_shape[1]//2, :], cmap="gray")
plt.pause(0.1)
iteration = 0
def callback(current_solution):
    global iteration
    iteration += 1
    ax.clear()
    ax.imshow(ordinary_rec[:, rec_shape[1]//2, :], cmap="gray")
    fig.suptitle(str(iteration))
    vertices = (S @ current_solution).reshape((-1, 3))
    faces = all_faces
    normals = recalculate_normals(vertices, faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_normals=normals)
    lines = trimesh.intersections.mesh_plane(mesh, np.array([0,1.0,0]), np.array([0,0,0.0]))
    for line in lines:
        ax.plot(*line[:, [0,2]].T+np.array([[rec_shape[2]//2, rec_shape[0]//2]]).T, 'b')
    plt.pause(0.01)

# rec = BB(
#     grad_f,
#     x0=control_vertices.ravel(),
#     max_iter=80,
#     verbose=True,
#     callback=callback,
# )[0]

# control_vertices = (S1 @ rec).reshape((-1, 3))
# control_faces = level1_faces
# S = S2

# rec = BB(
#     grad_f,
#     x0=control_vertices.ravel(),
#     max_iter=40,
#     verbose=True,
#     callback=callback,
# )[0]

rec = quasi_newton(
    f,
    grad_f,
    x0=control_vertices.ravel(),
    H_strat="BFGS",
    max_d=phantom_height_vox,
    max_iter=80,
    verbose=True,
    callback=callback
)
np.save(f"data/control_vertices_{subscan}", rec.reshape(control_vertices.shape))
np.save(f"data/control_faces_{subscan}", control_faces)
all_vertices = (S @ rec).reshape((-1, 3))
all_normals = recalculate_normals(all_vertices, all_faces)
mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, face_normals=all_normals)
trimesh.exchange.export.export_mesh(mesh, f"data/rec_{subscan}.stl")