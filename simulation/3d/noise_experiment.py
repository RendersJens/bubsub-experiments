import numpy as np
import astra
import trimesh
import pylops
from matplotlib import pyplot as plt
from mesh_tomography.projection3D.project_mesh_gpu64 import project_mesh, grad_project_mesh
from mesh_tomography.reconstruction import quasi_newton, BB
from mesh_tomography.utils.recalculate_normals import recalculate_normals
from mesh_tomography.utils.subdivision import loop_subdivide
from mesh_tomography.utils.spheres import generate_sphere
from simulate_scan import simulate_scan
from scipy.sparse.linalg import lsqr
from tqdm import tqdm
import os
import sys

int_type = np.int64
float_type = np.float64

attenuation = 1/180
n_proj = 25
realisation = sys.argv[1]

if not os.path.exists(f"data/noise_experiment_{realisation}"):
   os.makedirs(f"data/noise_experiment_{realisation}")

for power in np.linspace(2, 5, 30):
    I0 = round(10**power)
    print("generating sinogram")
    sino = simulate_scan(n_proj=n_proj, I0=I0).astype(float_type)
    print("done")

    angles = np.linspace(0, 2*np.pi, sino.shape[0], endpoint=False)
    geom_settings = (1, 1, 256, 512, angles, 5000, 200)


    # create astra optomo operator
    vol_geom = astra.create_vol_geom(sino.shape[2], sino.shape[2], sino.shape[1])
    proj_geom = astra.create_proj_geom('cone', *geom_settings)
    proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
    W = astra.optomo.OpTomo(proj_id)

    # ordinary reconstruction
    lsqr_rec = lsqr(W, sino.swapaxes(0, 1).ravel(), iter_lim=20, show=True)[0]
    lsqr_rec = lsqr_rec.reshape((sino.shape[1], sino.shape[2], sino.shape[2]))
    np.save(f"data/noise_experiment_{realisation}/lsqr_rec{I0}", lsqr_rec)
    rec_shape = lsqr_rec.shape
    rec_size = lsqr_rec.size

    cfg = astra.astra_dict("FDK_CUDA")
    sino_id = astra.data3d.create("-sino", proj_geom, sino.swapaxes(0, 1))
    vol_id = astra.data3d.create("-vol", vol_geom)
    cfg["ProjectionDataId"] = sino_id
    cfg["ReconstructionDataId"] = vol_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    fdk_rec = astra.data3d.get(vol_id)
    astra.astra.delete([sino_id, vol_id, alg_id])
    np.save(f"data/noise_experiment_{realisation}/fdk_rec{I0}", fdk_rec)

    geom_settings = (1, 1, 256, 512, angles + np.pi/2, 5000, 200)

    # initial guess
    sphere = generate_sphere(1)
    sphere_vertices = np.asarray(sphere.vertices, dtype=float_type)
    sphere_faces = np.asarray(sphere.faces, dtype=int_type)

    centers = np.load("data/centers.npy")
    n_bubbles = len(centers)
    centers[:, 0] -= rec_shape[0]//2
    centers[:, 1] -= rec_shape[1]//2
    centers[:, 2] -= rec_shape[2]//2
    centers = centers[:, [2,1,0]]
    centers[:, 2] *= -1
    radius = 50/2

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

    level1_vertices, level1_faces, S1 = loop_subdivide(control_vertices, control_faces, return_matrix=True)
    all_vertices, all_faces, S2 = loop_subdivide(level1_vertices, level1_faces, return_matrix=True) 
    S = pylops.MatrixMult(S2) @ pylops.MatrixMult(S1)

    def grad_f(x):
        all_vertices = (S @ x).reshape((-1, 3))
        all_normals = recalculate_normals(all_vertices, all_faces)
        proj_diff = project_mesh(all_vertices, all_faces, all_normals, *geom_settings).get()
        proj_diff *= attenuation
        proj_diff -= sino
        print(np.dot(proj_diff.ravel(), proj_diff.ravel()))
        grad = grad_project_mesh(all_vertices, all_faces, all_normals, proj_diff, *geom_settings).get()
        grad *= attenuation
        return S.T @ grad.ravel()


    def f(x):
        all_vertices = (S @ x).reshape((-1, 3))
        all_normals = recalculate_normals(all_vertices, all_faces)
        proj_diff = project_mesh(all_vertices, all_faces, all_normals, *geom_settings).get()
        proj_diff *= attenuation
        proj_diff -= sino
        return 1/2*np.dot(proj_diff.ravel(), proj_diff.ravel())

    rec = BB(
        grad_f,
        x0=control_vertices.ravel(),
        max_iter=10,
        verbose=True
    )[0]

    rec = quasi_newton(
        f,
        grad_f,
        x0=rec,
        H_strat="BFGS",
        max_iter=15,
        verbose=True
    )

    all_vertices = (S @ rec).reshape((-1, 3))
    all_normals = recalculate_normals(all_vertices, all_faces)
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, face_normals=all_normals)
    trimesh.exchange.export.export_mesh(mesh, f"data/noise_experiment_{realisation}/rec{I0}.stl")