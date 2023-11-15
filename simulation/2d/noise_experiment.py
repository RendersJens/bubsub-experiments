import sys
sys.path.append("../../..")

import numpy as np
import astra
import pylops
from matplotlib import pyplot as plt
from mesh_tomography.projection2D.project_poly_gpu64 import project_poly, grad_project_poly
from mesh_tomography.reconstruction import quasi_newton, BB
from mesh_tomography.utils.convert_poly import convert_polys
from mesh_tomography.utils.parametric import make_circles, d_make_circles
from mesh_tomography.utils.subdivision import chaikin_subdivide
from simulate_scan import simulate_scan
from scipy.sparse.linalg import lsqr
from tqdm import tqdm
import os

attenuation = 1/2000
n_control_points = 9
n_proj = 150
realisation = sys.argv[1]

if not os.path.exists(f"data/noise_experiment_{realisation}"):
   os.makedirs(f"data/noise_experiment_{realisation}")

for power in np.linspace(2, 5, 30):
    I0 = round(10**power)
    sino = np.fliplr(simulate_scan(n_proj=n_proj, I0=I0))
    angles = np.linspace(0, np.pi, sino.shape[0], endpoint=False)
    geom_settings = (1, 2400, angles, 500000, 1)


    # create astra optomo operator
    vol_geom = astra.create_vol_geom(2000, 2000)
    proj_geom = astra.create_proj_geom('fanflat', *geom_settings)
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    W = astra.optomo.OpTomo(proj_id)


    # ordinary reconstructions
    lsqr_rec = lsqr(W, np.fliplr(sino).ravel(), iter_lim=20, show=True)[0]
    lsqr_rec = lsqr_rec.reshape(2000, 2000)
    np.save(f"data/noise_experiment_{realisation}/lsqr_rec{I0}", lsqr_rec)
    rec_shape = lsqr_rec.shape

    fbp_rec = astra.create_reconstruction("FBP_CUDA", proj_id, np.fliplr(sino))[1]
    np.save(f"data/noise_experiment_{realisation}/fbp_rec{I0}", fbp_rec)

    geom_settings = (1, 2400, angles, 500000, 1)

    # initial guess
    centers = np.loadtxt("data/centers.txt")
    n_bubbles = len(centers)
    centers[:, 0] -= rec_shape[0]//2
    #centers[:, 0] *= -1
    centers[:, 1] -= rec_shape[1]//2
    radius = 27

    x0 = np.hstack([centers, np.ones((n_bubbles,1))*radius])
    x0 = x0.ravel()

    def g(x):
        return f(make_circles(x, n=n_control_points))


    def grad_g(x):
        A = d_make_circles(x, n=n_control_points).T
        B =  grad_f(make_circles(x, n=n_control_points))
        return A @ B

    v, S1 = chaikin_subdivide(make_circles(x0, n=n_control_points).reshape((n_bubbles, -1, 2)), return_matrix=True)
    _, S2 = chaikin_subdivide(v, return_matrix=True)
    S = pylops.MatrixMult(S2) @ pylops.MatrixMult(S1)

    def grad_f(x):
        polys = (S @ x).reshape((n_bubbles, -1, 2))
        proj_diff = attenuation*project_poly(*convert_polys(polys), *geom_settings) - sino
        grad = attenuation*grad_project_poly(*convert_polys(polys), proj_diff, *geom_settings)
        return S.T @ grad.ravel()


    def f(x):
        polys = (S @ x).reshape((n_bubbles, -1, 2))
        proj_diff = attenuation*project_poly(*convert_polys(polys), *geom_settings) - sino
        return 1/2*np.dot(proj_diff.ravel(), proj_diff.ravel())

    # solve
    rec = BB(
        grad_g,
        x0=x0,
        #H_strat="BFGS",
        max_iter=30,
        verbose=True
    )[0]

    rec = quasi_newton(
        f,
        grad_f,
        x0=make_circles(rec, n=n_control_points),
        H_strat="BFGS",
        max_iter=80,
        verbose=True
    )
    rec = (S @ rec).reshape((n_bubbles,-1,2))
    np.save(f"data/noise_experiment_{realisation}/rec{I0}", rec)
