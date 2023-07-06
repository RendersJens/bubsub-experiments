import sys
sys.path.append("../../..")

import numpy as np
import astra
import pylops
from matplotlib import pyplot as plt
from mesh_projector.projection2D.project_poly_gpu64 import project_poly, grad_project_poly
from mesh_projector.solvers import quasi_newton, BB
from mesh_projector.utils.convert_poly import convert_polys
from mesh_projector.pixelation import pixelate
from parametric import make_circles, d_make_circles, chaikin_subdivide
from scipy.sparse.linalg import lsqr
from tqdm import tqdm

n_proj = 300

attenuation = -0.00032
n_control_points = 8

sino = np.load("data/sino_with_paganin_no_bg_slice66.npy")
angles = np.linspace(0, np.pi, sino.shape[0], endpoint=False)
sino = sino[::(300//n_proj)]
angles = angles[::(300//n_proj)]
geom_settings = (1, sino.shape[1], angles, 50000, 1)


# create astra optomo operator
vol_geom = astra.create_vol_geom(sino.shape[1], sino.shape[1])
proj_geom = astra.create_proj_geom('fanflat', *geom_settings)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
W = astra.optomo.OpTomo(proj_id)

# ordinary reconstructions
lsqr_rec = lsqr(W, np.fliplr(sino).ravel(), iter_lim=15, show=True)[0]
lsqr_rec = lsqr_rec.reshape((sino.shape[1], sino.shape[1]))
np.save(f"data/lsqr_rec{n_proj}", lsqr_rec)
rec_shape = lsqr_rec.shape

fbp_rec = astra.create_reconstruction("FBP_CUDA", proj_id, np.fliplr(sino))[1]
np.save(f"data/fbp_rec{n_proj}", fbp_rec)

geom_settings = (1, sino.shape[1], angles+np.pi/2, 50000, 1)


# initial guess
centers = np.loadtxt("data/centers.txt")
n_bubbles = len(centers)
centers[:, 0] -= rec_shape[0]//2
centers[:, 1] -= rec_shape[1]//2
centers[:, 0] *= -1
radius = 35

x0 = np.hstack([centers, np.ones((n_bubbles,1))*radius])
x0[75, 2] = 10
x0[76, 2] = 10
x0[77, 2] = 10
x0 = x0.ravel()

def g(x):
    return f(make_circles(x, n=10))

def grad_g(x):
    A = d_make_circles(x, n=10).T
    B =  grad_f(make_circles(x, n=10))
    return A @ B  

v, S1 = chaikin_subdivide(make_circles(x0, n=10).reshape((n_bubbles, -1, 2)), return_matrix=True) 
v, S2 = chaikin_subdivide(v, return_matrix=True)
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


# plot during the iterations
sino_x0 = attenuation*project_poly(*convert_polys(v), *geom_settings)
plt.ion()
fig, ax = plt.subplots(2, 3, figsize=(10, 7))
im_diff = ax[1,2].imshow(sino_x0-sino, cmap="gray", aspect="auto")
ax[1,2].set_xlabel("Difference")
im_current = ax[1,0].imshow(-sino_x0/sino_x0.min(), cmap="gray", aspect="auto")
ax[1,0].set_xlabel("Simulated sinogram")
ax[1,1].imshow(sino, cmap="gray", aspect="auto")
ax[1,1].set_xlabel("Real sinogram")
ax[0,0].axis('off')
ax[0,2].axis('off')
ax[0,2].set_xlabel("Difference")
plt.pause(0.1)
iteration = 0
def callback(current_solution):
    global iteration
    iteration += 1
    ax[0,1].clear()
    ax[0,1].imshow(np.fliplr(lsqr_rec), cmap="gray")
    fig.suptitle(f"Iteration {iteration}")
    bubbles = np.split(S @ current_solution, n_bubbles)
    for bubble in bubbles:
        b = bubble.reshape((-1,2)).copy()
        b[:, 0] += rec_shape[0]//2
        b[:, 1] += rec_shape[1]//2
        ax[0,1].plot(*np.vstack([b, b[0]]).T)#, "-o", markersize=2)

    polys = convert_polys((S @ current_solution).reshape((n_bubbles, -1, 2)))
    sino_current = attenuation*project_poly(*polys, *geom_settings)
    im_diff.set_data(sino_current-sino)
    im_current.set_data(-sino_current/sino_current.min())
    plt.draw()

    plt.pause(0.01)

def callback2(current_solution):
    global iteration
    iteration += 1
    ax[0,1].clear()
    ax[0,1].imshow(np.fliplr(lsqr_rec), cmap="gray")
    ax[0,1].set_xlabel("Simulated sinogram")
    fig.suptitle(f"Iteration {iteration} (radius estimation)")
    x = make_circles(current_solution)
    bubbles = np.split(x, n_bubbles)
    for bubble in bubbles:
        b = bubble.reshape((-1,2)).copy()
        b[:, 0] += rec_shape[0]//2
        b[:, 1] += rec_shape[1]//2
        ax[0,1].plot(*np.vstack([b, b[0]]).T)#, "-o", markersize=2)

    polys = convert_polys(x.reshape((n_bubbles, -1, 2)))
    sino_current = attenuation*project_poly(*polys, *geom_settings)
    im_diff.set_data(sino_current-sino)
    im_current.set_data(-sino_current/sino_current.min())
    plt.draw()

    plt.pause(0.01)


# solve

rec = BB(
    grad_g,
    x0=x0,
    #H_strat="BFGS",
    max_iter=15,
    verbose=True,
    callback=callback2
)[0]
iteration=0

v, S1 = chaikin_subdivide(make_circles(x0, n=n_control_points).reshape((n_bubbles, -1, 2)), return_matrix=True) 
_, S2 = chaikin_subdivide(v, return_matrix=True)
S = pylops.MatrixMult(S2) @ pylops.MatrixMult(S1)


rec = quasi_newton(
    f,
    grad_f,
    x0=make_circles(rec, n=n_control_points),
    H_strat="BFGS",
    max_iter=40,
    verbose=True,
    callback=callback
)
rec = (S @ rec).reshape((n_bubbles,-1,2))
np.save(f"data/rec{n_proj}", rec)
plt.ioff()
plt.show()