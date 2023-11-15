import sys
sys.path.append("../../..")

import numpy as np
import astra
import pylops
from matplotlib import pyplot as plt
from mesh_tomography.projection2D.project_circles import project_circles, grad_project_circles
from mesh_tomography.projection2D.project_poly_gpu64 import project_poly, grad_project_poly
from mesh_tomography.reconstruction import quasi_newton, BB
from mesh_tomography.utils.convert_poly import convert_polys
from mesh_tomography.pixelation import pixelate
from mesh_tomography.utils.parametric import make_circles, d_make_circles
from mesh_tomography.utils.subdivision import chaikin_subdivide
from scipy.sparse.linalg import lsqr
from tqdm import tqdm

attenuation = 1/2000
n_control_points = 9

sino = np.fliplr(np.load("data/sino.npy"))
angles = np.linspace(0, np.pi, sino.shape[0], endpoint=False)
geom_settings = (1, 2400, angles, 500000, 1)


# create astra optomo operator
vol_geom = astra.create_vol_geom(sino.shape[1], sino.shape[1])
proj_geom = astra.create_proj_geom('fanflat', *geom_settings)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
W = astra.optomo.OpTomo(proj_id)


# ordinary reconstruction
rec_shape = (2000, 2000)
rec_size = 2000 * 2000
ordinary_rec = np.load("data/ordinary_reconstruction.npy").T

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

plt.figure()
plt.title("initial guess")
plt.imshow(ordinary_rec, cmap="gray")
for i, bubble in enumerate(make_circles(x0).reshape((n_bubbles, -1, 2))):
    bubble = bubble.copy()
    bubble[:, 0] += rec_shape[0]//2
    bubble[:, 1] += rec_shape[1]//2
    plt.fill(*bubble.T)
    plt.annotate(i, bubble.mean(axis=0), ha="center", va="center")

sino_x0 = attenuation*project_circles(x0.reshape((-1, 3)), *geom_settings)
plt.figure()
plt.title("sino initial guess")
plt.imshow(sino_x0, cmap="gray", aspect="auto")
plt.colorbar()

plt.figure()
plt.title("sino")
plt.imshow(sino, cmap="gray", aspect="auto")
plt.colorbar()

plt.show()


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

# plot during the iterations
plt.ion()
fig, ax = plt.subplots(2, 3)
im_diff = ax[1,2].imshow(sino_x0-sino, cmap="gray", aspect="auto")
#im_current = ax[1,0].imshow(-sino_x0/sino_x0.min(), cmap="gray", aspect="auto")
ax[1,1].imshow(sino, cmap="gray", aspect="auto")
ax[0,1].imshow(ordinary_rec, cmap="gray")
plt.pause(0.1)
iteration = 0
def callback(current_solution):
    global iteration
    iteration += 1
    ax[0,0].clear()
    ax[0,2].clear()
    ax[0,2].imshow(ordinary_rec, cmap="gray")
    ax[0,0].imshow(np.zeros(rec_shape), cmap="gray")
    fig.suptitle(str(iteration))
    bubbles = np.split(S @ current_solution, n_bubbles)
    for bubble in bubbles:
        b = bubble.reshape((-1,2)).copy()
        b[:, 0] += rec_shape[0]//2
        b[:, 1] += rec_shape[1]//2
        ax[0,2].plot(*np.vstack([b, b[0]]).T)#, "-o", markersize=2)
        ax[0,0].fill(*b.T, 'w')

    polys = convert_polys((S @ current_solution).reshape((n_bubbles, -1, 2)))
    sino_current = attenuation*project_poly(*polys, *geom_settings)
    im_diff.set_data(sino_current-sino)
    #im_current.set_data(-sino_current/sino_current.min())
    plt.draw()

    plt.pause(0.01)

def callback2(current_solution):
    global iteration
    iteration += 1
    ax[0,0].clear()
    ax[0,2].clear()
    ax[0,2].imshow(ordinary_rec, cmap="gray")
    ax[0,0].imshow(np.zeros(rec_shape), cmap="gray")
    fig.suptitle(str(iteration))
    x = make_circles(current_solution)
    bubbles = np.split(x, n_bubbles)
    for bubble in bubbles:
        b = bubble.reshape((-1,2)).copy()
        b[:, 0] += rec_shape[0]//2
        b[:, 1] += rec_shape[1]//2
        ax[0,2].plot(*np.vstack([b, b[0]]).T)#, "-o", markersize=2)
        ax[0,0].fill(*b.T, 'w')

    polys = convert_polys(x.reshape((n_bubbles, -1, 2)))
    sino_current = attenuation*project_poly(*polys, *geom_settings)
    im_diff.set_data(sino_current-sino)
    #im_current.set_data(-sino_current/sino_current.min())
    plt.draw()

    plt.pause(0.01)


# solve

#callback = callback2 = lambda x: None
rec = BB(
    grad_g,
    x0=x0,
    #H_strat="BFGS",
    max_iter=30,
    verbose=True,
    callback=callback2,
)[0]

rec = quasi_newton(
    f,
    grad_f,
    x0=make_circles(rec, n=n_control_points),
    H_strat="BFGS",
    max_iter=80,
    verbose=True,
    callback=callback
)
rec = (S @ rec).reshape((n_bubbles,-1,2))
np.save("data/bubble_fit", rec)


# subdivisions = 1
# for i in range(subdivisions):
#     x0 = chaikin_subdivide(rec)
#     rec = quasi_newton(
#         f,
#         grad_f,
#         x0=x0.ravel(),
#         H_strat="BFGS",
#         max_iter=40,
#         verbose=True,
#         callback=callback,
#     )
#     rec = rec.reshape((n_bubbles,-1,2))


# show result
plt.ioff()

print("pixelating")
pixelation = pixelate(
    rec[0],
    np.array([-rec_shape[0]//2, rec_shape[0]//2]),
    np.array([-rec_shape[0]//2, rec_shape[0]//2]),
)*0
for poly in tqdm(rec):
    pixel_poly = pixelate(
        poly,
        np.array([-rec_shape[0]//2, rec_shape[0]//2]),
        np.array([-rec_shape[0]//2, rec_shape[0]//2]),
    )
    pixelation += pixel_poly

pixelation = np.flipud(pixelation)

plt.figure()
plt.title("bubble fit")
plt.imshow(np.clip(pixelation,0,1), cmap="gray")

plt.figure()
plt.title("ordinary reconstruction")
plt.imshow(ordinary_rec, cmap="gray")

plt.figure()
plt.title("bubble fit labeled")
plt.imshow(pixelation, cmap="gray")
for i, bubble in enumerate(rec):
    bubble = bubble.copy()
    bubble[:, 0] += rec_shape[0]//2
    bubble[:, 1] += rec_shape[1]//2
    plt.fill(*bubble.T)
    #plt.annotate(i, bubble[0])

polys = convert_polys(rec)
sino_rec = attenuation*project_poly(*polys, *geom_settings)
plt.figure()
plt.title("sino of reconstruction")
plt.imshow(sino_rec, cmap="gray", aspect="auto")
plt.colorbar()

plt.figure()
plt.title("sino")
plt.imshow(sino, cmap="gray", aspect="auto")
plt.colorbar()


plt.show()
