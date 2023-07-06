import numpy as np
from mesh_tomography.pixelation import pixelate
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening
from skimage.measure import block_reduce
from scipy.spatial.distance import dice
import matplotlib.pyplot as plt
from time import time
import sys

realisation = sys.argv[1]

phantom = plt.imread("data/phantom512.png")[:, :, 0] > 0.5
vals = [round(10**power) for power in np.linspace(2, 5, 30)]

errs = []
lsqr_errs = []
fbp_errs = []

for I0 in vals:
    lsqr_rec = np.load(f"data/noise_experiment_{realisation}/lsqr_rec{I0}.npy")
    fbp_rec = np.load(f"data/noise_experiment_{realisation}/fbp_rec{I0}.npy")
    print("Otsu thresholding")
    lsqr_tres = threshold_otsu(lsqr_rec)
    fbp_tres = threshold_otsu(fbp_rec)

    print("binarizing")
    lsqr_rec = lsqr_rec > lsqr_tres
    lsqr_rec = binary_opening(lsqr_rec)

    fbp_rec = fbp_rec > fbp_tres
    fbp_rec = binary_opening(fbp_rec)

    rec_polys = np.load(f"data/noise_experiment_{realisation}/rec{I0}.npy")*2
    rec = pixelate(
        rec_polys[0],
        np.array([-phantom.shape[0], phantom.shape[0]]),
        np.array([-phantom.shape[0], phantom.shape[0]]),
    )*0
    for poly in rec_polys:
        pixel_poly = pixelate(
            poly,
            np.array([-phantom.shape[0], phantom.shape[0]]),
            np.array([-phantom.shape[0], phantom.shape[0]]),
        )
        rec += pixel_poly
    rec = np.fliplr(rec.T)
    rec = block_reduce(rec, (2,2), np.min)

    rnmp = np.logical_xor(rec, phantom).sum()/phantom.sum()
    mesh_dice = dice(rec.ravel(), phantom.ravel())
    errs.append(mesh_dice)

    lsqr_rnmp = np.logical_xor(lsqr_rec, phantom).sum()/phantom.sum()
    lsqr_dice = dice(lsqr_rec.ravel(), phantom.ravel())
    lsqr_errs.append(lsqr_dice)

    fbp_rnmp = np.logical_xor(fbp_rec, phantom).sum()/phantom.sum()
    fbp_dice = dice(fbp_rec.ravel(), phantom.ravel())
    fbp_errs.append(fbp_dice)

    print(f"bubble fit error at {I0} photons:", mesh_dice)
    print(f"lsqr error at {I0} photons:", lsqr_dice)
    print(f"fbp error at {I0} photons:", fbp_dice)
    # if n_proj == 20:
    #     plt.figure()
    #     plt.imshow(rec)
    #     plt.figure()
    #     plt.imshow(lsqr_rec)
    #     plt.figure()
    #     plt.imshow(fbp_rec)
    #     plt.show()

np.save(f"data/noise_experiment_{realisation}/errs", np.array(errs))
np.save(f"data/noise_experiment_{realisation}/lsqr_errs", np.array(lsqr_errs))
np.save(f"data/noise_experiment_{realisation}/fbp_errs", np.array(fbp_errs))
plt.figure()
plt.plot(vals, errs)
plt.plot(vals, lsqr_errs)
plt.plot(vals, fbp_errs)
plt.xscale("log")
#plt.yscale("log")
# plt.show()
