import numpy as np
import trimesh
from mesh_tomography.voxelization import voxelize
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening
import matplotlib.pyplot as plt
from time import time
import matplotlib
import matplotlib.style
import matplotlib as mpl
matplotlib.rcParams.update({'errorbar.capsize': 5})
mpl.style.use('ggplot')


gt_mesh = trimesh.load("data/bucket3D2000.stl")
gt_mesh.vertices *= 180
gt_vol = voxelize(trimesh.graph.split(gt_mesh), (512, 512, 256))
gt_vol = np.flip(np.flip(gt_vol.T.swapaxes(1,2), 2), 0)

errs = []
lsqr_errs = []
fdk_errs = []
for i in range(10):
    errs.append(np.load(f"data/n_proj_experiment_{i}/errs.npy"))
    lsqr_errs.append(np.load(f"data/n_proj_experiment_{i}/lsqr_errs.npy"))
    fdk_errs.append(np.load(f"data/n_proj_experiment_{i}/fdk_errs.npy"))
errs = np.array(errs)
lsqr_errs = np.array(lsqr_errs)
fdk_errs = np.array(fdk_errs)
vals = list(range(3, 103, 6))

def to_yerr(arr):
    return 3*arr.std(axis=0)
errorevery = 2
barsabove = True
ecolor = None

plt.figure(figsize=(4, 3))
plt.errorbar(vals, fdk_errs.mean(axis=0), ls="--",
    yerr=to_yerr(fdk_errs),
    ecolor=ecolor,
    errorevery=errorevery,
    barsabove=barsabove
)
plt.errorbar(vals, lsqr_errs.mean(axis=0), ls="-.",
    yerr=to_yerr(lsqr_errs),
    ecolor=ecolor,
    errorevery=errorevery,
    barsabove=barsabove
)
plt.errorbar(vals, errs.mean(axis=0),
    yerr=to_yerr(errs),
    ecolor=ecolor,
    errorevery=errorevery,
    barsabove=barsabove
)
# plt.plot(vals, fdk_errs.mean(axis=0), "--")
# plt.plot(vals, lsqr_errs.mean(axis=0), "-.")
# plt.plot(vals, errs.mean(axis=0))
plt.legend(["FDK", "LSQR", "BubSub"])
plt.xlabel("Number of projections")
plt.ylabel("Dice dissimilarity")
plt.tight_layout()
plt.savefig("images/n_proj_err.eps")

plt.figure(figsize=(4, 3))
plt.errorbar(vals, fdk_errs.mean(axis=0), ls="--",
    yerr=to_yerr(fdk_errs),
    ecolor=ecolor,
    errorevery=errorevery,
    barsabove=barsabove
)
plt.errorbar(vals, lsqr_errs.mean(axis=0), ls="-.",
    yerr=to_yerr(lsqr_errs),
    ecolor=ecolor,
    errorevery=errorevery,
    barsabove=barsabove
)
plt.errorbar(vals, errs.mean(axis=0),
    yerr=to_yerr(errs),
    ecolor=ecolor,
    errorevery=errorevery,
    barsabove=barsabove
)
# plt.plot(vals, fdk_errs.mean(axis=0), "--")
# plt.plot(vals, lsqr_errs.mean(axis=0), "-.")
# plt.plot(vals, errs.mean(axis=0))
print(fdk_errs.std(axis=0).max())
print(lsqr_errs.std(axis=0).max())
print(errs.std(axis=0).max())
plt.ylim((0.03, 0.07))
plt.legend(["FDK", "LSQR", "BubSub"])
plt.xlabel("Number of projections")
plt.ylabel("Dice dissimilarity")
plt.tight_layout()
plt.savefig("images/n_proj_err_zoom.eps")


# lsqr_rec = np.load(f"data/noise_experiment_0/lsqr_rec{1083}.npy")
# fdk_rec = np.load(f"data/noise_experiment_0/fdk_rec{1083}.npy")
# print("Otsu thresholding")
# lsqr_tres = threshold_otsu(np.clip(lsqr_rec, 0, 1/180))
# fdk_tres = threshold_otsu(np.clip(fdk_rec, 0, 1/180))
# print("binarizing")
# lsqr_rec = lsqr_rec > lsqr_tres
# lsqr_rec = binary_opening(lsqr_rec)
# fdk_rec = fdk_rec > fdk_tres
# fdk_rec = binary_opening(fdk_rec)
# rec_mesh = trimesh.load(f"data/noise_experiment_0/rec{1083}.stl")
# rec = voxelize(trimesh.graph.split(rec_mesh), (512, 512, 256))
# rec = np.flip(rec.T, 0)

# plt.figure()
# plt.imshow(gt_vol[:, rec.shape[1]//2, :], cmap="gray")
# plt.imsave("images/gt.png", gt_vol[:, rec.shape[1]//2, :], cmap="gray")

# plt.figure()
# plt.imshow(rec[:, rec.shape[1]//2, :], cmap="gray")
# plt.imsave("images/rec25.png", rec[:, rec.shape[1]//2, :], cmap="gray")

# plt.figure()
# plt.imshow(lsqr_rec[:, rec.shape[1]//2, :], cmap="gray")
# plt.imsave("images/lsqr_rec25.png", lsqr_rec[:, rec.shape[1]//2, :], cmap="gray")

# plt.figure()
# plt.imshow(fdk_rec[:, rec.shape[1]//2, :], cmap="gray")
# plt.imsave("images/fdk_rec25.png", fdk_rec[:, rec.shape[1]//2, :], cmap="gray")

# rec_diff = np.logical_xor(rec, gt_vol)
# lsqr_rec_diff = np.logical_xor(lsqr_rec, gt_vol)
# fdk_rec_diff = np.logical_xor(fdk_rec, gt_vol)
# gt_diff = np.logical_xor(gt_vol, gt_vol)

# plt.figure()
# plt.imshow(gt_diff[:, rec.shape[1]//2, :], cmap="gray")
# plt.imsave("images/gt_diff.png", gt_diff[:, rec.shape[1]//2, :], cmap="gray")

# plt.figure()
# plt.imshow(rec_diff[:, rec.shape[1]//2, :], cmap="gray")
# plt.imsave("images/rec_diff25.png", rec_diff[:, rec.shape[1]//2, :], cmap="gray")

# plt.figure()
# plt.imshow(lsqr_rec_diff[:, rec.shape[1]//2, :], cmap="gray")
# plt.imsave("images/lsqr_rec_diff25.png", lsqr_rec_diff[:, rec.shape[1]//2, :], cmap="gray")

# plt.figure()
# plt.imshow(fdk_rec_diff[:, rec.shape[1]//2, :], cmap="gray")
# plt.imsave("images/fdk_rec_diff25.png", fdk_rec_diff[:, rec.shape[1]//2, :], cmap="gray")

# plt.show()
