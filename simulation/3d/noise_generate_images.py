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
    errs.append(np.load(f"data/noise_experiment_{i}/errs.npy"))
    lsqr_errs.append(np.load(f"data/noise_experiment_{i}/lsqr_errs.npy"))
    fdk_errs.append(np.load(f"data/noise_experiment_{i}/fdk_errs.npy"))
errs = np.array(errs)
lsqr_errs = np.array(lsqr_errs)
fdk_errs = np.array(fdk_errs)
vals = [round(10**power) for power in np.linspace(2, 5, 30)]

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
print(fdk_errs.std(axis=0).max())
print(lsqr_errs.std(axis=0).max())
print(errs.std(axis=0).max())
plt.xscale("log")
plt.xlim(right=1e4)
plt.legend(["FDK", "LSQR", "BubSub"])
plt.xlabel("Photon count")
plt.ylabel("Dice dissimilarity")
plt.tight_layout()
plt.savefig("images/noise_err.eps")

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
plt.xscale("log")
plt.xlim(right=1e4)
plt.ylim((0.03, 0.15))
plt.legend(["FDK", "LSQR", "BubSub"])
plt.xlabel("Photon count")
plt.ylabel("Dice dissimilarity")
plt.tight_layout()
plt.savefig("images/noise_err_zoom.eps")

plt.show()
