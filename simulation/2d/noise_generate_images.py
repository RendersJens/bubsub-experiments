import numpy as np
from mesh_tomography.pixelation import pixelate
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from time import time
import matplotlib
matplotlib.rcParams.update({'errorbar.capsize': 2})

phantom = plt.imread("data/phantom512.png")[:, :, 0] > 0.5
errs = []
lsqr_errs = []
fbp_errs = []
for i in range(5):
    errs.append(np.load(f"data/noise_experiment_{i}/errs.npy"))
    lsqr_errs.append(np.load(f"data/noise_experiment_{i}/lsqr_errs.npy"))
    fbp_errs.append(np.load(f"data/noise_experiment_{i}/fbp_errs.npy"))
errs = np.array(errs)
lsqr_errs = np.array(lsqr_errs)
fbp_errs = np.array(fbp_errs)
vals = [round(10**power) for power in np.linspace(2, 5, 30)]

def to_yerr(arr):
    return arr.std(axis=0)
errorevery = 1
barsabove = True
ecolor = None

plt.figure(figsize=(4, 3))
# plt.errorbar(vals, fbp_errs.mean(axis=0),
#     yerr=to_yerr(fbp_errs),
#     ecolor=ecolor,
#     errorevery=errorevery,
#     barsabove=barsabove
# )
# plt.errorbar(vals, lsqr_errs.mean(axis=0),
#     yerr=to_yerr(lsqr_errs),
#     ecolor=ecolor,
#     errorevery=errorevery,
#     barsabove=barsabove
# )
# plt.errorbar(vals, errs.mean(axis=0),
#     yerr=to_yerr(errs),
#     ecolor=ecolor,
#     errorevery=errorevery,
#     barsabove=barsabove
# )
plt.plot(vals, fbp_errs.mean(axis=0), "--")
plt.plot(vals, lsqr_errs.mean(axis=0), "-.")
plt.plot(vals, errs.mean(axis=0))
plt.xscale("log")
plt.legend(["FBP", "LSQR", "BubSub"])
plt.xlabel("Photon count")
plt.ylabel("Dice dissimilarity")
plt.tight_layout()
plt.savefig("images/noise_err.pdf")

plt.figure(figsize=(4, 3))
# plt.errorbar(vals, fbp_errs.mean(axis=0),
#     yerr=to_yerr(fbp_errs),
#     ecolor=ecolor,
#     errorevery=errorevery,
#     barsabove=barsabove
# )
# plt.errorbar(vals, lsqr_errs.mean(axis=0),
#     yerr=to_yerr(lsqr_errs),
#     ecolor=ecolor,
#     errorevery=errorevery,
#     barsabove=barsabove
# )
# plt.errorbar(vals, errs.mean(axis=0),
#     yerr=to_yerr(errs),
#     ecolor=ecolor,
#     errorevery=errorevery,
#     barsabove=barsabove
# )
plt.plot(vals, fbp_errs.mean(axis=0), "--")
plt.plot(vals, lsqr_errs.mean(axis=0), "-.")
plt.plot(vals, errs.mean(axis=0))
plt.xscale("log")
plt.ylim((0.015, 0.08))
plt.legend(["FBP", "LSQR", "BubSub"])
plt.xlabel("Photon count")
plt.ylabel("Dice dissimilarity")
plt.tight_layout()
plt.savefig("images/noise_err_zoom.pdf")

plt.show()
