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
for i in range(10):
    errs.append(np.load(f"data/n_proj_experiment_{i}/errs.npy"))
    lsqr_errs.append(np.load(f"data/n_proj_experiment_{i}/lsqr_errs.npy"))
    fbp_errs.append(np.load(f"data/n_proj_experiment_{i}/fbp_errs.npy"))
errs = np.array(errs)
lsqr_errs = np.array(lsqr_errs)
fbp_errs = np.array(fbp_errs)
vals = list(range(10, 310, 10))

def to_yerr(arr):
    return arr.std(axis=0)
errorevery = 2
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
#plt.yscale("log")
plt.legend(["FBP", "LSQR", "BubSub"])
plt.xlabel("Number of projections")
plt.ylabel("Dice dissimilarity")
plt.tight_layout()
plt.savefig("images/n_proj_err.pdf")

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
plt.ylim((0.014, 0.04))
plt.legend(["FBP", "LSQR", "BubSub"])
plt.xlabel("Number of projections")
plt.ylabel("Dice dissimilarity")
plt.tight_layout()
plt.savefig("images/n_proj_err_zoom.pdf")




# rec_polys = np.load(f"data/n_proj_experiment_0/rec{10}.npy")*2
# rec = pixelate(
#     rec_polys[0],
#     np.array([-phantom.shape[0], phantom.shape[0]]),
#     np.array([-phantom.shape[0], phantom.shape[0]]),
# )*0
# for poly in rec_polys:
#     pixel_poly = pixelate(
#         poly,
#         np.array([-phantom.shape[0], phantom.shape[0]]),
#         np.array([-phantom.shape[0], phantom.shape[0]]),
#     )
#     rec += pixel_poly
# rec = np.fliplr(rec.T)
# rec = block_reduce(rec, (2,2), np.min)

# lsqr_rec = np.load(f"data/n_proj_experiment_0/lsqr_rec{10}.npy")

# plt.figure()
# plt.imshow(rec, cmap="gray")
# plt.imsave("images/rec10.png", rec, cmap="gray")

# plt.figure()
# plt.imshow(lsqr_rec, cmap="gray")
# plt.imsave("images/lsqr_rec10.png", lsqr_rec, cmap="gray")

# rec_polys = np.load(f"data/n_proj_experiment_0/rec{20}.npy")*2
# rec = pixelate(
#     rec_polys[0],
#     np.array([-phantom.shape[0], phantom.shape[0]]),
#     np.array([-phantom.shape[0], phantom.shape[0]]),
# )*0
# for poly in rec_polys:
#     pixel_poly = pixelate(
#         poly,
#         np.array([-phantom.shape[0], phantom.shape[0]]),
#         np.array([-phantom.shape[0], phantom.shape[0]]),
#     )
#     rec += pixel_poly
# rec = np.fliplr(rec.T)
# rec = block_reduce(rec, (2,2), np.min)

# lsqr_rec = np.load(f"data/n_proj_experiment_0/lsqr_rec{20}.npy")

# plt.figure()
# plt.imshow(rec, cmap="gray")
# plt.imsave("images/rec20.png", rec, cmap="gray")

# plt.figure()
# plt.imshow(lsqr_rec, cmap="gray")
# plt.imsave("images/lsqr_rec20.png", lsqr_rec, cmap="gray")






# rec_polys = np.load(f"data/n_proj_experiment_0/rec{50}.npy")*2
# rec = pixelate(
#     rec_polys[0],
#     np.array([-phantom.shape[0], phantom.shape[0]]),
#     np.array([-phantom.shape[0], phantom.shape[0]]),
# )*0
# for poly in rec_polys:
#     pixel_poly = pixelate(
#         poly,
#         np.array([-phantom.shape[0], phantom.shape[0]]),
#         np.array([-phantom.shape[0], phantom.shape[0]]),
#     )
#     rec += pixel_poly
# rec = np.fliplr(rec.T)
# rec = block_reduce(rec, (2,2), np.min)

# lsqr_rec = np.load(f"data/n_proj_experiment_0/lsqr_rec{50}.npy")
# fbp_rec = np.load(f"data/n_proj_experiment_0/fbp_rec{50}.npy")

# lsqr_tres = threshold_otsu(lsqr_rec)
# fbp_tres = threshold_otsu(fbp_rec)

# lsqr_rec = lsqr_rec > lsqr_tres
# lsqr_rec = binary_opening(lsqr_rec)

# fbp_rec = fbp_rec > fbp_tres
# fbp_rec = binary_opening(fbp_rec)

# plt.figure()
# plt.imshow(rec, cmap="gray")
# plt.imsave("images/rec50.png", rec, cmap="gray")

# plt.figure()
# plt.imshow(lsqr_rec, cmap="gray")
# plt.imsave("images/lsqr_rec50.png", lsqr_rec, cmap="gray")

# plt.figure()
# plt.imshow(fbp_rec, cmap="gray")
# plt.imsave("images/fbp_rec50.png", fbp_rec, cmap="gray")

# plt.figure()
# plt.imshow(phantom, cmap="gray")
# plt.imsave("images/phantom.png", phantom, cmap="gray")

# rec_diff = np.logical_xor(rec, phantom)
# lsqr_rec_diff = np.logical_xor(lsqr_rec, phantom)
# fbp_rec_diff = np.logical_xor(fbp_rec, phantom)
# phantom_diff = np.logical_xor(phantom, phantom)

# plt.figure()
# plt.imshow(rec_diff, cmap="gray")
# plt.imsave("images/rec_diff50.png", rec_diff, cmap="gray")

# plt.figure()
# plt.imshow(lsqr_rec_diff, cmap="gray")
# plt.imsave("images/lsqr_rec_diff50.png", lsqr_rec_diff, cmap="gray")

# plt.figure()
# plt.imshow(fbp_rec_diff, cmap="gray")
# plt.imsave("images/fbp_rec_diff50.png", fbp_rec_diff, cmap="gray")

# plt.figure()
# plt.imshow(phantom_diff, cmap="gray")
# plt.imsave("images/phantom_diff.png", phantom_diff, cmap="gray")

plt.show()
