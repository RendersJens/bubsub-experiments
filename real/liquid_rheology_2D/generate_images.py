import numpy as np
from mesh_projector.pixelation import pixelate
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt, label
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from time import time

colors = np.random.rand(256,3)
colors[0] = 0
cmap = ListedColormap(colors)
centers = np.loadtxt("data/centers.txt").astype(np.int64)

lsqr_rec0 = np.load(f"data/lsqr_rec{300}.npy")
fbp_rec0 = np.load(f"data/fbp_rec{300}.npy")

lsqr_tres = threshold_otsu(lsqr_rec0)
fbp_tres = threshold_otsu(fbp_rec0)

lsqr_rec = lsqr_rec0 < lsqr_tres
#lsqr_rec = binary_opening(lsqr_rec)

fbp_rec = fbp_rec0 < fbp_tres
#fbp_rec = binary_opening(fbp_rec)


rec_polys = np.load(f"data/rec{300}.npy")
rec = pixelate(
    rec_polys[0],
    np.array([-lsqr_rec.shape[0]//2, lsqr_rec.shape[0]//2]),
    np.array([-lsqr_rec.shape[0]//2, lsqr_rec.shape[0]//2]),
)*0
for poly in rec_polys:
    pixel_poly = pixelate(
        poly,
        np.array([-lsqr_rec.shape[0]//2, lsqr_rec.shape[0]//2]),
        np.array([-lsqr_rec.shape[0]//2, lsqr_rec.shape[0]//2]),
    )
    rec += pixel_poly
rec = np.flipud(np.fliplr(rec)).astype(np.bool)

dt = distance_transform_edt(lsqr_rec)
mask = np.zeros(dt.shape, dtype=bool)
mask[tuple(centers[:,::-1].T)] = True
markers, _ = label(mask)
lsqr_labels = watershed(-dt, markers, mask=lsqr_rec, compactness=0)

dt = distance_transform_edt(fbp_rec)
mask = np.zeros(dt.shape, dtype=bool)
mask[tuple(centers[:,::-1].T)] = True
markers, _ = label(mask)
fbp_labels = watershed(-dt, markers, mask=fbp_rec, compactness=0)

dt = distance_transform_edt(rec)
mask = np.zeros(dt.shape, dtype=bool)
mask[tuple(centers[:,::-1].T)] = True
markers, _ = label(mask)
rec_labels = watershed(-dt, markers, mask=rec, compactness=0)

plt.figure()
plt.imshow(rec, cmap="gray")
plt.imsave("images/rec300.png", rec, cmap="gray")

plt.figure()
plt.imshow(lsqr_rec0, cmap="gray")
plt.imsave("images/lsqr_rec300.png", lsqr_rec, cmap="gray")

# plt.figure()
# plt.imshow(fbp_rec, cmap="gray")
# plt.imsave("images/fbp_rec300.png", fbp_rec, cmap="gray")

plt.figure()
plt.imshow(lsqr_labels, cmap=cmap)
plt.imsave("images/lsqr_labels300.png", lsqr_labels, cmap=cmap)

# plt.figure()
# plt.imshow(fbp_labels, cmap=cmap)
# plt.imsave("images/fbp_labels300.png", fbp_labels, cmap=cmap)

plt.figure()
plt.imshow(rec_labels, cmap=cmap)
plt.imsave("images/rec_labels300.png", rec_labels, cmap=cmap)


lsqr_rec0 = np.load(f"data/lsqr_rec{60}.npy")
fbp_rec0 = np.load(f"data/fbp_rec{60}.npy")

lsqr_tres = threshold_otsu(lsqr_rec0)
fbp_tres = threshold_otsu(fbp_rec0)

lsqr_rec = lsqr_rec0 < lsqr_tres
#lsqr_rec = binary_opening(lsqr_rec)

fbp_rec = fbp_rec0 < lsqr_tres
#fbp_rec = binary_opening(fbp_rec)


rec_polys = np.load(f"data/rec{60}.npy")
rec = pixelate(
    rec_polys[0],
    np.array([-lsqr_rec.shape[0]//2, lsqr_rec.shape[0]//2]),
    np.array([-lsqr_rec.shape[0]//2, lsqr_rec.shape[0]//2]),
)*0
for poly in rec_polys:
    pixel_poly = pixelate(
        poly,
        np.array([-lsqr_rec.shape[0]//2, lsqr_rec.shape[0]//2]),
        np.array([-lsqr_rec.shape[0]//2, lsqr_rec.shape[0]//2]),
    )
    rec += pixel_poly
rec = np.flipud(np.fliplr(rec)).astype(np.bool)

dt = distance_transform_edt(lsqr_rec)
mask = np.zeros(dt.shape, dtype=bool)
mask[tuple(centers[:,::-1].T)] = True
markers, _ = label(mask)
lsqr_labels = watershed(-dt, markers, mask=lsqr_rec, compactness=0)

dt = distance_transform_edt(fbp_rec)
mask = np.zeros(dt.shape, dtype=bool)
mask[tuple(centers[:,::-1].T)] = True
markers, _ = label(mask)
fbp_labels = watershed(-dt, markers, mask=fbp_rec, compactness=0)

dt = distance_transform_edt(rec)
mask = np.zeros(dt.shape, dtype=bool)
mask[tuple(centers[:,::-1].T)] = True
markers, _ = label(mask)
rec_labels = watershed(-dt, markers, mask=rec, compactness=0)

plt.figure()
plt.imshow(rec, cmap="gray")
plt.imsave("images/rec60.png", rec, cmap="gray")

plt.figure()
plt.imshow(lsqr_rec, cmap="gray")
plt.imsave("images/lsqr_rec60.png", lsqr_rec, cmap="gray")

# plt.figure()
# plt.imshow(fbp_rec, cmap="gray")
# plt.imsave("images/fbp_rec60.png", fbp_rec, cmap="gray")

plt.figure()
plt.imshow(lsqr_labels, cmap=cmap)
plt.imsave("images/lsqr_labels60.png", lsqr_labels, cmap=cmap)

# plt.figure()
# plt.imshow(fbp_labels, cmap=cmap)
# plt.imsave("images/fbp_labels60.png", fbp_labels, cmap=cmap)

plt.figure()
plt.imshow(rec_labels, cmap=cmap)
plt.imsave("images/rec_labels60.png", rec_labels, cmap=cmap)


plt.show()