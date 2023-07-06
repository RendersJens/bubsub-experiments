import astra
import numpy as np
import cupy as cp
import dxchange
import tomopy
from imwip import affine_warp_2D
from swapped_optomo import OpTomo
from imsolve.linear import BBLS
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, corner_peaks
from skimage.morphology import binary_dilation
from scipy.ndimage import distance_transform_edt, label
from skimage.color import label2rgb
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

path = "/media/jens/Samsung_T5/journal-paper-mesh-reconstructions/real/luflee/"

# data
ndark = 10
nflat = 10
nproj = 900
path = "/media/jens/Samsung_T5/journal-paper-mesh-reconstructions/real/luflee/"

print("""
+---------------+
| Preprocessing |
+---------------+
""")

print("reading data")
flat = dxchange.read_tiff_stack(path+"data/Tomo/flat_1.tif", range(1, nflat+1))
dark = dxchange.read_tiff_stack(path+"data/Tomo/dark_1.tif", range(1, ndark+1))
proj = dxchange.read_tiff_stack(path+"data/Tomo/TOMO_001.tif", range(1, nproj+1))

print("flat and dark field correction")
data = tomopy.normalize(proj, flat, dark)

# data1 = tomopy.remove_stripe_fw(data1,level=7,wname='sym16',sigma=1,pad=True)
# data = tomopy.remove_stripe_fw(data,level=7,wname='sym16',sigma=1,pad=True)

print("log transform")
data = tomopy.minus_log(data)
data = tomopy.remove_nan(data, val=0.0)
data = tomopy.remove_neg(data, val=0.0)
data[np.where(data == np.inf)] = 0.0

# print("downsampling")
#data = tomopy.downsample(data, level=2, axis=2)
#data = tomopy.downsample(data, level=2, axis=1)

theta = -np.linspace(0, np.pi, nproj,endpoint=False)

print("fixing rotation center")
center = tomopy.find_center_pc(data[0], data[-1], tol=0.1)
shift = center - data.shape[2]/2
print(shift)
for i in range(nproj):
    data[i] = affine_warp_2D(
        data[i],
        np.eye(2, dtype=np.float32),
        np.array([0, shift], dtype=np.float32)
    )
center = tomopy.find_center_pc(data[0], data[-1], tol=0.1)
shift = center - data.shape[2]/2
print(shift)

print("paganin filter")
data_paganin = tomopy.retrieve_phase(data, energy=16, dist=25, pixel_size=0.0003*4, alpha=0.001)

plt.figure("without paganin")
plt.imshow(data[0], cmap="gray")
plt.colorbar()

plt.figure("with paganin")
plt.imshow(data_paganin[0], cmap="gray")
plt.colorbar()

data = data_paganin

# create astra optomo operator
vol_geom = astra.create_vol_geom(data.shape[2], data.shape[2], data.shape[1])
proj_geom = astra.create_proj_geom('parallel3d', 1, 1, data.shape[1], data.shape[2], theta)
proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
W = OpTomo(proj_id)

p = data.ravel()

# print("reconstructing")
# rec = BBLS(W, p, bounds=(0, np.inf), max_iter=30, verbose=True)
# rec = rec.reshape((data.shape[1], data.shape[2], data.shape[2])).astype(np.float32)
# dxchange.write_tiff_stack(rec, path+"data/rec/slice", overwrite=True)

# plt.figure()
# plt.imshow(rec[rec.shape[0]//2, :, :], cmap="gray")

# plt.figure()
# plt.imshow(rec[:, rec.shape[1]//2, :], cmap="gray")

# plt.figure()
# plt.imshow(rec[:, :, rec.shape[2]//2], cmap="gray")

# plt.show()

# del rec

# rec = dxchange.read_tiff_stack(path + "data/rec/slice_00000.tiff", range(301))
# rec = rec/rec.max()
# thres = threshold_otsu(rec)
# binary = rec < thres

# dt = distance_transform_edt(binary)
# np.save(path + "data/dt", dt)
# dt = np.load(path + "data/dt.npy")

# centers = corner_peaks(dt, min_distance=3, threshold_rel=0)
# print(len(centers))

# mask = np.zeros(dt.shape, dtype=bool)
# mask[tuple(centers.T)] = True
# markers, _ = label(mask)
# del mask
# labels = watershed(-dt, markers, mask=dt>0, compactness=0)
# del markers
# del dt
# np.save(path + "data/labels", labels)


labels = np.load(path + "data/labels.npy")
# voxels_per_label = []
# for label in tqdm(range(len(centers)-1)):
#     voxels_per_label.append(cp.count_nonzero(labels==label).get())

# np.save(path + "data/voxels_per_label", np.array(voxels_per_label))
# voxels_per_label = np.load(path + "data/voxels_per_label.npy")

roi = np.zeros(labels.shape, dtype=bool)
roi[2:-2, 700:980, 550:900] = True
# roi_labels = labels[2:-2, 700:980, 550:900].copy()
# labels = labels.get()
# incomplete_bubbles = []
# for label in tqdm(range(len(centers)-1)):
#     if np.count_nonzero(roi_labels == label) < voxels_per_label[label]:
#         incomplete_bubbles.append(label)

# np.save(path + "data/incomplete_bubbles", np.array(incomplete_bubbles))
# incomplete_bubbles = np.load(path + "data/incomplete_bubbles.npy")

# labels = cp.asarray(labels)
# roi_labels = cp.zeros_like(labels)
# for bubble in tqdm(set(range(len(centers)-1)) - set(incomplete_bubbles)):
#     cp.place(roi_labels, labels==bubble, bubble)
# roi_labels = roi_labels.get()
# labels = labels.get()
# np.save(path + "data/roi_labels", roi_labels)
roi_labels = np.load(path + "data/roi_labels.npy")


rec = dxchange.read_tiff_stack(path + "data/rec/slice_00000.tiff", range(301))
bg = np.ones_like(rec)*0.003522
bg[~roi] = rec[~roi]
partial_bubbles = np.logical_and(labels!=0, roi_labels==0)
for i in tqdm(range(3)):
    partial_bubbles = binary_dilation(partial_bubbles)
bg[partial_bubbles] = rec[partial_bubbles]

colors = np.random.rand(256,3)
colors[0] = 0
cmap = ListedColormap(colors)

labels[roi] += 500

fig, ax = plt.subplots(1, 3)
ax[0].imshow(labels[labels.shape[0]//2, :, :], interpolation="nearest", cmap=cmap)
ax[1].imshow(labels[:, labels.shape[1]//2, :], interpolation="nearest", cmap=cmap)
ax[2].imshow(labels[:, :, labels.shape[2]//2], interpolation="nearest", cmap=cmap)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(partial_bubbles[partial_bubbles.shape[0]//2, :, :])
ax[1].imshow(partial_bubbles[:, partial_bubbles.shape[1]//2, :])
ax[2].imshow(partial_bubbles[:, :, partial_bubbles.shape[2]//2])

goal = rec - bg

sino_bg = W @ bg.ravel()
sino_no_bg = data - sino_bg.reshape(data.shape)
np.save(path+"data/sino", sino_no_bg)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(goal[rec.shape[0]//2, :, :], cmap="gray")
ax[1].imshow(goal[:, rec.shape[1]//2, :], cmap="gray")
ax[2].imshow(goal[:, :, rec.shape[2]//2], cmap="gray")

plt.show()