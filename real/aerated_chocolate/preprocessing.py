import astra
import numpy as np
import dxchange
import tomopy
from imwip import affine_warp_2D
from swapped_optomo import OpTomo
from imsolve.linear import BBLS
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, corner_peaks
from scipy.ndimage import distance_transform_edt, label
from skimage.color import label2rgb
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

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
center = tomopy.find_center(data, theta)[0]
shift = center - data.shape[2]/2
print(shift)
for i in range(nproj):
    data[i] = affine_warp_2D(
        data[i],
        np.eye(2, dtype=np.float32),
        np.array([0, shift], dtype=np.float32)
    )
center = tomopy.find_center(data, theta)[0]
shift = center - data.shape[2]/2
print(shift)

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
# del rec

rec = dxchange.read_tiff_stack(path + "data/rec/slice_00000.tiff", range(301))[::2, ::2, ::2]
rec = rec/rec.max()
thres = threshold_otsu(rec)
binary = rec < thres

dt = distance_transform_edt(binary)

centers = corner_peaks(dt, min_distance=8, threshold_rel=0)
print(len(centers))
mask = np.zeros(dt.shape, dtype=bool)
mask[tuple(centers.T)] = True
markers, _ = label(mask)
del mask
labels = watershed(-dt, markers, mask=binary, compactness=0)
del dt
del markers
del binary

rec = dxchange.read_tiff_stack(path + "data/rec/slice_00000.tiff", range(301))
mask = labels == labels[0, 0, 0]
mask = mask.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
mask = mask[:-1, :-1, :-1]
bg = rec.copy()
bg[~mask] = 0.0034

colors = np.random.rand(256,3)
colors[0] = 0
cmap = ListedColormap(colors)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(labels[rec.shape[0]//2, :, :], interpolation="nearest", cmap=cmap)
ax[1].imshow(labels[:, rec.shape[1]//2, :], interpolation="nearest", cmap=cmap)
ax[2].imshow(labels[:, :, rec.shape[2]//2], interpolation="nearest", cmap=cmap)
goal = rec - bg

sino_bg = W @ bg.ravel()
sino_no_bg = data - sino_bg.reshape(data.shape)
np.save(path+"data/sino", sino_no_bg)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(goal[rec.shape[0]//2, :, :], cmap="gray")
ax[1].imshow(goal[:, rec.shape[1]//2, :], cmap="gray")
ax[2].imshow(goal[:, :, rec.shape[2]//2], cmap="gray")

plt.show()