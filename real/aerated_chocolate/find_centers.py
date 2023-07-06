import numpy as np
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max, corner_peaks
from scipy.ndimage import distance_transform_edt, convolve
import dxchange

path = "/media/jens/Samsung_T5/journal-paper-mesh-reconstructions/real/luflee/"

print("loading data")
rec = np.load(path + "data/ordinary_reconstruction.npy")

# print("downsampling")
# rec = rec[::2, ::2, ::2]

print("Otsu thresholding")
tres = threshold_otsu(rec)

print("binarizing")
binary = rec < tres

print("calculating Euclidean distance_transform")
dt = distance_transform_edt(binary)

print("getting centers")
#centers = peak_local_max(dt, min_distance=10)
centers = corner_peaks(dt, min_distance=3, threshold_rel=0)
radii = np.array([dt[tuple(center)] for center in centers])
print(len(centers))
np.save("data/centers", centers)
np.save("data/radii", radii)
#centers = np.loadtxt("data/centers.txt").astype(np.int64)

# print("creating annotated volume")
# vol = np.zeros(dt.shape, dtype=bool)
# vol[tuple(centers.T)] = True
# vol = convolve(vol, np.ones((5,)*3))

# print("writing to harddisk")
# dxchange.write_tiff_stack(
#     binary + vol.astype(np.float32),
#     "data/centers_vol/slice_",
#     overwrite=True
# )