import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, corner_peaks
from scipy.ndimage import distance_transform_edt, convolve

print("loading data")
rec = np.load("data/ordinary_reconstruction.npy")

# print("downsampling")
# rec = rec[::2, ::2, ::2]

print("Otsu thresholding")
tres = threshold_otsu(rec)

print("binarizing")
binary = rec > tres

print("calculating Euclidean distance_transform")
dt = distance_transform_edt(binary)

print("getting centers")
#centers = peak_local_max(dt, min_distance=10)
centers = corner_peaks(dt, min_distance=10, threshold_rel=0)
np.save("data/centers", centers)
#centers = np.loadtxt("data/centers.txt").astype(np.int64)

# print("creating annotated volume")
# vol = np.zeros(dt.shape, dtype=bool)
# vol[tuple(centers.T)] = True
# vol = convolve(vol, np.ones((5,)*3))

# print("writing to harddisk")
# dxchange.write_tiff_stack(binary + vol.astype(np.float32), "/media/jens/JensHDD/LBFoam/centers/slice_")