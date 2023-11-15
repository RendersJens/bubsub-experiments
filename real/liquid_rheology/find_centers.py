import numpy as np
from scipy.sparse.linalg import lsqr
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import corner_peaks
from scipy.ndimage import distance_transform_edt, convolve
import dxchange

print("loading initial rec")
rec = -np.load("data/initial_rec_without_background.npy")

print("Otsu thresholding")
tres = threshold_otsu(rec)

print("binarizing")
binary = rec > tres

print("calculating Euclidean distance_transform")
dt = distance_transform_edt(binary)

print("getting centers")
centers = corner_peaks(dt, min_distance=10, threshold_rel=0)
np.save("data/centers", centers)

print("creating annotated volume")
vol = np.zeros(dt.shape, dtype=bool)
vol[tuple(centers.T)] = True
vol = convolve(vol, np.ones((5,)*3))

print("writing to harddisk")
dxchange.write_tiff_stack(binary + vol.astype(np.float32), "data/centers_visualisation/slice_")