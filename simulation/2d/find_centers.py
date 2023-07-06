import numpy as np
import astra
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, corner_peaks
from scipy.ndimage import distance_transform_edt, convolve


rec = np.load("data/ordinary_reconstruction.npy")

tres = threshold_otsu(rec)
binary = rec > tres

dt = distance_transform_edt(binary)

#centers = peak_local_max(dt, min_distance=10)
centers = corner_peaks(dt, min_distance=10, threshold_rel=0)

mask = np.zeros(dt.shape)
mask[tuple(centers.T)] = 1
mask = convolve(mask, np.ones((10,)*2))

np.savetxt("data/centers.txt", centers)

plt.figure()
plt.title("ordinary reconstruction")
plt.imshow(rec, cmap="gray")
plt.scatter(*centers[:,::-1].T)
plt.colorbar()

plt.figure()
plt.title("Otsu threshold")
plt.imshow(binary, cmap="gray")

plt.figure()
plt.title("Euclidean distance transform")
plt.imshow(dt, cmap="gray")
plt.colorbar()

plt.figure()
plt.title("mask")
plt.imshow(binary+mask, cmap="gray")
plt.colorbar()

plt.show()