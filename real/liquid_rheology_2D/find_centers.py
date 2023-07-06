import numpy as np
import astra
from matplotlib import pyplot as plt
from scipy.sparse.linalg import lsqr
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, corner_peaks
from scipy.ndimage import distance_transform_edt, convolve



rec = np.load(f"data/lsqr_rec{300}.npy")
tres = threshold_otsu(rec)
binary = rec < tres

dt = distance_transform_edt(binary)

centers = corner_peaks(dt, min_distance=10, threshold_rel=0)
np.savetxt("data/centers.txt", centers[:,::-1])

plt.figure()
plt.title("initial reconstruction")
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

plt.show()