import numpy as np
from scipy.sparse.linalg import lsqr
from skimage.filters import threshold_otsu
import dxchange

print("loading initial rec")
rec = -np.load("data/initial_rec_without_background.npy")

# print("Otsu thresholding")
# tres = threshold_otsu(rec)

# print("binarizing")
# binary = rec > tres

print("writing to harddisk")
dxchange.write_tiff_stack(rec.astype(np.float32), "data/ordinary_reconstruction_60/slice_")