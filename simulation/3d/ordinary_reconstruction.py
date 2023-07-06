import numpy as np
import astra
from matplotlib import pyplot as plt
from scipy.sparse.linalg import lsqr
from tqdm import tqdm


sino = np.load("data/sino.npy")

# create astra optomo operator
angles = np.linspace(0, 2*np.pi, sino.shape[0])
# sino = sino[::5]
# angles = angles[::5]
geom_settings = (1, 1, 256, 512, angles, 5000, 200)
vol_geom = astra.create_vol_geom(sino.shape[2], sino.shape[2], sino.shape[1])
proj_geom = astra.create_proj_geom('cone', *geom_settings)
proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
W = astra.optomo.OpTomo(proj_id)
p = sino.swapaxes(0,1).ravel()

# create reconstruction
rec_shape = (sino.shape[1], sino.shape[2], sino.shape[2])
rec_size = sino.shape[1] * sino.shape[2] * sino.shape[2]

# ordinary reconstruction
rec = lsqr(W, p, iter_lim=20, show=True)[0]
rec = rec.reshape((sino.shape[1], sino.shape[2], sino.shape[2]))


plt.figure()
plt.title("ordinary reconstruction")
plt.imshow(rec[rec.shape[0]//2, :, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("ordinary reconstruction")
plt.imshow(rec[:, rec.shape[1]//2, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("ordinary reconstruction")
plt.imshow(rec[:, :, rec.shape[2]//2 ], cmap="gray")
plt.colorbar()

plt.show()

np.save("data/ordinary_reconstruction", rec)