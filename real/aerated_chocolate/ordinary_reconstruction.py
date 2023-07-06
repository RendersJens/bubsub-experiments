import numpy as np
import astra
import dxchange
from matplotlib import pyplot as plt
from imsolve.linear import BBLS
from tqdm import tqdm

path = "/media/jens/Samsung_T5/journal-paper-mesh-reconstructions/real/luflee/"

sino = np.load(path + "data/sino.npy")[:, :, 500:-500]

# create astra optomo operator
angles = -np.linspace(0, np.pi, sino.shape[0], endpoint=False)
# sino = sino[::5]
# angles = angles[::5]
geom_settings = (
    1, 1,
    sino.shape[1],
    sino.shape[2], 
    angles,
    50000,
    1
)
vol_geom = astra.create_vol_geom(sino.shape[2], sino.shape[2], sino.shape[1])
proj_geom = astra.create_proj_geom('cone', *geom_settings)
proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
W = astra.optomo.OpTomo(proj_id)
p = sino.swapaxes(0,1).ravel()

# create reconstruction
rec_shape = (sino.shape[1], sino.shape[2], sino.shape[2])
rec_size = sino.shape[1] * sino.shape[2] * sino.shape[2]

# ordinary reconstruction
rec = BBLS(W, p, bounds=(-np.inf, 0), max_iter=30, verbose=True)
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

np.save(path + "data/ordinary_reconstruction", rec)