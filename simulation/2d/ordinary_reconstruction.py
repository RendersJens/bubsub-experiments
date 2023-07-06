import numpy as np
import astra
from matplotlib import pyplot as plt
from scipy.sparse.linalg import lsqr
from tqdm import tqdm


sino = np.load("data/sino.npy")

angles = np.linspace(0, np.pi, sino.shape[0], endpoint=False)
geom_settings = (1, sino.shape[1], angles, 50000, 1)

# create astra optomo operator
vol_geom = astra.create_vol_geom(sino.shape[1], sino.shape[1])
proj_geom = astra.create_proj_geom('fanflat', *geom_settings)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
W = astra.optomo.OpTomo(proj_id)
p = sino.ravel()

# ordinary reconstruction
rec = lsqr(W, p, iter_lim=20, show=True)[0]
rec = rec.reshape((sino.shape[1], sino.shape[1]))


plt.figure()
plt.title("ordinary reconstruction")
plt.imshow(rec, cmap="gray")
plt.colorbar()

plt.show()

np.save("data/ordinary_reconstruction", rec)