import numpy as np
import astra
from matplotlib import pyplot as plt
from scipy.sparse.linalg import lsqr

slice_nr = 66
#slice_nr = 366

# load sino
data = np.load("data/sino_with_paganin.npy")
#data = np.load("data/tomobank_foam_frame_3.npy")
#data = data[:, :, 260:-260]
sino = data[:, slice_nr, :]


# create astra optomo operator
angles = np.linspace(0, np.pi, sino.shape[0], endpoint=False)
vol_geom = astra.create_vol_geom(sino.shape[1], sino.shape[1])
proj_geom = astra.create_proj_geom('fanflat', 1, sino.shape[1], angles, 50000, 1)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
W = astra.optomo.OpTomo(proj_id)
p = sino.ravel()


# create reconstruction
rec_shape = (sino.shape[1], sino.shape[1])
rec_size = sino.shape[1] * sino.shape[1]
rec = lsqr(W, p, iter_lim=30, show=True)[0]
rec = rec.reshape((sino.shape[1], sino.shape[1]))


# remove background in sinogram
r=433
cx=645
cy=647.5
y, x = np.meshgrid(np.arange(rec.shape[0]),
    np.arange(rec.shape[1]),
    indexing="ij",
)
bg = rec.copy()
bg[(x-cx)**2 + (y-cy)**2 <= r**2] = 0.00032
p_no_bg = p - W @ bg.ravel()
sino_no_bg = p_no_bg.reshape(sino.shape)
np.save(f"data/sino_with_paganin_no_bg_slice{slice_nr}", sino_no_bg)


# create reconstruction without background
rec_no_bg = lsqr(W, p_no_bg, iter_lim=20, show=True)[0]
rec_no_bg = rec_no_bg.reshape((sino.shape[1], sino.shape[1]))


#plot
plt.figure()
plt.imshow(rec, cmap="gray")

plt.figure()
plt.imshow(bg, cmap="gray")

plt.figure()
plt.imshow(rec_no_bg, cmap="gray")

plt.figure()
plt.imshow(sino)

plt.figure()
plt.imshow(sino_no_bg)

#plt.figure()
#plt.plot(res)

plt.show()