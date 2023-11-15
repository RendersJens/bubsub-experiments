import numpy as np
import astra
from matplotlib import pyplot as plt
from scipy.sparse.linalg import lsqr
from tqdm import tqdm

# load sino
sino = np.load("data/sino_with_paganin.npy")


# create astra optomo operator
angles = np.linspace(0, np.pi, sino.shape[0], endpoint=False)
vol_geom = astra.create_vol_geom(sino.shape[2], sino.shape[2], sino.shape[1])
proj_geom = astra.create_proj_geom('cone', 1, 1, sino.shape[1], sino.shape[2], angles, 50000, 1)
proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
W = astra.optomo.OpTomo(proj_id)
p = sino.swapaxes(0,1).ravel()


# load reconstruction
rec = np.load("data/initial_rec_with_background.npy")

# remove background in sinogram
cx = 645/2 + 258/2
cy = 647.5/2 + 263/2
y, x = np.meshgrid(np.arange(rec.shape[1]),
    np.arange(rec.shape[2]),
    indexing="ij",
)
bg = rec.copy()
for z in tqdm(range(bg.shape[0])):
    if z <= 460/2:
        a = z/(460/2)
        r = a * 444/2 + (1-a) * 770/2

    bg_slice = bg[z, :, :]
    bg_slice[(x-cx)**2 + (y-cy)**2 <= r**2] = 0.00032*2

p_no_bg = p - W @ bg.ravel()
sino_no_bg = p_no_bg.reshape(sino.swapaxes(0,1).shape).swapaxes(0,1)
np.save("data/sino", sino_no_bg)
np.save("data/initial_rec_without_background.npy", rec - bg)

plt.figure()
plt.imshow(bg[bg.shape[0]//2,:,:], cmap="gray")

plt.figure()
plt.imshow(bg[:,bg.shape[1]//2,:], cmap="gray")

plt.figure()
plt.imshow(bg[:,:,bg.shape[2]//2], cmap="gray")

del sino
del rec
del bg

# create reconstruction without background
rec_no_bg = lsqr(W, p_no_bg, iter_lim=20, show=True)[0]
rec_no_bg = rec_no_bg.reshape((sino_no_bg.shape[1], sino_no_bg.shape[2], sino_no_bg.shape[2]))
np.save("data/initial_rec.npy", rec_no_bg)

#plot
# plt.figure()
# plt.imshow(rec[rec.shape[0]//2,:,:], cmap="gray")

# plt.figure()
# plt.imshow(rec[:,rec.shape[1]//2,:], cmap="gray")

# plt.figure()
# plt.imshow(rec[:,:,rec.shape[2]//2], cmap="gray")

# plt.figure()
# plt.imshow(bg[bg.shape[0]//2,:,:], cmap="gray")

# plt.figure()
# plt.imshow(bg[:,bg.shape[1]//2,:], cmap="gray")

# plt.figure()
# plt.imshow(bg[:,:,bg.shape[2]//2], cmap="gray")

plt.figure()
plt.imshow(rec_no_bg[rec_no_bg.shape[0]//2,:,:], cmap="gray")

plt.figure()
plt.imshow(rec_no_bg[:,rec_no_bg.shape[1]//2,:], cmap="gray")

plt.figure()
plt.imshow(rec_no_bg[:,:,rec_no_bg.shape[2]//2], cmap="gray")

# plt.figure()
# plt.imshow(sino)

# plt.figure()
# plt.imshow(sino_no_bg)

plt.show()
