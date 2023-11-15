import numpy as np
import h5py
import tomopy
import astra
from matplotlib import pyplot as plt
from scipy.sparse.linalg import lsqr


# settings
frame = 3
z_start = 1200
z_stop = 1800
y_start = 100
y_stop = 2016-100

# data
nproj = 300
path = "/media/jens/Samsung_T5/tomobank_foam/"
fname = path + "dk_MCFG_1_p_s1_.h5"

print("""
+---------------+
| Preprocessing |
+---------------+
""")

print("reading data")
with h5py.File(fname, "r") as f:
    print(f['exchange']['data'].shape)
    proj = f['exchange']['data'][nproj*frame:nproj*(frame+1),
                                 z_start:z_stop,y_start:y_stop]
    print(proj.shape)
    flat = f['exchange']['data_white'][:, z_start:z_stop,y_start:y_stop]
    dark = f['exchange']['data_dark'][:, z_start:z_stop,y_start:y_stop]

print("dark field correction")
proj = proj.astype(np.float64) - dark.mean(axis=0)
flat = flat.astype(np.float64) - dark.mean(axis=0)

# fix dead pixels during flat fields
flat[:, 347, 347] = 2/3*flat[:, 346, 347] + 1/3*flat[:, 349, 347]
flat[:, 348, 347] = 1/3*flat[:, 346, 347] + 2/3*flat[:, 349, 347]

plt.figure()
plt.imshow(flat.mean(axis=0))

# plt.figure()
# plt.imshow(flat.mean(axis=1))

# plt.figure()
# plt.imshow(flat.mean(axis=2))

# plt.figure()
# plt.imshow(flat[flat.shape[0]//2, :, :])

# plt.figure()
# plt.imshow(flat[:, flat.shape[1]//2, :])

# plt.figure()
# plt.imshow(flat[:, :, flat.shape[2]//2])

plt.show()

data1 = proj/flat.mean(axis=0)
mean = data1.sum(axis=(1,2)).mean()

data = np.zeros(proj.shape)

from tqdm import tqdm

mean_flat = np.zeros(flat.shape)
for j in range(flat.shape[0] - 5):
    mean_flat[j, :, :] = flat[j:j+5].mean(axis=0)

for i in tqdm(range(proj.shape[0])):
    best = np.inf
    for j in range(flat.shape[0] - 5):
        f = flat[j]
        if not (f == 0).any():
            normalized_p = proj[i]/f
            error = abs(normalized_p.sum() - mean)
            if error < best:
                best_p = normalized_p
                best = error
    data[i, :, :] = best_p

data = tomopy.remove_stripe_fw(data,level=7,wname='sym16',sigma=1,pad=True)

print("log transform")
data = tomopy.minus_log(data)
data = tomopy.remove_nan(data, val=0.0)
data = tomopy.remove_neg(data, val=0.0)
data[np.where(data == np.inf)] = 0.0
np.save("data/sino_without_paganin", data)

print("applying paganin filter")
data_paganin = tomopy.retrieve_phase(data, energy=16, dist=25, pixel_size=0.0003, alpha=0.00008)
np.save("data/sino_with_paganin", data_paganin)

print("downsampling")
data = tomopy.downsample(data, level=1, axis=2)
data = tomopy.downsample(data, level=1, axis=1)
np.save("data/sino_without_paganin", data)
data_paganin = tomopy.downsample(data_paganin, level=1, axis=2)
data_paganin = tomopy.downsample(data_paganin, level=1, axis=1)
np.save("data/sino_with_paganin", data_paganin)

print("""
+----------------+
| Reconstruction |
+----------------+
""")
# create astra optomo operator
angles = np.linspace(0, np.pi, nproj, endpoint=False)
vol_geom = astra.create_vol_geom(data.shape[2], data.shape[2], data.shape[1])
proj_geom = astra.create_proj_geom('parallel3d', 1, 1, data.shape[1], data.shape[2], angles)
proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
W = astra.OpTomo(proj_id)

# create gradient operator
rec_shape = (data.shape[1], data.shape[2], data.shape[2])
rec_size = data.shape[1] * data.shape[2] * data.shape[2]

p = data_paganin[::5, :, :].swapaxes(0,1).ravel()
proj_geom = astra.create_proj_geom('parallel3d', 1, 1, data.shape[1], data.shape[2], angles[::5])
proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
W = astra.OpTomo(proj_id)

#p = data_paganin.swapaxes(0,1).ravel()

print("reconstructing")
rec = lsqr(W, p, iter_lim=20, show=True)[0]
rec = rec.reshape((data.shape[1], data.shape[2], data.shape[2])).astype(np.float32)
np.save("data/initial_rec_with_background", rec)

print("done")

fig, ax = plt.subplots(1, 3)
ax[0].imshow(rec[rec.shape[0]//2, :, :], cmap="gray")
ax[1].imshow(rec[:, rec.shape[1]//2, :], cmap="gray")
ax[2].imshow(rec[:, :,rec.shape[2]//2], cmap="gray")

plt.figure()
plt.imshow(data[:,0,:])

plt.figure()
plt.imshow(data_paganin[:,0,:])

plt.show()
