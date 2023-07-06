import numpy as np
import astra
import dxchange
from matplotlib import pyplot as plt
from imsolve.linear import BBLS
from tqdm import tqdm
import pylops

phantom_width_cm = 3
phantom_width_vox = 500
phantom_height_vox = 620
voxel_size = phantom_width_cm/phantom_width_vox

sino_complete = dxchange.read_tiff_stack("../phantom/scan/subscan_99/proj_00000.tiff", range(100))
sino_container = dxchange.read_tiff_stack("../phantom/scan_container/subscan_99/proj_00000.tiff", range(100))
sino = sino_container - sino_complete
sino /= voxel_size
del sino_complete
del sino_container

# create astra optomo operator
angles = np.linspace(0, 2*np.pi, sino.shape[0])
# sino = sino[::5]
# angles = angles[::5]
geom_settings = (
    1, 1,
    phantom_height_vox,
    phantom_width_vox,
    angles,
    phantom_width_vox*10,
    phantom_width_vox/2
)
vol_geom = astra.create_vol_geom(sino.shape[2], sino.shape[2], sino.shape[1])
proj_geom = astra.create_proj_geom('cone', *geom_settings)
proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
W = pylops.VStack([astra.optomo.OpTomo(proj_id)])
p = sino.swapaxes(0,1).ravel()

# create reconstruction
rec_shape = (sino.shape[1], sino.shape[2], sino.shape[2])
rec_size = sino.shape[1] * sino.shape[2] * sino.shape[2]

# ordinary reconstruction
rec = BBLS(W, p, bounds=(0, np.inf), max_iter=30, verbose=True)
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
