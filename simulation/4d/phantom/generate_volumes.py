import foam_ct_phantom
import h5py
from matplotlib import pyplot as plt
import dxchange

foam_ct_phantom.ExpandingFoamPhantom.generate(
    'bubble_configuration_4d.h5',
    'bubble_configuration.h5',
    12345
)
phantom = foam_ct_phantom.ExpandingFoamPhantom('bubble_configuration_4d.h5')
geom = foam_ct_phantom.VolumeGeometry(512, 512, 700, 3/512)
phantom.generate_volume('phantom_3d_t1.h5', geom, time=0.9)
vol = 1 - foam_ct_phantom.load_volume('phantom_3d_t1.h5')
dxchange.write_tiff_stack(vol, "drishti/slice.tiff")

plt.figure()
plt.imshow(vol[vol.shape[0]//2, :, :], cmap="gray")

plt.figure()
plt.imshow(vol[:, vol.shape[1]//2, :], cmap="gray")

plt.figure()
plt.imshow(vol[:, :, vol.shape[2]//2], cmap="gray")

plt.show()