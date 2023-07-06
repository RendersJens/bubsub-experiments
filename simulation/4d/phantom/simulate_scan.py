import numpy as np
import astra
import foam_ct_phantom
import h5py
from matplotlib import pyplot as plt
import dxchange
from tqdm import tqdm
import sys

subprocess = int(sys.argv[1])

# settings
flux = 1e4
phantom_width_cm = 3
phantom_width_vox = 500
phantom_height_vox = 620
voxel_size = phantom_width_cm/phantom_width_vox
n_angles = 200
n_subscans = 100

# cointainer
X, Y, Z = np.meshgrid(
    range(phantom_height_vox),
    range(phantom_width_vox),
    range(phantom_width_vox),
    indexing="ij"
)
center = phantom_width_vox/2
radius = phantom_width_vox/2 * 0.7
container = (Y - center)**2 + (Z - center)**2 <= radius**2

# phantom
phantom = foam_ct_phantom.ExpandingFoamPhantom('bubble_configuration_4d.h5')
geom = foam_ct_phantom.VolumeGeometry(
    phantom_width_vox,
    phantom_width_vox,
    phantom_height_vox,
    3/phantom_width_vox # this 3 is not phantom_width_cm (I think)
)

phantom.generate_volume(f'phantom_3d_{subprocess}.h5', geom, time=1)
vol = container - (1 - foam_ct_phantom.load_volume(f'phantom_3d_{subprocess}.h5'))

# geometry
vol_geom = astra.create_vol_geom(
    phantom_width_vox,
    phantom_width_vox,
    phantom_height_vox,
)
vol_id = astra.data3d.create('-vol', vol_geom, vol)
container_vol_id = astra.data3d.create('-vol', vol_geom, container)
angles = np.linspace(0, 2*np.pi, n_angles)

part = np.split(np.arange(n_subscans), 5)[subprocess]
for i in tqdm(part):
    for j, angle in enumerate(tqdm(angles)):
        time = (i * n_angles + j)/(n_subscans*n_angles)
        print(time)

        # generate next phase of the phantom
        phantom.generate_volume(f'phantom_3d_{subprocess}.h5', geom, time=time)
        vol = container - (1 - foam_ct_phantom.load_volume(f'phantom_3d_{subprocess}.h5'))
        astra.data3d.store(vol_id, vol)

        # update geometry to next angle
        proj_geom = astra.create_proj_geom(
            'cone',
            1, 1,
            phantom_height_vox,
            phantom_width_vox,
            np.array([angle]),
            phantom_width_vox*10,
            phantom_width_vox/2
        )
        sino_id = astra.data3d.create('-sino', proj_geom)

        # projection of bubbles
        proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
        alg_id = astra.algorithm.create({
            "type" : "FP3D_CUDA",
            "ProjectorId" : proj_id,
            "ProjectionDataId" : sino_id,
            "VolumeDataId" : vol_id,
        })
        astra.algorithm.run(alg_id)
        astra.astra.delete([alg_id])
        sino = astra.data3d.get(sino_id)
        astra.data3d.store(sino_id, 0)
        sino = sino[:, 0, :]*voxel_size

        # add noise
        # sino = flux*np.exp(-sino)
        # sino = np.random.poisson(sino)
        # sino = -np.log(sino/flux)

        # also project the contrainer to subtract
        astra.data3d.store(sino_id, 0)
        alg_id = astra.algorithm.create({
            "type" : "FP3D_CUDA",
            "ProjectorId" : proj_id,
            "ProjectionDataId" : sino_id,
            "VolumeDataId" : container_vol_id,
        })
        astra.algorithm.run(alg_id)
        sino_container = astra.data3d.get(sino_id)
        sino_container = sino_container[:, 0, :]*voxel_size
        astra.astra.delete([alg_id, sino_id, proj_id])
        dxchange.write_tiff(sino.astype(np.float32), f"scan/subscan_{i}/proj_{j:0>5d}", overwrite=True)
        dxchange.write_tiff(sino_container.astype(np.float32), f"scan_container/subscan_{i}/proj_{j:0>5d}", overwrite=True)

# plt.figure()
# plt.imshow(sino_container, cmap="gray")
# plt.colorbar()

# plt.figure()
# plt.imshow(sino_no_bg, cmap="gray")
# plt.colorbar()

plt.figure()
plt.imshow(sino, cmap="gray")
plt.colorbar()

plt.show()
