import os
import trimesh
import foam_ct_phantom
import numpy as np
from mesh_tomography.voxelization import voxelize
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
from tqdm import tqdm

phantom_width_cm = 3
phantom_width_vox = 500
phantom_height_vox = 620

phantom = foam_ct_phantom.ExpandingFoamPhantom('../phantom/bubble_configuration_4d.h5')
geom = foam_ct_phantom.VolumeGeometry(
    phantom_width_vox,
    phantom_width_vox,
    phantom_height_vox,
    3/phantom_width_vox # this 3 is not phantom_width_cm (I think)
)

for i in tqdm(np.round(np.linspace(27, 89, 4)).astype(int)):
    print("loading ordinary rec")
    ordinary = np.load(f"../dynamic_rec/data/ordinary_reconstruction_{i}.npy")
    shape = ordinary.shape
    ordinary_slice = ordinary[:, shape[1]//2, :]
    
    print("thresholding")
    tres = threshold_otsu(ordinary)
    tres_slice = ordinary_slice > tres

    print("loading mesh")
    mesh = trimesh.load(f"../dynamic_rec/data/rec_{i}.stl")
    mesh.vertices = -mesh.vertices[:, [2, 1, 0]]

    print("voxelizing mesh")
    mesh_vol = voxelize(mesh.split(), ordinary.shape)
    mesh_slice = np.fliplr(mesh_vol[:, shape[1]//2, :])

    print("loading phantom")
    phantom.generate_volume('phantom_3d.h5', geom, time=(i)/100)
    phantom_vol = 1 - foam_ct_phantom.load_volume('phantom_3d.h5')
    phantom_slice = phantom_vol[:, shape[1]//2, :]

    print("stacking frame")
    frame = np.hstack([ordinary_slice, tres_slice, mesh_slice, phantom_slice])
    frame = np.clip(frame, 0, 1)

    print("saving frame")
    plt.imsave(f"images/vol_{i:0>5d}.png", ordinary_slice, cmap="gray")
    plt.imsave(f"images/tres_{i:0>5d}.png", tres_slice, cmap="gray")
    plt.imsave(f"images/mesh_{i:0>5d}.png", mesh_slice, cmap="gray")
    plt.imsave(f"images/gt_{i:0>5d}.png", phantom_slice, cmap="gray")
    


