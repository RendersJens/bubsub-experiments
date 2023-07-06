import sys
import numpy as np
import cupy as cp
import astra
import trimesh
import pylops
import dxchange
from matplotlib import pyplot as plt
from tqdm import tqdm
from mesh_tomography.projection3D.project_mesh_gpu64 import project_mesh
from mesh_tomography.projection3D.project_mesh_gpu64 import grad_project_mesh
from mesh_tomography.reconstruction import quasi_newton, BB
from mesh_tomography.utils.recalculate_normals import recalculate_normals
from mesh_tomography.utils.subdivision import loop_subdivide
from mesh_tomography.utils.remesh import remesh
from cleaning import clean_bubble_mesh
from skimage.filters import threshold_otsu

flux = 1e4
angle_skip = 4
attenuation = 1
phantom_width_cm = 3
phantom_width_vox = 500
phantom_height_vox = 620
voxel_size = phantom_width_cm/phantom_width_vox

int_type = np.int64
float_type = np.float64

mesh_errors = []
vol_errors = []
tres_errors = []
for subscan in range(27, 90):
    print("loading sinogram")
    sino_complete = dxchange.read_tiff_stack(f"../phantom/scan/subscan_{subscan}/proj_00000.tiff", range(100))
    sino_container = dxchange.read_tiff_stack(f"../phantom/scan_container/subscan_{subscan}/proj_00000.tiff", range(100))

    sino_complete = flux*np.exp(-sino_complete)
    np.random.seed(12345)
    sino_complete = np.random.poisson(sino_complete)
    sino_complete = -np.log(sino_complete/flux)

    sino = sino_container - sino_complete
    sino /= voxel_size
    sino = cp.asarray(sino).astype(float_type)
    del sino_complete
    del sino_container
    print("done")

    angles = np.linspace(0, 2*np.pi, sino.shape[0])

    # these angles were not used during reconstruction
    sino = sino[2::angle_skip].copy()
    angles = angles[2::angle_skip].copy()
    geom_settings = (
        1, 1,
        phantom_height_vox,
        phantom_width_vox,
        angles,
        phantom_width_vox*10,
        phantom_width_vox//2
    )


    # create astra optomo operator
    vol_geom = astra.create_vol_geom(sino.shape[2], sino.shape[2], sino.shape[1])
    proj_geom = astra.create_proj_geom('cone', *geom_settings)
    proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
    W = astra.optomo.OpTomo(proj_id)
    ordinary_rec = np.load(f"data/ordinary_reconstruction_{subscan}.npy")
    print("Otsu thresholding")
    tres = threshold_otsu(ordinary_rec)
    print("binarizing")
    binary = ordinary_rec > tres
    p_ord = (W @ ordinary_rec.ravel()).reshape(sino.swapaxes(0, 1).shape)
    p_tres = (W @ binary.ravel()).reshape(sino.swapaxes(0, 1).shape)
    p_ord = p_ord.swapaxes(0, 1)
    p_tres = p_tres.swapaxes(0, 1)
    proj_diff = p_ord - sino.get()
    vol_error = np.dot(proj_diff.ravel(), proj_diff.ravel())/proj_diff.size
    vol_errors.append(vol_error)
    print(f"vol error in subscan {subscan}: {vol_error}")

    proj_diff = p_tres - sino.get()
    tres_error = np.dot(proj_diff.ravel(), proj_diff.ravel())/proj_diff.size
    tres_errors.append(tres_error)
    print(f"tres error in subscan {subscan}: {tres_error}")

    # geom settings for mesh projector
    geom_settings = (
        1, 1,
        phantom_height_vox,
        phantom_width_vox,
        angles + np.pi/2,
        phantom_width_vox*10,
        phantom_width_vox//2
    )

    control_vertices = np.load(f"data/control_vertices_{subscan}.npy")
    control_faces = np.load(f"data/control_faces_{subscan}.npy")
    all_vertices, all_faces = loop_subdivide(control_vertices, control_faces)
    all_normals = recalculate_normals(all_vertices, all_faces)

    proj_diff = project_mesh(all_vertices, all_faces, all_normals, *geom_settings)
    proj_diff *= attenuation
    proj_diff -= sino
    mesh_error = cp.dot(proj_diff.ravel(), proj_diff.ravel()).get()/proj_diff.size
    mesh_errors.append(mesh_error)
    print(f"mesh error in subscan {subscan}: {mesh_error}")

np.save("data/vol_errors", vol_errors)
np.save("data/tres_errors", tres_errors)
np.save("data/mesh_errors", mesh_errors)
