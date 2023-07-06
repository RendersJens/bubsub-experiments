import numpy as np
import astra
import dxchange
import pylops
from matplotlib import pyplot as plt
from imsolve.linear import BBLS
from skimage.filters import threshold_otsu
from tqdm import tqdm

int_type = np.int64
float_type = np.float64

attenuation = -0.0034
total_angles = 900

path = "/media/jens/Samsung_T5/journal-paper-mesh-reconstructions/real/luflee/"

sino_full = np.load(path + "data/sino.npy")[:, :, 500:-500]
sino_val = sino_full[1::2]
angles = -np.linspace(0, np.pi, total_angles, endpoint=False)
angles = angles[1::2]
geom_settings = (
    1, 1,
    sino_val.shape[1],
    sino_val.shape[2], 
    angles,
    50000,
    1
)
vol_geom = astra.create_vol_geom(sino_val.shape[2], sino_val.shape[2], sino_val.shape[1])
proj_geom_val = astra.create_proj_geom('cone', *geom_settings)
proj_id_val = astra.create_projector('cuda3d', proj_geom_val, vol_geom)
W_val = astra.optomo.OpTomo(proj_id_val)
p_val = sino_val.swapaxes(0, 1).ravel()

errors = []
tres_errors = []
for n_angles in list(range(2, 11, 2)) + list(range(15, 51, 5)) + list(range(60, 201, 10)):
    angle_inds = np.round(np.linspace(0, total_angles//2, n_angles, endpoint=False)).astype(np.int64)
    sino = sino_val[angle_inds]
    sino = np.asarray(sino, dtype=float_type)

    # create astra optomo operator
    angles = -np.linspace(0, np.pi, total_angles, endpoint=False)[::2]
    angles = angles[angle_inds]
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
    W = pylops.VStack([astra.optomo.OpTomo(proj_id)])
    p = sino.swapaxes(0,1).ravel()

    # volumetric reconstruction
    rec = BBLS(W, p, bounds=(-np.inf, 0), max_iter=30, verbose=True)
    rec = rec.reshape((sino.shape[1], sino.shape[2], sino.shape[2]))
    np.save(f"data/ordinary_reconstruction_{n_angles}", rec)

    print("Otsu thresholding")
    tres = threshold_otsu(rec)
    print("binarizing")
    binary = (rec < tres)*attenuation

    proj_diff = W_val @ rec.ravel() - p_val
    error = np.dot(proj_diff, proj_diff)/proj_diff.size
    print(error)
    errors.append(error)

    proj_diff = W_val @ binary.ravel() - p_val
    tres_error = np.dot(proj_diff, proj_diff)/proj_diff.size
    print(tres_error)
    tres_errors.append(tres_error)

np.save("data/mean_squared_projection_errors_vol_nonneg.npy", errors)
np.save("data/mean_squared_projection_errors_vol_tres.npy", tres_errors)