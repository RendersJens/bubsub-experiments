import numpy as np
import trimesh
from mesh_tomography.voxelization import voxelize
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening
from scipy.spatial.distance import dice
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

realisation = sys.argv[1]

vals = list(range(3, 103, 6))

gt_mesh = trimesh.load("data/bucket3D2000.stl")
gt_mesh.vertices *= 180
gt_vol = voxelize(trimesh.graph.split(gt_mesh), (512, 512, 256))
gt_vol = np.flip(np.flip(gt_vol.T.swapaxes(1,2), 2), 0)

errs = []
lsqr_errs = []
fdk_errs = []

for n_proj in tqdm(vals):
    lsqr_rec = np.load(f"data/n_proj_experiment_{realisation}/lsqr_rec{n_proj}.npy")
    fdk_rec = np.load(f"data/n_proj_experiment_{realisation}/fdk_rec{n_proj}.npy")
    print("Otsu thresholding")
    lsqr_tres = threshold_otsu(np.clip(lsqr_rec, 0, 1/180))
    fdk_tres = threshold_otsu(np.clip(fdk_rec, 0, 1/180))

    print("binarizing")
    lsqr_rec = lsqr_rec > lsqr_tres
    lsqr_rec = binary_opening(lsqr_rec)

    fdk_rec = fdk_rec > fdk_tres
    fdk_rec = binary_opening(fdk_rec)

    rec_mesh = trimesh.load(f"data/n_proj_experiment_{realisation}/rec{n_proj}.stl")
    rec = voxelize(trimesh.graph.split(rec_mesh), (512, 512, 256))
    rec = np.flip(rec.T, 0)

    phantom = gt_vol
    rnmp = np.logical_xor(rec, phantom).sum()/phantom.sum()
    mesh_dice = dice(rec.ravel(), phantom.ravel())
    errs.append(mesh_dice)

    lsqr_rnmp = np.logical_xor(lsqr_rec, phantom).sum()/phantom.sum()
    lsqr_dice = dice(lsqr_rec.ravel(), phantom.ravel())
    lsqr_errs.append(lsqr_dice)

    fdk_rnmp = np.logical_xor(fdk_rec, phantom).sum()/phantom.sum()
    fdk_dice = dice(fdk_rec.ravel(), phantom.ravel())
    fdk_errs.append(fdk_dice)

    print(f"bubble fit error at {n_proj} proj:", mesh_dice)
    print(f"lsqr error at {n_proj} proj:", lsqr_dice)
    print(f"fbp error at {n_proj} proj:", fdk_dice)
    # if True: #n_proj == 100:
    #     # plt.figure()
    #     # plt.imshow(np.logical_xor(rec, gt_vol)[rec.shape[0]//4-20, :, :])
    #     plt.figure()
    #     plt.imshow(lsqr_rec[rec.shape[0]//2, :, :])
    #     plt.figure()
    #     plt.imshow(fdk_rec[rec.shape[0]//2, :, :])

np.save(f"data/n_proj_experiment_{realisation}/errs", np.array(errs))
np.save(f"data/n_proj_experiment_{realisation}/lsqr_errs", np.array(lsqr_errs))
np.save(f"data/n_proj_experiment_{realisation}/fdk_errs", np.array(fdk_errs))

plt.figure()
plt.plot(vals, errs)
plt.plot(vals, lsqr_errs)
plt.plot(vals, fdk_errs)
#plt.yscale("log")
plt.show()
