import numpy as np
from scipy.sparse.linalg import lsqr
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max, corner_peaks
from scipy.ndimage import distance_transform_edt, label
from skimage.segmentation import watershed
from skimage.measure import marching_cubes
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import trimesh

n_angles = 300

print("loading initial rec")
rec = -np.load(f"data/vol_rec_{n_angles}.npy")
centers = np.load("data/centers.npy")

print("Otsu thresholding")
tres = threshold_otsu(rec)

print("binarizing")
binary = rec > tres

print("distance transform")
dt = distance_transform_edt(binary)
markers = np.zeros(dt.shape, dtype=np.int64)
markers[tuple(centers.T)] = np.arange(len(centers)) + 1

print("watershed")
labeled_rec = watershed(-dt, markers, mask=binary, compactness=0)

for i in tqdm(range(len(centers))):
    try:
        vertices, faces, normals, values = marching_cubes(
            labeled_rec == i + 1,
            level=0.5,
            step_size=1
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_normals=normals)
        trimesh.exchange.export.export_mesh(mesh, f"data/mc_{n_angles}/mc_{n_angles}_{i}.stl")
    except RuntimeError:
        print(f"skipped bubble {i}")
    except ValueError:
        print(f"skipped bubble {i}")

colors = np.random.rand(256,3)
colors[0] = 0
cmap = ListedColormap(colors)

plt.figure()
plt.imshow(labeled_rec[rec.shape[0]//2, :, :], cmap=cmap)

plt.figure()
plt.imshow(labeled_rec[:, rec.shape[1]//2, :], cmap=cmap)

plt.figure()
plt.imshow(labeled_rec[:, :, rec.shape[2]//2], cmap=cmap)

plt.show()
