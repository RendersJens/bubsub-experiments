import numpy as np
import trimesh
from mesh_tomography.projection3D.project_mesh_gpu64 import project_mesh

# the phantom is interpreted to have a width of 2cm
# the background represents water-like liquid, with attenuation 1/cm
# this is approximately the attenuation of water at 16keV
# the bubbles contain air-like gas, with 0 attenuation.

def simulate_scan(n_proj=100, I0=1e5):
    foam = trimesh.load("data/bucket3D2000.stl")
    container = trimesh.load("data/container.stl")

    foam_vertices = np.asarray(foam.vertices, dtype=np.float64)*180 # times 180 to scale the mesh up
    foam_faces = np.asarray(foam.faces, dtype=np.int64)
    foam_normals = np.asarray(foam.face_normals, dtype=np.float64).copy()

    container_vertices = np.asarray(container.vertices, dtype=np.float64)*180 # times 180 to scale the mesh up
    container_faces = np.asarray(container.faces, dtype=np.int64)
    container_normals = np.asarray(container.face_normals, dtype=np.float64).copy()

    angles = np.linspace(0, 2*np.pi, n_proj, dtype=np.float64, endpoint=False)

    # run the projector
    # bubbles (att 1)
    foam_sino = project_mesh(foam_vertices, foam_faces, foam_normals,
                        1, # det pixel width
                        1, # det pixel height
                        256, # det row count
                        512, # det col count
                        angles,
                        5000, # source origin distance
                        200).get()  # origin detector distance
    foam_sino /= 180 # compensate previous upscaling

    # container (att 1)
    container_sino = project_mesh(container_vertices, container_faces, container_normals,
                        1, # det pixel width
                        1, # det pixel height
                        256, # det row count
                        512, # det col count
                        angles,
                        5000, # source origin distance
                        200).get()  # origin detector distance
    container_sino /= 180 # compensate previous upscaling

    # container (att 1) with bubbles (att 0)
    sino = container_sino - foam_sino

    # convert to intensity domain to add realistic poisson noise
    intensities = I0*np.exp(-sino)
    noisy_intensities = np.random.poisson(intensities)
    noisy_intensities[noisy_intensities==0] = 1

    # back to attenuation domain
    sino = -np.log(noisy_intensities/I0)

    # back to bubbles with att 1, equivalent to background removal step on real data.
    foam_sino = container_sino - sino

    return foam_sino

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    sino = simulate_scan()
    np.save("data/sino", sino)

    plt.figure()
    plt.imshow(sino[sino.shape[0]//2, :, :])

    plt.figure()
    plt.imshow(sino[:, sino.shape[1]//2, :])

    plt.figure()
    plt.imshow(sino[:, :, sino.shape[2]//2])

    plt.show()