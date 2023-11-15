import numpy as np
import astra
import matplotlib.pyplot as plt

# the phantom is interpreted to have a width of 1cm
# the background represents water-like liquid, with attenuation 1/cm
# this is approximately the attenuation of water at 16keV
# the bubbles contain air-like gas, with 0 attenuation.

# in the image, backround is 0 and bubbles are 1, so we take 1 - phantom
# to get the correct attenuation coefficients.
# after the simulation (including realistic poisson noise) we undo this
# to get a sinogram of the original image (bg 0 and bubbles 1) but with noise
# corresponding to an attenuating background and non-attenuating bubbles
# this is equivalent to the background removal step in the
# preprocessing of the real datasets.


def simulate_scan(n_proj=30, I0=1e4):
    phantom = plt.imread("data/foam.png")[:, :, 0]

    angles = np.linspace(0, np.pi, n_proj, endpoint=False)
    geom_settings = (1, 2400, angles, 500000, 1)

    # create astra optomo operator
    vol_geom = astra.create_vol_geom(2000, 2000)
    proj_geom = astra.create_proj_geom('fanflat', *geom_settings)
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    W = astra.optomo.OpTomo(proj_id)
    W = (1/2000) * W
    p = W @ (1 - phantom.ravel())
    bg = W @ np.ones_like(phantom.ravel())

    sino = p.reshape(len(angles), -1)
    bg = bg.reshape(len(angles), -1)
    intensities = I0*np.exp(-sino)
    noisy_intensities = np.random.poisson(intensities)
    sino = -np.log(noisy_intensities/I0)
    sino = bg - sino

    noiseless = W @ phantom.ravel()
    print(np.std(noiseless - sino.ravel())/noiseless.max())

    # plt.figure()
    # plt.imshow(sino, cmap="gray")
    # plt.colorbar()
    # plt.axis("auto")
    # plt.show()
    # np.save("data/sino", sino)

    return sino

if __name__ == "__main__":
    simulate_scan(500, 1e5)
