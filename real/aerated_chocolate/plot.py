import numpy as np
from matplotlib import pyplot as plt
import matplotlib.style
import matplotlib as mpl
mpl.style.use('ggplot')

errors_vol = np.load("data/mean_squared_projection_errors_vol.npy")
errors_vol_nonneg = np.load("data/mean_squared_projection_errors_vol_nonneg.npy")
errors_vol_box = np.load("data/mean_squared_projection_errors_vol_box.npy")
errors_vol_tres = np.load("data/mean_squared_projection_errors_vol_tres.npy")
errors_mesh = np.load("data/mean_squared_projection_errors_mesh.npy")
n_angles = list(range(2, 11, 2)) + list(range(15, 51, 5)) + list(range(60, 201, 10))

plt.figure(figsize=(4, 3))
plt.plot(n_angles, errors_vol_tres, "--")
plt.plot(n_angles, errors_vol_nonneg, "-.")
# plt.plot(n_angles, errors_vol_box)
plt.plot(n_angles, errors_mesh)
# plt.xlim([0, 50])
plt.ylim([0.0014, 0.01])
# plt.yscale("log")
plt.xlabel("Number of projections")
plt.ylabel("MSE in projection space")
plt.legend(["Thresholded volume", "Volume", "BubSub"])
plt.tight_layout()
plt.savefig("images/luflee_error.eps")

plt.figure(figsize=(4, 3))
plt.plot(n_angles, errors_vol_tres, "--")
plt.plot(n_angles, errors_vol_nonneg, "-.")
# plt.plot(n_angles, errors_vol_box)
plt.plot(n_angles, errors_mesh)
plt.xlim([0, 50])
plt.ylim([0.0014, 0.006])
# plt.yscale("log")
plt.xlabel("Number of projections")
plt.ylabel("MSE in projection space")
plt.legend(["Thresholded volume", "Volume", "BubSub"])
plt.tight_layout()
plt.savefig("images/luflee_error_zoom.eps")

plt.show()