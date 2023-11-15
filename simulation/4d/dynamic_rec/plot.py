import numpy as np
from matplotlib import pyplot as plt
import matplotlib.style
import matplotlib as mpl
mpl.style.use('ggplot')

vol_errors = np.load("data/vol_errors.npy")
tres_errors = np.load("data/tres_errors.npy")
mesh_errors = np.load("data/mesh_errors.npy")


plt.figure(figsize=(4, 3))
plt.plot(vol_errors, "--")
plt.plot(tres_errors, "-.")
plt.plot(mesh_errors)
plt.legend(["Volume", "Tresholded volume", "BubSub"])
# plt.ylim((0.06, 0.13))
plt.xlabel("Subscan")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig("images/error_plot.eps")
plt.show()
