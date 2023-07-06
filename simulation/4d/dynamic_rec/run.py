import os

for subscan in range(89, -1, -1):
    # os.system(f"python ordinary_reconstruction.py {subscan}")
    os.system(f"python reconstruct_mesh.py {subscan}")