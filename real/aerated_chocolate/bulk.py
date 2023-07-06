import os

for i in range(2, 11, 2):
    os.system(f"python reconstruct_mesh.py {i}")

for i in range(15, 51, 5):
    os.system(f"python reconstruct_mesh.py {i}")

for i in range(60, 201, 10):
    os.system(f"python reconstruct_mesh.py {i}")