import numpy as np
import trimesh
from mesh_projector.utils.recalculate_normals import recalculate_normals

all_vertices = np.load("data/last_results/iterate55.npy")
all_faces = np.load("data/last_results/all_faces.npy")
all_normals = recalculate_normals(all_vertices, all_faces)
mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, face_normals=all_normals)
trimesh.exchange.export.export_mesh(mesh, "data/last_results/iterate55.stl")