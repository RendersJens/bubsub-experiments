import numpy as np
import trimesh
from mesh_tomography.utils.remesh import remesh


def clean_bubble_mesh(vertices, faces, resolution=0.5):
    bubbles = trimesh.Trimesh(vertices=vertices, faces=faces).split(only_watertight=True)
    index = 0
    clean_vertices = []
    clean_faces = []
    print(f"{len(bubbles)} bubbles")
    for i, bubble in enumerate(bubbles):

        # shrink bubble
        center = bubble.vertices.mean(axis=0)
        bubble.vertices -= center
        bubble.vertices *= 0.8
        bubble.vertices += center

        # estimate radius
        dx = bubble.vertices[:, 0].max() - bubble.vertices[:, 0].min()
        dy = bubble.vertices[:, 1].max() - bubble.vertices[:, 1].min()
        dz = bubble.vertices[:, 2].max() - bubble.vertices[:, 2].min()
        diameter = (dx + dy + dz)/3
        radius = diameter/2

        # remesh relative to radius
        rel_resolution = radius * resolution
        try:
            bubble_vertices, bubble_faces = remesh(
            bubble.vertices, bubble.faces, rel_resolution)
        except:
            bubble_vertices, bubble_faces = remesh(
            bubble.vertices, bubble.faces, rel_resolution/2)

        clean_vertices.append(bubble_vertices)
        clean_faces.append(bubble_faces+index)
        index += len(bubble_vertices)
    clean_vertices = np.vstack(clean_vertices)
    clean_faces = np.vstack(clean_faces)
    return clean_vertices, clean_faces