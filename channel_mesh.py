#!/usr/bin/env python
# coding: utf-8
"""Generate a plain channel (box) tetrahedral mesh with local refinement and
build the finite-volume connectivity/geometry used by ``les_solver``.

Unlike ``processed_mesh.npz`` (channel with a sphere), this produces an empty
channel so that a membrane can be immersed in the flow as a thin baffle for the
fluid-structure-interaction (FSI) coupling.

    python channel_mesh.py            # writes channel_mesh.npz
"""
import numpy as np


def generate_channel_gmsh(L=10.0, W=5.0, H=5.0,
                          refine_box=(3.5, 6.5, 0.5, 4.5, 0.5, 4.5),
                          size_fine=0.18, size_coarse=0.7,
                          out_msh="channel.msh"):
    import gmsh
    gmsh.initialize()
    gmsh.model.add("channel")
    gmsh.model.occ.addBox(0, 0, 0, L, W, H)
    gmsh.model.occ.synchronize()

    x0, x1, y0, y1, z0, z1 = refine_box
    f = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f, "VIn", size_fine)
    gmsh.model.mesh.field.setNumber(f, "VOut", size_coarse)
    gmsh.model.mesh.field.setNumber(f, "XMin", x0)
    gmsh.model.mesh.field.setNumber(f, "XMax", x1)
    gmsh.model.mesh.field.setNumber(f, "YMin", y0)
    gmsh.model.mesh.field.setNumber(f, "YMax", y1)
    gmsh.model.mesh.field.setNumber(f, "ZMin", z0)
    gmsh.model.mesh.field.setNumber(f, "ZMax", z1)
    gmsh.model.mesh.field.setNumber(f, "Thickness", 1.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)

    gmsh.model.mesh.generate(3)
    gmsh.write(out_msh)
    gmsh.finalize()
    print(f"Wrote {out_msh}")


def build_fv(points, tetra):
    """Build finite-volume face connectivity/geometry from a tet mesh."""
    n_cells = len(tetra)
    cell_centroids = points[tetra].mean(axis=1)
    # signed volume of each tet
    a, b, c, d = (points[tetra[:, 0]], points[tetra[:, 1]],
                  points[tetra[:, 2]], points[tetra[:, 3]])
    cell_volumes = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0

    # local faces of a tet (opposite each vertex)
    lf = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    face_nodes = tetra[:, lf].reshape(-1, 3)                     # (4*nt, 3)
    owner_of_face = np.repeat(np.arange(n_cells), 4)
    key = np.sort(face_nodes, axis=1)

    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    key_s = key[order]
    fn_s = face_nodes[order]
    own_s = owner_of_face[order]

    same = np.all(key_s[1:] == key_s[:-1], axis=1)              # same[j]: row j+1 == row j
    # first occurrence indices of each unique face
    is_first = np.ones(len(key_s), dtype=bool)
    is_first[1:] = ~same
    uf_idx = np.where(is_first)[0]

    unique_faces = fn_s[uf_idx]
    owner = own_s[uf_idx]
    neighbour = np.full(len(uf_idx), -1, dtype=np.int64)
    # row j+1 duplicates row j (the first occurrence) -> internal face
    dup = np.where(same)[0]             # first-occurrence rows that have a partner
    uf_pos_for_row = np.cumsum(is_first) - 1
    neighbour[uf_pos_for_row[dup]] = own_s[dup + 1]

    p0, p1, p2 = (points[unique_faces[:, 0]], points[unique_faces[:, 1]],
                  points[unique_faces[:, 2]])
    face_centroids = (p0 + p1 + p2) / 3.0
    Sf = 0.5 * np.cross(p1 - p0, p2 - p0)                        # area vector
    face_areas = np.linalg.norm(Sf, axis=1)
    # orient outward from owner
    outward = face_centroids - cell_centroids[owner]
    flip = np.einsum("ij,ij->i", outward, Sf) < 0
    Sf[flip] *= -1.0
    face_normals = Sf / face_areas[:, None]

    return {
        "cell_centroids": cell_centroids,
        "cell_volumes": cell_volumes,
        "owner": owner.astype(np.int64),
        "neighbour": neighbour,
        "face_centroids": face_centroids,
        "face_area_vectors": Sf,
        "face_areas": face_areas,
        "face_normals": face_normals,
    }


def generate(out_npz="channel_mesh.npz", **kw):
    import meshio
    generate_channel_gmsh(out_msh="channel.msh", **kw)
    m = meshio.read("channel.msh")
    tetra = None
    for cb in m.cells:
        if cb.type == "tetra":
            tetra = cb.data.astype(np.int64)
    points = m.points
    fv = build_fv(points, tetra)
    np.savez(out_npz, points=points, tetra=tetra, **fv)
    print(f"Saved {out_npz}: {len(tetra)} cells, {len(fv['owner'])} faces "
          f"(internal {np.sum(fv['neighbour']!=-1)}, boundary {np.sum(fv['neighbour']==-1)})")
    return out_npz


if __name__ == "__main__":
    generate()
