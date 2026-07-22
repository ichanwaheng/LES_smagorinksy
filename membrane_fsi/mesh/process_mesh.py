#!/usr/bin/env python3
"""Convert Gmsh .msh with membrane patch into processed_mesh.npz (FV connectivity)."""

from __future__ import annotations

import argparse
from pathlib import Path

import meshio
import numpy as np


def tet_volume(points: np.ndarray, tet: np.ndarray) -> float:
    a, b, c, d = points[tet]
    return abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0


def extract_faces(tetra: np.ndarray):
    local = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    face_to_cells: dict[tuple[int, int, int], list[int]] = {}
    for ci, tet in enumerate(tetra):
        for i, j, k in local:
            key = tuple(sorted((int(tet[i]), int(tet[j]), int(tet[k]))))
            face_to_cells.setdefault(key, []).append(ci)
    return face_to_cells


def build_face_connectivity(face_to_cells: dict):
    unique_faces = []
    owner = []
    neighbour = []
    for face, cells in face_to_cells.items():
        unique_faces.append(face)
        owner.append(cells[0])
        neighbour.append(cells[1] if len(cells) > 1 else -1)
    return (
        np.asarray(unique_faces, dtype=np.int64),
        np.asarray(owner, dtype=np.int64),
        np.asarray(neighbour, dtype=np.int64),
    )


def face_geometry(points, unique_faces, owner, cell_centroids):
    p0 = points[unique_faces[:, 0]]
    p1 = points[unique_faces[:, 1]]
    p2 = points[unique_faces[:, 2]]
    face_centroids = (p0 + p1 + p2) / 3.0
    nraw = np.cross(p1 - p0, p2 - p0)
    areas = 0.5 * np.linalg.norm(nraw, axis=1)
    # Orient out of owner
    to_face = face_centroids - cell_centroids[owner]
    flip = np.einsum("ij,ij->i", nraw, to_face) < 0
    nraw[flip] *= -1
    norms = np.linalg.norm(nraw, axis=1) + 1e-30
    face_normals = nraw / norms[:, None]
    face_area_vectors = face_normals * areas[:, None]
    return face_centroids, face_area_vectors, areas, face_normals


def process(msh_path: str, out_npz: str) -> str:
    m = meshio.read(msh_path)
    # tet block is last
    tetra = None
    for block in m.cells:
        if block.type in ("tetra", "tetra10"):
            tetra = block.data[:, :4].astype(np.int64)
    if tetra is None:
        raise RuntimeError("No tetrahedra in mesh")
    points = m.points.astype(np.float64)

    cell_centroids = points[tetra].mean(axis=1)
    cell_volumes = np.array([tet_volume(points, t) for t in tetra], dtype=np.float64)

    face_to_cells = extract_faces(tetra)
    unique_faces, owner, neighbour = build_face_connectivity(face_to_cells)
    face_centroids, Sf, areas, normals = face_geometry(
        points, unique_faces, owner, cell_centroids
    )

    # Map gmsh physical tags → boundary_tag
    # Expected names: inlet=1, outlet=2, walls=3, membrane=4 (object-equivalent)
    name_to_id = {}
    for name, arr in m.field_data.items():
        # arr = [tag, dim]
        if int(arr[1]) == 2:
            lname = name.lower()
            if "inlet" in lname:
                name_to_id[int(arr[0])] = 1
            elif "outlet" in lname:
                name_to_id[int(arr[0])] = 2
            elif "wall" in lname:
                name_to_id[int(arr[0])] = 3
            elif "membrane" in lname or "object" in lname:
                name_to_id[int(arr[0])] = 4

    face_lookup = {tuple(f): i for i, f in enumerate(unique_faces)}
    boundary_tag = np.full(len(unique_faces), 5, dtype=np.int32)
    buckets = {1: [], 2: [], 3: [], 4: []}

    for block, tags in zip(m.cells, m.cell_data.get("gmsh:physical", [])):
        if block.type not in ("triangle", "triangle6"):
            continue
        tris = block.data[:, :3]
        for tri, tag in zip(tris, tags):
            key = tuple(sorted(map(int, tri)))
            if key not in face_lookup:
                continue
            fi = face_lookup[key]
            bid = name_to_id.get(int(tag), 3)
            boundary_tag[fi] = bid
            if bid in buckets:
                buckets[bid].append(fi)

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        points=points,
        tetra=tetra,
        cell_centroids=cell_centroids,
        cell_volumes=cell_volumes,
        unique_faces=unique_faces,
        owner=owner,
        neighbour=neighbour,
        face_centroids=face_centroids,
        face_area_vectors=Sf,
        face_areas=areas,
        face_normals=normals,
        boundary_tag=boundary_tag,
        inlet_faces=np.asarray(buckets[1], dtype=np.int32),
        outlet_faces=np.asarray(buckets[2], dtype=np.int32),
        wall_faces=np.asarray(buckets[3], dtype=np.int32),
        object_faces=np.asarray(buckets[4], dtype=np.int32),  # membrane
        membrane_faces=np.asarray(buckets[4], dtype=np.int32),
    )
    print(f"Saved {out_npz}")
    for k, v in [("inlet", 1), ("outlet", 2), ("walls", 3), ("membrane", 4)]:
        print(f"  {k}: {np.sum(boundary_tag == v)}")
    return out_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msh", default="membrane_fsi/data/fluid_mesh_membrane.msh")
    ap.add_argument("-o", "--out", default="membrane_fsi/data/processed_mesh_membrane.npz")
    args = ap.parse_args()
    process(args.msh, args.out)


if __name__ == "__main__":
    main()
