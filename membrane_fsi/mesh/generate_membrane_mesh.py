#!/usr/bin/env python3
"""
Generate a 3D unstructured fluid mesh: rectangular box minus a thin rectangular plate
(flexible membrane geometry). Physical groups: inlet, outlet, walls, membrane.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gmsh
import numpy as np


def generate_membrane_mesh(
    outfile: str = "fluid_mesh_membrane.msh",
    L: float = 10.0,
    W: float = 5.0,
    H: float = 5.0,
    # Plate (membrane) geometry: thin box cut from the fluid
    plate_x: float = 3.0,  # mid-plane x of plate
    plate_t: float = 0.08,  # thickness (streamwise)
    plate_w: float = 2.0,  # width in y
    plate_h: float = 2.0,  # height in z
    plate_yc: float = 2.5,
    plate_zc: float = 2.5,
    size_min: float = 0.08,
    size_max: float = 0.45,
    nopopup: bool = True,
) -> str:
    gmsh.initialize()
    gmsh.model.add("membrane_channel_3d")

    # Outer box
    box = gmsh.model.occ.addBox(0, 0, 0, L, W, H)

    # Thin rectangular plate (solid obstacle to subtract)
    x0 = plate_x - 0.5 * plate_t
    y0 = plate_yc - 0.5 * plate_w
    z0 = plate_zc - 0.5 * plate_h
    plate = gmsh.model.occ.addBox(x0, y0, z0, plate_t, plate_w, plate_h)

    fluid_v, _ = gmsh.model.occ.cut([(3, box)], [(3, plate)])
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getBoundary(fluid_v, oriented=False)
    inlet, outlet, walls, membrane = [], [], [], []

    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        xmin, ymin, zmin = com
        # classify by centre of mass
        if abs(xmin - 0.0) < 1e-3:
            inlet.append(tag)
        elif abs(xmin - L) < 1e-3:
            outlet.append(tag)
        elif (
            (plate_x - plate_t - 0.2 < xmin < plate_x + plate_t + 0.2)
            and (y0 - 0.2 < ymin < y0 + plate_w + 0.2)
            and (z0 - 0.2 < zmin < z0 + plate_h + 0.2)
        ):
            membrane.append(tag)
        else:
            walls.append(tag)

    gmsh.model.addPhysicalGroup(2, inlet, name="inlet")
    gmsh.model.addPhysicalGroup(2, outlet, name="outlet")
    gmsh.model.addPhysicalGroup(2, walls, name="walls")
    gmsh.model.addPhysicalGroup(2, membrane, name="membrane")  # tag will be 4th group → id 4

    fluid_tags = [tag for dim, tag in fluid_v]
    gmsh.model.addPhysicalGroup(3, fluid_tags, name="internalfield")

    # Refinement near membrane
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList", membrane)
    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", size_min)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", size_max)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.15)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", 2.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)

    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.model.mesh.generate(3)

    out = Path(outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(out))

    # Save geometry sidecar for the solid model
    meta = out.with_suffix(".npz")
    np.savez(
        meta,
        L=L,
        W=W,
        H=H,
        plate_x=plate_x,
        plate_t=plate_t,
        plate_w=plate_w,
        plate_h=plate_h,
        plate_yc=plate_yc,
        plate_zc=plate_zc,
        x0=x0,
        y0=y0,
        z0=z0,
    )
    print(f"Wrote mesh → {out}")
    print(f"Wrote geometry meta → {meta}")
    print(f"Surfaces: inlet={len(inlet)} outlet={len(outlet)} walls={len(walls)} membrane={len(membrane)}")

    if not nopopup:
        gmsh.fltk.run()
    gmsh.finalize()
    return str(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate fluid mesh with rectangular membrane")
    p.add_argument("-o", "--outfile", default="membrane_fsi/data/fluid_mesh_membrane.msh")
    p.add_argument("--coarse", action="store_true", help="Faster/coarser mesh for demos")
    p.add_argument("--nopopup", action="store_true", default=True)
    args = p.parse_args()
    kwargs = {}
    if args.coarse:
        kwargs.update(dict(size_min=0.12, size_max=0.6))
    generate_membrane_mesh(outfile=args.outfile, nopopup=args.nopopup, **kwargs)


if __name__ == "__main__":
    main()
