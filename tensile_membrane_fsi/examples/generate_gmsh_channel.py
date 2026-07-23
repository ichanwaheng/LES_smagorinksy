#!/usr/bin/env python3
"""Optional Gmsh mesh for a fluid channel with an embedded membrane surface.

Produces ``membrane_channel.msh`` compatible with meshio / OpenFOAM-style
workflows. The Cartesian FSI solver in this package does not require Gmsh;
use this when you want an unstructured mesh for external CFD.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def generate(path: Path | None = None):
    try:
        import gmsh
    except ImportError as exc:
        raise SystemExit("gmsh Python API required: pip install gmsh") from exc

    path = path or (ROOT / "output" / "membrane_channel.msh")
    path.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    gmsh.model.add("membrane_channel")

    L, W, H = 8.0, 3.0, 2.5
    mx0, my0, mz0 = 2.0, 0.75, 1.0
    mL, mW = 2.0, 1.5

    box = gmsh.model.occ.addBox(0, 0, 0, L, W, H)
    # thin membrane slab as CAD surface proxy (cut a sheet volume then keep interface)
    sheet = gmsh.model.occ.addBox(mx0, my0, mz0 - 0.01, mL, mW, 0.02)
    fluid, _ = gmsh.model.occ.cut([(3, box)], [(3, sheet)])
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getBoundary(fluid, oriented=False)
    inlet, outlet, walls, membrane = [], [], [], []
    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(com[0] - 0.0) < 1e-3:
            inlet.append(tag)
        elif abs(com[0] - L) < 1e-3:
            outlet.append(tag)
        elif (mx0 - 0.05 < com[0] < mx0 + mL + 0.05) and (
            my0 - 0.05 < com[1] < my0 + mW + 0.05
        ) and abs(com[2] - mz0) < 0.05:
            membrane.append(tag)
        else:
            walls.append(tag)

    gmsh.model.addPhysicalGroup(2, inlet, name="inlet")
    gmsh.model.addPhysicalGroup(2, outlet, name="outlet")
    gmsh.model.addPhysicalGroup(2, walls, name="walls")
    gmsh.model.addPhysicalGroup(2, membrane, name="membrane")
    gmsh.model.addPhysicalGroup(3, [t for d, t in fluid], name="internalField")

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList", membrane or walls[:1])
    f_th = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_th, "SizeMin", 0.08)
    gmsh.model.mesh.field.setNumber(f_th, "SizeMax", 0.4)
    gmsh.model.mesh.field.setNumber(f_th, "DistMin", 0.2)
    gmsh.model.mesh.field.setNumber(f_th, "DistMax", 2.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_th)

    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(path))
    print(f"Wrote {path}")
    if "-nopopup" not in sys.argv:
        try:
            gmsh.fltk.run()
        except Exception:
            pass
    gmsh.finalize()


if __name__ == "__main__":
    generate()
