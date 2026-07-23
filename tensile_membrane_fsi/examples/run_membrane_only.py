#!/usr/bin/env python3
"""Membrane-only demo: prestress form-finding + dynamic response to a gust load."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.membrane import (
    MembraneMaterial,
    apply_isotropic_prestress,
    build_rectangular_membrane,
)
from src.utils.io import ensure_dir, write_membrane_vtk
from src.utils.viz import plot_membrane_3d


def main():
    out = ensure_dir(ROOT / "output" / "membrane_only")
    mesh = build_rectangular_membrane(
        length=2.0, width=1.5, nx=20, ny=14, fixed_edges=["left", "right"]
    )
    material = MembraneMaterial(E=5e8, nu=0.3, thickness=0.001, prestress=5e4)
    solver = apply_isotropic_prestress(mesh, material, n_steps=100)
    write_membrane_vtk(out / "prestressed.vtk", solver.state.x, mesh.elements)
    plot_membrane_3d(solver.state.x, mesh.elements, out / "prestressed.png", title="Prestress form")

    # Apply a uniform pressure gust for a short transient
    n_elem = mesh.n_elements
    pressure = np.full(n_elem, 200.0)  # Pa
    solver.damping = 30.0
    dt = min(solver.critical_dt(), 2e-4)
    print(f"dt={dt:.3e}, steps=400")
    for step in range(400):
        solver.add_pressure_loads(pressure)
        solver.step(dt)
        if step % 50 == 0:
            print(f"  step {step}: max disp = {solver.max_displacement():.4e} m")

    plot_membrane_3d(
        solver.state.x,
        mesh.elements,
        out / "after_gust.png",
        displacement=solver.state.x - mesh.nodes,
        title="After pressure gust",
    )
    write_membrane_vtk(
        out / "after_gust.vtk",
        solver.state.x,
        mesh.elements,
        point_data={"velocity": solver.state.v},
    )
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
