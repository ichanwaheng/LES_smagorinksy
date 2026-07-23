"""Prestress / form-finding helpers for tensile membranes."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .geometry import MembraneMesh
from .materials import MembraneMaterial
from .solver import MembraneSolver


def apply_isotropic_prestress(
    mesh: MembraneMesh,
    material: MembraneMaterial,
    damping: float = 200.0,
    n_steps: int = 200,
    dt: Optional[float] = None,
    mass_scale: float = 50.0,
) -> MembraneSolver:
    """Relax membrane under prestress + gravity to a stable initial form.

    Uses heavily damped dynamics (dynamic relaxation) so the starting
    configuration for FSI is near equilibrium.
    """
    solver = MembraneSolver(mesh, material, damping=damping, mass_scale=mass_scale)
    if dt is None:
        dt = min(solver.critical_dt(), 5e-4)
    solver.set_external_forces(np.zeros_like(mesh.nodes))
    for i in range(n_steps):
        solver.step(dt, n_sub=1)
        if i % 20 == 0:
            solver.state.v *= 0.5
    solver.state.v[:] = 0.0
    solver.state.a[:] = 0.0
    return solver


def initial_sag_shape(
    mesh: MembraneMesh,
    sag: float = 0.05,
) -> np.ndarray:
    """Analytical catenary-like initial z-displacement for visualization."""
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    x0, y0 = x.min(), y.min()
    L, W = mesh.length, mesh.width
    xi = (x - x0) / max(L, 1e-12)
    eta = (y - y0) / max(W, 1e-12)
    dz = -sag * np.sin(np.pi * xi) * np.sin(np.pi * eta)
    nodes = mesh.nodes.copy()
    nodes[:, 2] += dz
    return nodes
