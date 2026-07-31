"""Transfer fluid loads onto the membrane surface.

Continuous fluid loading is applied every outer FSI / quasi-static step from
the *current* deformed geometry. Prefer ``pressure_jump`` for thin membranes:
it samples fluid pressure on both sides of the sheet so the net load can
reverse (up or down) as the flow and shape evolve. The older
``dynamic_pressure`` incidence model often locks onto a one-sided force after
the membrane sags because centroids fall inside the immersed-boundary band.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..fluid.piso import FluidSolver
from ..membrane.geometry import MembraneMesh, element_areas_normals


def element_centroids(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    return nodes[elements].mean(axis=1)


def _orient_normals_up(normals: np.ndarray) -> np.ndarray:
    """Flip element normals so the +z component is non-negative.

    Keeps a consistent 'top' side for two-sided pressure sampling as the
    membrane folds / sags.
    """
    n = np.asarray(normals, dtype=float).copy()
    flip = n[:, 2] < 0.0
    n[flip] *= -1.0
    return n


def _scatter_pressure_to_nodes(
    nodes: np.ndarray,
    elements: np.ndarray,
    areas: np.ndarray,
    normals: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    f_nodal = np.zeros_like(nodes)
    for e, (a, b, c) in enumerate(elements):
        Fe = (pressure[e] * areas[e] / 3.0) * normals[e]
        f_nodal[a] += Fe
        f_nodal[b] += Fe
        f_nodal[c] += Fe
    return f_nodal


def dynamic_pressure_loads(
    fluid: FluidSolver,
    mesh: MembraneMesh,
    nodes: np.ndarray,
    rho: float,
    U_ref: float,
    offset: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate loads from incidence + two-sided pressure samples.

    Samples velocity/pressure just outside the immersed-boundary band on both
    sides of the membrane so loading remains defined after large deflections.
    Net pressure is
        Δp = ½ ρ U_n |U_n| + 0.5 (p_minus − p_plus)
    applied along the upward-oriented normal.
    """
    centroids = element_centroids(nodes, mesh.elements)
    areas, normals = element_areas_normals(nodes, mesh.elements)
    normals = _orient_normals_up(normals)
    if offset is None:
        # use fluid grid spacing when available
        offset = 3.0 * float(getattr(getattr(fluid, "grid", None), "dz", 0.05))

    # sample outside the IB band on both sides
    c_plus = centroids + offset * normals
    c_minus = centroids - offset * normals
    vel_p, p_plus = fluid.sample_at(c_plus)
    vel_m, p_minus = fluid.sample_at(c_minus)
    # average near-surface velocity for incidence
    vel = 0.5 * (vel_p + vel_m)
    Un = np.sum(vel * normals, axis=1)
    q_dyn = 0.5 * rho * Un * np.abs(Un)
    # continuous two-sided fluid pressure contribution
    p_jump = p_minus - p_plus
    pressure = q_dyn + 0.5 * p_jump

    q_ref = 0.5 * rho * max(U_ref, 1e-6) ** 2
    pressure = np.clip(pressure, -5 * q_ref, 5 * q_ref)
    return pressure, _scatter_pressure_to_nodes(
        nodes, mesh.elements, areas, normals, pressure
    )


def pressure_jump_loads(
    fluid: FluidSolver,
    mesh: MembraneMesh,
    nodes: np.ndarray,
    rho: float,
    U_ref: float,
    offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Membrane load from the continuous pressure difference between sides.

    Samples fluid pressure at ``centroid ± offset * normal`` (outside the
    immersed-boundary band) and applies
        Δp = p(-n side) - p(+n side)
    along the upward normal. As the membrane deflects and the flow evolves,
    Δp can change sign — so the sheet can be driven down *and* back up.
    """
    centroids = element_centroids(nodes, mesh.elements)
    areas, normals = element_areas_normals(nodes, mesh.elements)
    normals = _orient_normals_up(normals)
    _, p_plus = fluid.sample_at(centroids + offset * normals)
    _, p_minus = fluid.sample_at(centroids - offset * normals)
    pressure = p_minus - p_plus

    q_ref = 0.5 * rho * max(U_ref, 1e-6) ** 2
    pressure = np.clip(pressure, -5 * q_ref, 5 * q_ref)
    return pressure, _scatter_pressure_to_nodes(
        nodes, mesh.elements, areas, normals, pressure
    )


def interpolated_field_loads(
    fluid: FluidSolver,
    mesh: MembraneMesh,
    nodes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-sided interpolated pressure jump (fallback continuous load)."""
    centroids = element_centroids(nodes, mesh.elements)
    areas, normals = element_areas_normals(nodes, mesh.elements)
    normals = _orient_normals_up(normals)
    # without a known grid spacing, use a small fraction of span
    span = float(np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0))) + 1e-6
    offset = max(0.05 * span, 1e-3)
    _, p_plus = fluid.sample_at(centroids + offset * normals)
    _, p_minus = fluid.sample_at(centroids - offset * normals)
    pressure = p_minus - p_plus
    return pressure, _scatter_pressure_to_nodes(
        nodes, mesh.elements, areas, normals, pressure
    )
