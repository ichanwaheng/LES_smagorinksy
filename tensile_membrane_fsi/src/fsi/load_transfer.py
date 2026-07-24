"""Transfer fluid loads onto the membrane surface."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..fluid.piso import FluidSolver
from ..membrane.geometry import MembraneMesh, element_areas_normals


def element_centroids(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    return nodes[elements].mean(axis=1)


def dynamic_pressure_loads(
    fluid: FluidSolver,
    mesh: MembraneMesh,
    nodes: np.ndarray,
    rho: float,
    U_ref: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate element pressures from dynamic pressure and local incidence.

    Uses Bernoulli-like estimate:
        Δp = 0.5 ρ U_n |U_n|
    where U_n is the fluid velocity relative to the membrane, projected
    onto the membrane normal (positive pressure pushes along +normal).

    Returns
    -------
    pressure : (n_elem,) Pa
    f_nodal : (n_nodes, 3) nodal forces
    """
    centroids = element_centroids(nodes, mesh.elements)
    areas, normals = element_areas_normals(nodes, mesh.elements)
    vel, p_field = fluid.sample_at(centroids)

    # relative velocity ≈ fluid (membrane velocity neglected in pressure estimate;
    # caller may subtract membrane velocity if desired)
    Un = np.sum(vel * normals, axis=1)
    q_dyn = 0.5 * rho * Un * np.abs(Un)
    # blend a fraction of interpolated static pressure difference vs freestream
    p_inf = 0.0
    pressure = q_dyn + 0.3 * (p_field - p_inf)

    # Cap extreme values for stability (stall / singularity protection)
    q_ref = 0.5 * rho * max(U_ref, 1e-6) ** 2
    pressure = np.clip(pressure, -5 * q_ref, 5 * q_ref)

    f_nodal = np.zeros_like(nodes)
    for e, (a, b, c) in enumerate(mesh.elements):
        Fe = (pressure[e] * areas[e] / 3.0) * normals[e]
        f_nodal[a] += Fe
        f_nodal[b] += Fe
        f_nodal[c] += Fe
    return pressure, f_nodal


def pressure_jump_loads(
    fluid: FluidSolver,
    mesh: MembraneMesh,
    nodes: np.ndarray,
    rho: float,
    U_ref: float,
    offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Membrane lift from the pressure difference between its two sides.

    Samples fluid pressure at ``centroid ± offset * normal`` (just outside
    the immersed-boundary band) and applies
        Δp = p(-n side) - p(+n side)
    as a load along +normal. This captures the unsteady lift on a membrane
    aligned with the flow, which the incidence/dynamic-pressure model misses.
    """
    centroids = element_centroids(nodes, mesh.elements)
    areas, normals = element_areas_normals(nodes, mesh.elements)
    _, p_plus = fluid.sample_at(centroids + offset * normals)
    _, p_minus = fluid.sample_at(centroids - offset * normals)
    pressure = p_minus - p_plus

    q_ref = 0.5 * rho * max(U_ref, 1e-6) ** 2
    pressure = np.clip(pressure, -5 * q_ref, 5 * q_ref)

    f_nodal = np.zeros_like(nodes)
    for e, (a, b, c) in enumerate(mesh.elements):
        Fe = (pressure[e] * areas[e] / 3.0) * normals[e]
        f_nodal[a] += Fe
        f_nodal[b] += Fe
        f_nodal[c] += Fe
    return pressure, f_nodal


def interpolated_field_loads(
    fluid: FluidSolver,
    mesh: MembraneMesh,
    nodes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Use interpolated fluid pressure directly as membrane surface pressure."""
    centroids = element_centroids(nodes, mesh.elements)
    areas, normals = element_areas_normals(nodes, mesh.elements)
    _, p_field = fluid.sample_at(centroids)
    # pressure jump ≈ p_front - p_back; without a thin-gap model, use p itself
    # relative to domain mean
    pressure = p_field - float(np.mean(p_field))
    f_nodal = np.zeros_like(nodes)
    for e, (a, b, c) in enumerate(mesh.elements):
        Fe = (pressure[e] * areas[e] / 3.0) * normals[e]
        f_nodal[a] += Fe
        f_nodal[b] += Fe
        f_nodal[c] += Fe
    return pressure, f_nodal
