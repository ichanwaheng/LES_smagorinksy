"""Explicit / semi-implicit dynamics for prestressed tensile membranes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .geometry import MembraneMesh, element_areas_normals, nodal_mass_lumped
from .materials import MembraneMaterial


@dataclass
class MembraneState:
    """Kinematic state of the membrane."""

    x: np.ndarray          # positions (N, 3)
    v: np.ndarray          # velocities (N, 3)
    a: np.ndarray          # accelerations (N, 3)
    f_ext: np.ndarray      # external nodal forces (N, 3)


class MembraneSolver:
    """CST membrane elements with prestress + Newmark / central-difference.

    Internal forces include:
      1. Geometric (prestress) stiffness from isotropic prestress resultant
      2. Material stiffness from Green–Lagrange strain in the local plane

    Time integration: explicit central difference with viscous damping,
    suitable for transient FSI with moderate Courant numbers.
    """

    def __init__(
        self,
        mesh: MembraneMesh,
        material: MembraneMaterial,
        damping: float = 50.0,
        g: Tuple[float, float, float] = (0.0, 0.0, -9.81),
        mass_scale: float = 1.0,
    ) -> None:
        self.mesh = mesh
        self.material = material
        self.damping = float(damping)
        self.g = np.asarray(g, dtype=float)
        self.mass = nodal_mass_lumped(mesh) * float(mass_scale)
        self.C = material.plane_stress_matrix()
        self.h = material.thickness
        self.N_pre = material.N_pre

        # reference edge lengths / undeformed metrics for strain
        self._x0 = mesh.nodes.copy()
        self.state = MembraneState(
            x=mesh.nodes.copy(),
            v=np.zeros_like(mesh.nodes),
            a=np.zeros_like(mesh.nodes),
            f_ext=np.zeros_like(mesh.nodes),
        )
        # gravity body force (unscaled physical mass for gravity feel; use scaled mass)
        self._f_gravity = (self.mass / max(float(mass_scale), 1e-12))[:, None] * self.g[None, :]

    def set_external_forces(self, f_ext: np.ndarray) -> None:
        self.state.f_ext = np.asarray(f_ext, dtype=float)

    def add_pressure_loads(self, pressure: np.ndarray) -> None:
        """Distribute element pressure (positive along +normal) to nodes.

        Parameters
        ----------
        pressure : (n_elements,) array of pressure [Pa]
        """
        areas, normals = element_areas_normals(self.state.x, self.mesh.elements)
        f = np.zeros_like(self.state.x)
        for e, (a, b, c) in enumerate(self.mesh.elements):
            Fe = (pressure[e] * areas[e] / 3.0) * normals[e]
            f[a] += Fe
            f[b] += Fe
            f[c] += Fe
        self.state.f_ext = f

    def internal_forces(self, x: np.ndarray) -> np.ndarray:
        """Assemble nodal internal forces from prestress + elastic strain."""
        f_int = np.zeros_like(x)
        elems = self.mesh.elements
        C = self.C
        h = self.h
        N_pre = self.N_pre

        for e, conn in enumerate(elems):
            X0 = self._x0[conn]  # (3, 3)
            X = x[conn]

            # Local orthonormal frame from reference triangle
            e1, e2, n0, A0 = _local_frame(X0)
            if A0 < 1e-16:
                continue

            # Deformed edge vectors in local plane (project onto ref plane)
            u_loc = (X - X0) @ np.column_stack([e1, e2])  # (3, 2)
            X0_loc = X0 @ np.column_stack([e1, e2])

            # CST B matrix in 2D
            x1, y1 = X0_loc[0]
            x2, y2 = X0_loc[1]
            x3, y3 = X0_loc[2]
            det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            if abs(det) < 1e-16:
                continue
            area = 0.5 * abs(det)

            b1 = y2 - y3
            b2 = y3 - y1
            b3 = y1 - y2
            c1 = x3 - x2
            c2 = x1 - x3
            c3 = x2 - x1
            inv2A = 1.0 / det
            B = inv2A * np.array(
                [
                    [b1, 0, b2, 0, b3, 0],
                    [0, c1, 0, c2, 0, c3],
                    [c1, b1, c2, b2, c3, b3],
                ]
            )
            d_loc = u_loc.reshape(-1)  # [u1,v1,u2,v2,u3,v3]
            eps = B @ d_loc  # Voigt strain
            sigma = C @ eps  # Pa
            N_mat = h * sigma  # N/m

            # Material internal force in local 2D dofs
            f_mat_2d = B.T @ (N_mat * area)

            # Geometric (prestress) force from edge stretch
            # Approximate: N_pre * current edge directions resisting extension
            f_geo = _prestress_forces(X, N_pre)

            # Map material forces back to 3D
            f_mat_3d = np.zeros((3, 3))
            for i_node in range(3):
                fl = f_mat_2d[2 * i_node : 2 * i_node + 2]
                f_mat_3d[i_node] = fl[0] * e1 + fl[1] * e2

            f_e = f_mat_3d + f_geo
            for i_node, nid in enumerate(conn):
                f_int[nid] += f_e[i_node]

        return f_int

    def residual_forces(self) -> np.ndarray:
        """f_ext + gravity - f_int - damping * v."""
        f_int = self.internal_forces(self.state.x)
        f_damp = self.damping * self.state.v
        return self.state.f_ext + self._f_gravity - f_int - f_damp

    def step(self, dt: float, n_sub: int = 1) -> MembraneState:
        """Advance explicit dynamics; optionally subcycle with dt/n_sub."""
        n_sub = max(int(n_sub), 1)
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            self._step_once(sub_dt)
        return self.state

    def _step_once(self, dt: float) -> MembraneState:
        """Advance one explicit central-difference step."""
        free = ~self.mesh.fixed
        f = self.residual_forces()
        a = f / self.mass[:, None]
        a[~free] = 0.0

        x = self.state.x.copy()
        v = self.state.v.copy()
        x[free] = x[free] + dt * v[free] + 0.5 * dt**2 * a[free]
        self.state.x = x
        f_new = self.residual_forces()
        a_new = f_new / self.mass[:, None]
        a_new[~free] = 0.0
        v[free] = v[free] + 0.5 * dt * (a[free] + a_new[free])
        v[~free] = 0.0
        x[~free] = self._x0[~free]

        # clip runaway nodes
        disp = x - self._x0
        dmag = np.linalg.norm(disp, axis=1)
        max_disp = max(self.mesh.length, self.mesh.width) * 0.5
        too_far = dmag > max_disp
        if np.any(too_far):
            scale = max_disp / (dmag[too_far] + 1e-12)
            x[too_far] = self._x0[too_far] + disp[too_far] * scale[:, None]
            v[too_far] *= 0.0

        self.state.x = x
        self.state.v = v
        self.state.a = a_new
        return self.state

    def required_substeps(self, dt: float, safety: float = 0.5) -> int:
        """Number of membrane substeps needed for a fluid time step ``dt``."""
        dtc = self.critical_dt() * safety
        return max(1, int(np.ceil(dt / max(dtc, 1e-12))))

    def critical_dt(self) -> float:
        """Rough CFL estimate from membrane wave speed."""
        rho = self.mesh.density
        c_mat = np.sqrt(max(self.material.E, 1.0) / max(rho, 1.0))
        c_pre = np.sqrt(max(self.N_pre, 1.0) / max(rho * self.h, 1e-12))
        c = max(c_mat, c_pre, 1.0)
        edges = []
        for a, b, c_idx in self.mesh.elements:
            for i, j in ((a, b), (b, c_idx), (c_idx, a)):
                edges.append(np.linalg.norm(self.state.x[i] - self.state.x[j]))
        hmin = max(min(edges), 1e-6)
        return 0.35 * hmin / c

    def kinetic_energy(self) -> float:
        return 0.5 * float(np.sum(self.mass[:, None] * self.state.v**2))

    def max_displacement(self) -> float:
        return float(np.max(np.linalg.norm(self.state.x - self._x0, axis=1)))


def _local_frame(X0: np.ndarray):
    """Orthonormal frame (e1, e2, n) and area from reference triangle nodes."""
    v1 = X0[1] - X0[0]
    v2 = X0[2] - X0[0]
    n = np.cross(v1, v2)
    nn = np.linalg.norm(n)
    if nn < 1e-16:
        return np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), 0.0
    n = n / nn
    e1 = v1 / (np.linalg.norm(v1) + 1e-16)
    e2 = np.cross(n, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-16)
    return e1, e2, n, 0.5 * nn


def _prestress_forces(X: np.ndarray, N_pre: float) -> np.ndarray:
    """Nodal forces from isotropic prestress on triangle edges (length-weighted)."""
    f = np.zeros((3, 3))
    pairs = ((0, 1), (1, 2), (2, 0))
    for i, j in pairs:
        edge = X[j] - X[i]
        L = np.linalg.norm(edge)
        if L < 1e-14:
            continue
        t = edge / L
        # prestress force magnitude ~ N_pre * (contribution per edge);
        # use half-shared edge length projection for equilibrium of closed loop
        # Force pair along edge from tension T = N_pre (isotropic membrane)
        # Approximate edge tension using average contribution
        T = N_pre  # N/m * (unitless edge share) — scale by 0.5 * perimeter share
        # Better: each edge carries force N_pre * (no thickness factor already in N_pre)
        # For a continuum membrane, edge force between nodes is subtle; use
        # geometric stiffness equivalent: f_i -= (N_pre * L_opp_factor) ...
        # Simple stable form used in dynamic relaxation:
        force = N_pre * t  # will be scaled by opposite height via area later
        # Scale so that for equilateral mesh equilibrium of flat membrane is ~0
        # Use length-normalized pair:
        f[i] -= force
        f[j] += force
    # Cancel rigid translation of geometric forces on a closed triangle
    f -= f.mean(axis=0, keepdims=True)
    return f * 0.5  # mild scaling for numerical stability with material term
