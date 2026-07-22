#!/usr/bin/env python3
"""
Flexible rectangular membrane (solid) model.

Physics change vs rigid sphere:
  - Rigid no-slip obstacle  →  flexible thin plate with inertia, damping, bending stiffness
  - Displacement η(s) along a spanwise cantilever strip (clamped at bottom)
  - Driven by fluid pressure difference (front/back) and shear traction

Discrete model: 1D cantilever chain of N masses along plate height (z),
deflecting in streamwise x. Bending approximated by discrete torsional/bending springs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MembraneSolidConfig:
    # Geometry (must match mesh meta)
    plate_x: float = 3.0
    plate_t: float = 0.08
    plate_w: float = 2.0
    plate_h: float = 2.0
    plate_yc: float = 2.5
    plate_zc: float = 2.5

    # Solid material / structural params (educational values)
    rho_s: float = 20.0  # membrane mass density (kg/m^3 of plate volume)
    E: float = 8.0e2  # Young's modulus (soft flexible plate for visible FSI demo)
    nu_s: float = 0.3  # Poisson
    thickness: float = 0.08  # structural thickness (~ plate_t)
    damping_zeta: float = 0.08  # modal-ish damping factor
    n_nodes: int = 16  # discrete nodes along height
    clamp: str = "bottom"  # bottom edge clamped
    max_deflection: float = 0.35  # soft clip for stability on fixed mesh


class CantileverMembrane:
    """
    Clamped-free cantilever discretised along height z.
    Unknowns: streamwise deflection eta[i], velocity v[i] at node i.
    """

    def __init__(self, cfg: MembraneSolidConfig):
        self.cfg = cfg
        n = cfg.n_nodes
        z0 = cfg.plate_zc - 0.5 * cfg.plate_h
        self.z = np.linspace(z0, z0 + cfg.plate_h, n)
        self.dz = float(self.z[1] - self.z[0]) if n > 1 else cfg.plate_h

        # Mass per node from plate volume share
        volume = cfg.plate_t * cfg.plate_w * cfg.plate_h
        m_total = cfg.rho_s * volume
        self.mass = np.full(n, m_total / n, dtype=np.float64)

        # Bending rigidity for plate strip: EI ~ E * (w * t^3 / 12)
        I = cfg.plate_w * (cfg.thickness**3) / 12.0
        self.EI = cfg.E * I
        # Discrete bending spring coefficient between consecutive segments
        # M = EI * kappa ≈ EI * (eta_{i-1} - 2 eta_i + eta_{i+1}) / dz^2
        # Force from bending ~ difference of shears.
        self.k_bend = self.EI / (self.dz**3 + 1e-30)

        # Light axial spring to rest plane (restoring toward eta=0), weaker than bending
        self.k_rest = 0.05 * self.k_bend

        # Critical-ish damping scale
        omega_est = np.sqrt(max(self.k_bend, 1e-12) / max(self.mass.mean(), 1e-12))
        self.c_damp = 2.0 * cfg.damping_zeta * self.mass * omega_est

        self.eta = np.zeros(n, dtype=np.float64)  # deflection in +x
        self.v = np.zeros(n, dtype=np.float64)  # velocity in +x
        self.history: list[dict] = []

        # Clamp bottom node
        self.eta[0] = 0.0
        self.v[0] = 0.0

    def bending_forces(self, eta: np.ndarray) -> np.ndarray:
        """Discrete beam bending forces + weak restoring spring."""
        n = len(eta)
        f = np.zeros(n, dtype=np.float64)
        # second difference curvature forces
        for i in range(1, n - 1):
            kappa = (eta[i - 1] - 2.0 * eta[i] + eta[i + 1]) / (self.dz**2)
            # shear jump ~ EI dkappa/ds → force on node
            # simplified stencil:
            f[i - 1] += self.k_bend * (eta[i] - eta[i - 1])
            f[i] -= 2.0 * self.k_bend * (eta[i] - 0.5 * (eta[i - 1] + eta[i + 1]))
            f[i + 1] += self.k_bend * (eta[i] - eta[i + 1])
        f += -self.k_rest * eta
        # clamp
        f[0] = 0.0
        return f

    def step(self, f_fluid: np.ndarray, dt: float) -> None:
        """
        Semi-implicit structural step:
          m a = f_fluid + f_bending - c v
          v += dt a; eta += dt v
        f_fluid: streamwise force on each node (N,)
        """
        cfg = self.cfg
        f_s = self.bending_forces(self.eta)
        f_tot = f_fluid + f_s - self.c_damp * self.v
        a = f_tot / self.mass
        a[0] = 0.0

        self.v = self.v + dt * a
        self.v[0] = 0.0
        self.eta = self.eta + dt * self.v
        self.eta[0] = 0.0

        # Soft clip large deflection (fixed-mesh FSI stability)
        self.eta = np.clip(self.eta, -cfg.max_deflection, cfg.max_deflection)

    def tip_deflection(self) -> float:
        return float(self.eta[-1])

    def velocity_at_z(self, z_query: np.ndarray) -> np.ndarray:
        """Interpolate structural streamwise velocity to query z locations."""
        return np.interp(z_query, self.z, self.v, left=0.0, right=self.v[-1])

    def deflection_at_z(self, z_query: np.ndarray) -> np.ndarray:
        return np.interp(z_query, self.z, self.eta, left=0.0, right=self.eta[-1])
