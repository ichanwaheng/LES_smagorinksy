"""Membrane constitutive model: prestressed isotropic linear elastic fabric."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MembraneMaterial:
    """Plane-stress isotropic membrane with isotropic prestress.

    Stress resultants (force/length) are used in the weak form:
        N = h * (σ_pre + C : ε)
    where ε is the Green–Lagrange strain in the local membrane plane
    (linearized for small strain increments; large-displacement geometry
    is handled by updating the reference normals each step).
    """

    E: float = 5.0e8
    nu: float = 0.3
    thickness: float = 0.001
    prestress: float = 5.0e4  # isotropic σ_pre [Pa]

    def plane_stress_matrix(self) -> np.ndarray:
        """3x3 Voigt C for plane stress (εxx, εyy, γxy)."""
        E, nu = self.E, self.nu
        factor = E / (1.0 - nu**2)
        return factor * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ]
        )

    @property
    def N_pre(self) -> float:
        """Isotropic prestress resultant [N/m]."""
        return self.prestress * self.thickness
