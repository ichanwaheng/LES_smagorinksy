"""Partitioned FSI driver: fluid ↔ tensile membrane."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..fluid.mesh import FluidGrid
from ..fluid.piso import FluidSolver
from ..membrane.geometry import build_rectangular_membrane
from ..membrane.materials import MembraneMaterial
from ..membrane.prestress import apply_isotropic_prestress, initial_sag_shape
from ..membrane.solver import MembraneSolver
from .load_transfer import (
    dynamic_pressure_loads,
    interpolated_field_loads,
    pressure_jump_loads,
)
from .mesh_update import under_relax, update_immersed_boundary


@dataclass
class FSIHistory:
    time: List[float] = field(default_factory=list)
    max_disp: List[float] = field(default_factory=list)
    kinetic: List[float] = field(default_factory=list)
    cfl: List[float] = field(default_factory=list)
    residual: List[float] = field(default_factory=list)


class FSISimulation:
    """Serial staggered FSI coupling.

    Loop per time step
    ------------------
    1. Update immersed boundary from membrane position/velocity
    2. Advance fluid (PISO after Smagorinsky LES discretisation)
    3. Transfer pressure / dynamic loads → membrane nodes
    4. Advance membrane dynamics
    5. Optional under-relaxed sub-iterations
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        mcfg = cfg["membrane"]
        fcfg = cfg["fluid"]
        tcfg = cfg["time"]
        fsi = cfg["fsi"]
        les = cfg.get("les", {})

        origin = (
            fcfg["membrane_x0"],
            fcfg["membrane_y0"],
            fcfg["membrane_z0"],
        )
        self.mesh = build_rectangular_membrane(
            length=mcfg["length"],
            width=mcfg["width"],
            nx=mcfg["nx"],
            ny=mcfg["ny"],
            origin=origin,
            fixed_edges=mcfg.get("fixed_edges", ["left", "right"]),
            thickness=mcfg["thickness"],
            density=mcfg["density"],
        )
        # slight initial sag so flow sees a curved sail
        self.mesh.nodes = initial_sag_shape(self.mesh, sag=0.04)
        self.mesh.areas0, self.mesh.normals0 = self.mesh.update_geometry(self.mesh.nodes)

        material = MembraneMaterial(
            E=float(mcfg["youngs_modulus"]),
            nu=float(mcfg["poisson"]),
            thickness=float(mcfg["thickness"]),
            prestress=float(mcfg["prestress"]),
        )
        self.membrane = apply_isotropic_prestress(
            self.mesh,
            material,
            damping=mcfg.get("damping", 50.0),
            n_steps=60,
            mass_scale=float(mcfg.get("mass_scale", 80.0)),
        )
        # restore user damping for transient
        self.membrane.damping = float(mcfg.get("damping", 50.0))

        self.grid = FluidGrid(
            L=float(fcfg["domain"]["L"]),
            W=float(fcfg["domain"]["W"]),
            H=float(fcfg["domain"]["H"]),
            nx=int(fcfg["nx"]),
            ny=int(fcfg["ny"]),
            nz=int(fcfg["nz"]),
        )
        # bump molecular viscosity on coarse teaching grids for stability
        nu = float(fcfg["nu"])
        self.fluid = FluidSolver(
            self.grid,
            rho=float(fcfg["rho"]),
            nu=nu,
            U_inlet=float(fcfg["U_inlet"]),
            use_les=les.get("enabled", True),
            Cs=float(les.get("Cs", 0.17)),
            gust_amp=float(fcfg.get("gust_amp", 0.0)),
            gust_freq=float(fcfg.get("gust_freq", 1.0)),
            u_clip=float(fcfg["u_clip"]) if "u_clip" in fcfg else None,
            n_correctors=int(les.get("piso_correctors", fcfg.get("piso_correctors", 2))),
        )
        update_immersed_boundary(
            self.fluid, self.grid, self.mesh, self.membrane.state.x, self.membrane.state.v
        )

        self.dt = float(tcfg["dt"])
        self.t_end = float(tcfg["t_end"])
        self.alpha = float(fsi.get("under_relaxation", 0.6))
        self.max_subiters = int(fsi.get("max_subiters", 5))
        self.residual_tol = float(fsi.get("residual_tol", 1e-3))
        self.load_mode = fsi.get("load_mode", "dynamic_pressure")
        self.load_scale = float(fsi.get("load_scale", 0.25))
        self.history = FSIHistory()
        self.time = 0.0
        self.step_id = 0

    def _compute_loads(self):
        if self.load_mode == "interpolated_field":
            pressure, f_nodal = interpolated_field_loads(
                self.fluid, self.mesh, self.membrane.state.x
            )
        elif self.load_mode == "pressure_jump":
            # probe just outside the immersed-boundary band (1.5 dz thick)
            offset = 3.0 * self.grid.dz
            pressure, f_nodal = pressure_jump_loads(
                self.fluid,
                self.mesh,
                self.membrane.state.x,
                rho=float(self.cfg["fluid"]["rho"]),
                U_ref=float(self.cfg["fluid"]["U_inlet"]),
                offset=offset,
            )
        else:
            pressure, f_nodal = dynamic_pressure_loads(
                self.fluid,
                self.mesh,
                self.membrane.state.x,
                rho=float(self.cfg["fluid"]["rho"]),
                U_ref=float(self.cfg["fluid"]["U_inlet"]),
            )
        return pressure, f_nodal * self.load_scale

    def step(self) -> Dict[str, float]:
        """Advance one coupled FSI time step (with optional sub-iterations)."""
        x_prev = self.membrane.state.x.copy()
        residual = np.inf
        pressure = None
        n_mem = min(self.membrane.required_substeps(self.dt), 200)

        for _sub in range(self.max_subiters):
            update_immersed_boundary(
                self.fluid,
                self.grid,
                self.mesh,
                self.membrane.state.x,
                self.membrane.state.v,
            )
            self.fluid.step(self.dt)
            pressure, f_nodal = self._compute_loads()
            if _sub == 0:
                f_use = f_nodal
            else:
                f_use = under_relax(f_nodal, getattr(self, "_f_old", f_nodal), self.alpha)
            self._f_old = f_use
            self.membrane.set_external_forces(f_use)
            self.membrane.step(self.dt, n_sub=n_mem)

            dx = self.membrane.state.x - x_prev
            residual = float(np.linalg.norm(dx) / (np.linalg.norm(x_prev) + 1e-12))
            x_prev = self.membrane.state.x.copy()
            if residual < self.residual_tol:
                break

        self.time += self.dt
        self.step_id += 1
        cfl = self.fluid.max_cfl(self.dt)
        info = {
            "time": self.time,
            "max_disp": self.membrane.max_displacement(),
            "kinetic": self.membrane.kinetic_energy(),
            "cfl": cfl,
            "residual": residual,
            "pressure_max": float(np.max(np.abs(pressure))) if pressure is not None else 0.0,
            "membrane_substeps": float(n_mem),
        }
        self.history.time.append(info["time"])
        self.history.max_disp.append(info["max_disp"])
        self.history.kinetic.append(info["kinetic"])
        self.history.cfl.append(info["cfl"])
        self.history.residual.append(info["residual"])
        return info

    def run(self, callback=None) -> FSIHistory:
        n_steps = int(np.ceil(self.t_end / self.dt))
        save_every = int(self.cfg.get("simulation", {}).get("save_interval", 10))
        for n in range(n_steps):
            info = self.step()
            if callback is not None and (n % save_every == 0 or n == n_steps - 1):
                callback(self, info, n)
        return self.history
