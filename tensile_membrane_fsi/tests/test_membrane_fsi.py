"""Unit tests for membrane geometry and a short FSI smoke run."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.membrane import (
    MembraneMaterial,
    MembraneSolver,
    build_rectangular_membrane,
    nodal_mass_lumped,
)
from src.fluid import FluidGrid, FluidSolver
from src.fsi.coupling import FSISimulation
from src.utils.io import load_config


def test_rectangular_mesh_topology():
    mesh = build_rectangular_membrane(nx=4, ny=3, fixed_edges=["left", "right"])
    assert mesh.n_nodes == 5 * 4
    assert mesh.n_elements == 4 * 3 * 2
    assert mesh.fixed.sum() == 2 * 4  # left + right columns
    mass = nodal_mass_lumped(mesh)
    assert np.all(mass > 0)
    assert abs(mass.sum() - mesh.density * mesh.thickness * mesh.length * mesh.width) < 1e-6


def test_membrane_step_stable():
    mesh = build_rectangular_membrane(nx=6, ny=4, fixed_edges=["left", "right"])
    mat = MembraneMaterial(prestress=1e4)
    solver = MembraneSolver(mesh, mat, damping=100.0)
    dt = min(solver.critical_dt(), 1e-4)
    for _ in range(20):
        solver.step(dt)
    assert np.isfinite(solver.state.x).all()
    assert solver.max_displacement() < 1.0


def test_fluid_step():
    grid = FluidGrid(L=4, W=2, H=2, nx=8, ny=4, nz=4)
    fluid = FluidSolver(grid, U_inlet=5.0, use_les=False)
    fluid.step(0.01)
    assert np.isfinite(fluid.state.u).all()
    assert fluid.state.u[0].mean() == pytest.approx(5.0)


def test_fsi_smoke():
    cfg = load_config(ROOT / "config" / "default.yaml")
    cfg["membrane"]["nx"] = 6
    cfg["membrane"]["ny"] = 4
    cfg["fluid"]["nx"] = 10
    cfg["fluid"]["ny"] = 6
    cfg["fluid"]["nz"] = 6
    cfg["time"]["dt"] = 0.01
    cfg["time"]["t_end"] = 0.03
    cfg["fsi"]["max_subiters"] = 1
    cfg["les"]["enabled"] = False
    sim = FSISimulation(cfg)
    info = sim.step()
    assert info["time"] > 0
    assert np.isfinite(info["max_disp"])
