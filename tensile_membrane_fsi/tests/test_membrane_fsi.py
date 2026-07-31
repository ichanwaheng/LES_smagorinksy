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
    fluid = FluidSolver(grid, U_inlet=5.0, use_les=False, n_correctors=2)
    fluid.step(0.01)
    assert np.isfinite(fluid.state.u).all()
    assert np.isfinite(fluid.state.p).all()
    # inlet profile tapers to zero at walls; peak equals U_inlet
    assert fluid.state.u[0].max() == pytest.approx(5.0, rel=0.05)
    assert fluid.state.u[0].mean() > 0.5 * 5.0


def test_piso_les_correctors_reduce_divergence():
    """PISO with LES: more correctors should not blow up; fields stay finite."""
    grid = FluidGrid(L=4, W=2, H=2, nx=8, ny=4, nz=4)
    fluid = FluidSolver(
        grid, U_inlet=5.0, use_les=True, Cs=0.17, nu=1e-3, n_correctors=2
    )
    for _ in range(5):
        fluid.step(0.005)
    assert np.isfinite(fluid.state.u).all()
    assert np.isfinite(fluid.state.v).all()
    assert np.isfinite(fluid.state.w).all()
    assert np.isfinite(fluid.state.p).all()
    assert fluid.n_correctors == 2
    assert np.all(fluid.state.nu_eff >= fluid.nu)

def test_gusty_inlet():
    grid = FluidGrid(L=4, W=2, H=2, nx=8, ny=4, nz=4)
    fluid = FluidSolver(grid, U_inlet=5.0, use_les=False, gust_amp=0.5, gust_freq=1.0)
    # step to quarter gust period → sin(2π f t) = 1 → maximum fluctuation
    for _ in range(25):
        fluid.step(0.01)
    assert np.isfinite(fluid.state.u).all()
    assert np.abs(fluid.state.w[0]).max() > 0.1  # vertical gust present at inlet
    assert fluid.state.u[0].max() > 5.0  # streamwise fluctuation above mean


def test_pressure_jump_loads():
    from src.fsi.load_transfer import pressure_jump_loads

    grid = FluidGrid(L=4, W=2, H=2, nx=8, ny=4, nz=4)
    fluid = FluidSolver(grid, U_inlet=5.0, use_les=False)
    fluid.step(0.01)
    mesh = build_rectangular_membrane(
        length=1.0, width=0.5, nx=4, ny=3, origin=(1.5, 0.75, 1.0)
    )
    pressure, f_nodal = pressure_jump_loads(
        fluid, mesh, mesh.nodes, rho=1.225, U_ref=5.0, offset=0.5
    )
    assert pressure.shape == (mesh.n_elements,)
    assert f_nodal.shape == mesh.nodes.shape
    assert np.isfinite(pressure).all()
    assert np.isfinite(f_nodal).all()


def test_flutter_frame_and_gif(tmp_path):
    from src.utils.viz import render_flutter_frame, save_gif

    mesh = build_rectangular_membrane(length=1.0, width=0.5, nx=4, ny=3)
    speed = np.random.default_rng(0).random((8, 6))
    gx = np.linspace(0, 4, 8)
    gz = np.linspace(0, 2, 6)
    frames = [
        render_flutter_frame(
            mesh.nodes, mesh.elements, mesh.nodes, speed, gx, gz,
            time=0.1 * i, mesh_nx=4, mesh_ny=3,
            speed_max=5.0, disp_max=0.3, z_limits=(-0.5, 0.5),
        )
        for i in range(2)
    ]
    gif = save_gif(frames, tmp_path / "test.gif", fps=5)
    assert gif.exists() and gif.stat().st_size > 0


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
