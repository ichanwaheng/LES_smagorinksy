#!/usr/bin/env python3
"""Entry point: run tensile membrane FSI simulation from YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running without install
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.fsi.coupling import FSISimulation
from src.utils.io import (
    ensure_dir,
    load_config,
    save_history_csv,
    save_snapshot,
    write_membrane_vtk,
)
from src.utils.viz import plot_history, plot_membrane_3d, plot_membrane_and_slice


def parse_args():
    p = argparse.ArgumentParser(
        description="Simulate tensile membrane structures under fluid flow (FSI)"
    )
    p.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(ROOT / "config" / "default.yaml"),
        help="Path to YAML config",
    )
    p.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="Override end time [s]",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override time step [s]",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Fast demo: coarse mesh, short time",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.quick:
        cfg["membrane"]["nx"] = 8
        cfg["membrane"]["ny"] = 6
        cfg["membrane"]["mass_scale"] = 120.0
        cfg["fluid"]["nx"] = 16
        cfg["fluid"]["ny"] = 8
        cfg["fluid"]["nz"] = 8
        cfg["fluid"]["U_inlet"] = 3.0
        cfg["fluid"]["nu"] = 5.0e-3
        cfg["time"]["dt"] = 0.01
        cfg["time"]["t_end"] = 0.2
        cfg["fsi"]["max_subiters"] = 1
        cfg["fsi"]["load_scale"] = 0.15
        cfg["les"]["enabled"] = False
        cfg["simulation"]["save_interval"] = 5

    if args.t_end is not None:
        cfg["time"]["t_end"] = args.t_end
    if args.dt is not None:
        cfg["time"]["dt"] = args.dt

    out = ensure_dir(ROOT / cfg["simulation"].get("output_dir", "output"))
    print(f"[FSI] output → {out}")
    print(
        f"[FSI] membrane {cfg['membrane']['nx']}x{cfg['membrane']['ny']}, "
        f"fluid {cfg['fluid']['nx']}x{cfg['fluid']['ny']}x{cfg['fluid']['nz']}, "
        f"dt={cfg['time']['dt']}, t_end={cfg['time']['t_end']}"
    )

    sim = FSISimulation(cfg)

    def on_save(simulation, info, n):
        print(
            f"  t={info['time']:.4f}s  "
            f"disp={info['max_disp']:.4e} m  "
            f"CFL={info['cfl']:.3f}  "
            f"res={info['residual']:.2e}  "
            f"|p|_max={info['pressure_max']:.2f} Pa"
        )
        save_snapshot(
            out,
            n,
            info["time"],
            simulation.membrane.state.x,
            simulation.mesh.elements,
            fluid_u=simulation.fluid.state.u,
            fluid_v=simulation.fluid.state.v,
            fluid_w=simulation.fluid.state.w,
            fluid_p=simulation.fluid.state.p,
            meta=info,
        )
        write_membrane_vtk(
            out / f"membrane_{n:06d}.vtk",
            simulation.membrane.state.x,
            simulation.mesh.elements,
            point_data={"velocity": simulation.membrane.state.v},
        )
        if cfg["simulation"].get("plot", True):
            j = simulation.grid.ny // 2
            plot_membrane_and_slice(
                simulation.membrane.state.x,
                simulation.mesh.elements,
                simulation.fluid.state.u,
                simulation.grid.x,
                simulation.grid.z,
                j,
                out / f"slice_{n:06d}.png",
                title=f"Membrane FSI  t={info['time']:.3f}s",
            )

    history = sim.run(callback=on_save)
    save_history_csv(out / "history.csv", history)
    plot_history(history, out / "history.png")
    plot_membrane_3d(
        sim.membrane.state.x,
        sim.mesh.elements,
        out / "membrane_final.png",
        displacement=sim.membrane.state.x - sim.mesh.nodes,
        title="Final membrane shape",
    )
    print(f"[FSI] done. History → {out / 'history.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
