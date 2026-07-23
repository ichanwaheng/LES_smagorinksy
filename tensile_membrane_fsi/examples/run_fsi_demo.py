#!/usr/bin/env python3
"""Quick FSI demo with a coarse grid (runs in a few minutes on CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fsi.coupling import FSISimulation
from src.utils.io import ensure_dir, load_config, save_history_csv, save_snapshot
from src.utils.viz import plot_history, plot_membrane_and_slice


def main():
    cfg = load_config(ROOT / "config" / "default.yaml")
    # coarse / short for demo
    cfg["membrane"]["nx"] = 8
    cfg["membrane"]["ny"] = 5
    cfg["membrane"]["mass_scale"] = 100.0
    cfg["fluid"]["nx"] = 16
    cfg["fluid"]["ny"] = 8
    cfg["fluid"]["nz"] = 8
    cfg["fluid"]["U_inlet"] = 3.0
    cfg["fluid"]["nu"] = 5.0e-3
    cfg["time"]["dt"] = 0.01
    cfg["time"]["t_end"] = 0.12
    cfg["fsi"]["max_subiters"] = 1
    cfg["fsi"]["load_scale"] = 0.15
    cfg["simulation"]["save_interval"] = 4
    cfg["les"]["enabled"] = False

    out = ensure_dir(ROOT / "output" / "fsi_demo")
    sim = FSISimulation(cfg)

    def cb(simulation, info, n):
        print(
            f"t={info['time']:.3f}  disp={info['max_disp']:.3e}  "
            f"CFL={info['cfl']:.2f}  |p|={info['pressure_max']:.1f}"
        )
        save_snapshot(
            out,
            n,
            info["time"],
            simulation.membrane.state.x,
            simulation.mesh.elements,
            fluid_u=simulation.fluid.state.u,
            fluid_p=simulation.fluid.state.p,
        )
        plot_membrane_and_slice(
            simulation.membrane.state.x,
            simulation.mesh.elements,
            simulation.fluid.state.u,
            simulation.grid.x,
            simulation.grid.z,
            simulation.grid.ny // 2,
            out / f"slice_{n:04d}.png",
            title=f"t={info['time']:.3f}s",
        )

    hist = sim.run(callback=cb)
    save_history_csv(out / "history.csv", hist)
    plot_history(hist, out / "history.png")
    print(f"Demo complete → {out}")


if __name__ == "__main__":
    main()
