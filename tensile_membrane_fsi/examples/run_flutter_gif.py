#!/usr/bin/env python3
"""Run the coupled fluid–membrane simulation and write a flutter GIF.

Simulates channel flow (LES) past a light, softly prestressed tensile
membrane with a gusty inflow, then renders an animated GIF of the
membrane fluttering:

    python examples/run_flutter_gif.py                 # full run (~6 s physical)
    python examples/run_flutter_gif.py --t-end 2.0     # shorter run
    python examples/run_flutter_gif.py --out my.gif    # custom output path

The GIF shows the 3D membrane surface (coloured by vertical displacement)
and the mid-plane fluid speed slice with the membrane side profile.
"""

from __future__ import annotations

import argparse
import sys
import time as walltime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fsi.coupling import FSISimulation
from src.utils.io import ensure_dir, load_config, save_history_csv
from src.utils.viz import plot_history, render_flutter_frame, save_gif


def parse_args():
    p = argparse.ArgumentParser(description="Membrane flutter FSI → animated GIF")
    p.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(ROOT / "config" / "flutter.yaml"),
        help="Path to YAML config (default: config/flutter.yaml)",
    )
    p.add_argument("--t-end", type=float, default=None, help="Override end time [s]")
    p.add_argument("--fps", type=int, default=12, help="GIF frames per second")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output GIF path (default: <output_dir>/membrane_flutter.gif)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.t_end is not None:
        cfg["time"]["t_end"] = args.t_end

    out = ensure_dir(ROOT / cfg["simulation"].get("output_dir", "output/flutter"))
    gif_path = Path(args.out) if args.out else out / "membrane_flutter.gif"

    sim = FSISimulation(cfg)
    nodes0 = sim.membrane.state.x.copy()
    mesh_nx, mesh_ny = sim.mesh.nx, sim.mesh.ny
    j_slice = sim.grid.ny // 2

    # fixed scales across all frames so the flutter reads clearly
    U = float(cfg["fluid"]["U_inlet"])
    gust = float(cfg["fluid"].get("gust_amp", 0.0))
    speed_max = 1.6 * U * (1.0 + gust)
    z0 = float(cfg["fluid"]["membrane_z0"])
    z_span = 0.25 * float(cfg["fluid"]["domain"]["H"])
    z_limits = (z0 - z_span, z0 + z_span)
    disp_max = 0.3  # colour scale for Δz [m]

    frames = []
    t0 = walltime.time()

    def on_frame(simulation, info, n):
        st = simulation.fluid.state
        speed = np.sqrt(
            st.u[:, j_slice, :] ** 2
            + st.v[:, j_slice, :] ** 2
            + st.w[:, j_slice, :] ** 2
        )
        frames.append(
            render_flutter_frame(
                simulation.membrane.state.x,
                simulation.mesh.elements,
                nodes0,
                speed,
                simulation.grid.x,
                simulation.grid.z,
                info["time"],
                mesh_nx,
                mesh_ny,
                speed_max,
                disp_max,
                z_limits,
            )
        )
        print(
            f"  frame {len(frames):3d}  t={info['time']:6.3f}s  "
            f"disp={info['max_disp']:.3f} m  CFL={info['cfl']:.2f}  "
            f"[{walltime.time() - t0:6.1f}s wall]"
        )

    print(f"[flutter] fluid {sim.grid.nx}x{sim.grid.ny}x{sim.grid.nz}, "
          f"membrane {mesh_nx}x{mesh_ny}, dt={sim.dt}, t_end={sim.t_end}")
    history = sim.run(callback=on_frame)

    save_history_csv(out / "history.csv", history)
    plot_history(history, out / "history.png")
    save_gif(frames, gif_path, fps=args.fps)
    print(f"[flutter] {len(frames)} frames → {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
