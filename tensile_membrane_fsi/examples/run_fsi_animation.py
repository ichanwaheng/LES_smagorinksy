#!/usr/bin/env python3
"""Run a coupled FSI transient and render animated GIFs:

  * membrane_flutter.gif  -- 3-D membrane motion (fluttering) over time
  * flow_past_membrane.gif -- mid-plane |U| field with the membrane cross-section

Uses the package's real transient timesteps (central-difference membrane +
projection fluid, staggered coupling), so the animation shows the genuine
time evolution of the fluid-structure interaction.

    python examples/run_fsi_animation.py
    python examples/run_fsi_animation.py --t-end 1.5 --save-interval 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fsi.coupling import FSISimulation
from src.utils.io import ensure_dir, load_config
from src.utils.viz import animate_flow_slice, animate_membrane_3d


def build_config(t_end: float, save_interval: int):
    cfg = load_config(str(ROOT / "config" / "default.yaml"))
    # coarse but animate-worthy: enough steps for a smooth GIF
    cfg["membrane"]["nx"] = 12
    cfg["membrane"]["ny"] = 8
    cfg["fluid"]["nx"] = 28
    cfg["fluid"]["ny"] = 10
    cfg["fluid"]["nz"] = 14
    cfg["time"]["dt"] = 0.01
    cfg["time"]["t_end"] = t_end
    cfg["fsi"]["max_subiters"] = 1          # looser coupling -> faster animation
    cfg.setdefault("simulation", {})["save_interval"] = save_interval
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-end", type=float, default=1.2)
    ap.add_argument("--save-interval", type=int, default=2)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--outdir", type=str, default=str(ROOT / "output"))
    args = ap.parse_args()

    cfg = build_config(args.t_end, args.save_interval)
    sim = FSISimulation(cfg)
    outdir = ensure_dir(args.outdir)

    elements = sim.mesh.elements
    x0 = sim.membrane._x0.copy()
    grid = sim.grid
    j_slice = grid.ny // 2
    y_slice = grid.y[j_slice]

    mem_frames, speed_frames, traces, times = [], [], [], []

    def capture(sim, info, n):
        st = sim.fluid.state
        speed = np.sqrt(st.u ** 2 + st.v ** 2 + st.w ** 2)[:, j_slice, :]
        speed_frames.append(speed.copy())
        nodes = sim.membrane.state.x
        mem_frames.append(nodes.copy())
        tol = 1.5 * (np.ptp(sim.mesh.nodes[:, 1]) / cfg["membrane"]["ny"] + 1e-9)
        near = np.abs(nodes[:, 1] - y_slice) < tol
        traces.append(nodes[near][:, [0, 2]].copy())
        times.append(info["time"])

    print(f"[anim] running FSI: membrane {cfg['membrane']['nx']}x{cfg['membrane']['ny']}, "
          f"fluid {grid.nx}x{grid.ny}x{grid.nz}, t_end={args.t_end}s")
    sim.run(callback=capture)
    print(f"[anim] captured {len(mem_frames)} frames; rendering GIFs ...")

    mem_gif = Path(outdir) / "membrane_flutter.gif"
    flow_gif = Path(outdir) / "flow_past_membrane.gif"
    animate_membrane_3d(mem_frames, elements, x0, mem_gif, times=times, fps=args.fps)
    animate_flow_slice(speed_frames, grid.x, grid.z, traces, flow_gif,
                       times=times, fps=args.fps)
    print(f"[anim] saved:\n  {mem_gif}\n  {flow_gif}")


if __name__ == "__main__":
    main()
