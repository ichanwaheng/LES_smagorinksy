#!/usr/bin/env python3
"""Run the working LES flow-past-sphere demo and write outputs here (including GIF)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from les_sphere_flow import SimConfig, SphereLESSolver  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Working LES sphere-flow runner")
    parser.add_argument("--quick", action="store_true", help="Short demo run")
    parser.add_argument("--steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    args = parser.parse_args()

    mesh = ROOT / "data" / "processed_mesh.npz"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mesh.exists():
        raise SystemExit(f"Missing mesh file: {mesh}")

    if args.quick:
        cfg = SimConfig(dt=1e-4, t_end=0.02, print_every=20, frame_every=10, inlet_ti=0.015)
        max_steps = args.steps or 100
    else:
        cfg = SimConfig(dt=1e-4, t_end=0.2, print_every=100, frame_every=25, inlet_ti=0.015)
        max_steps = args.steps or 800

    print(f"Mesh: {mesh}")
    print(f"Outputs → {out_dir}")

    solver = SphereLESSolver(mesh_path=str(mesh), cfg=cfg)
    solver.run(max_steps=max_steps, capture_gif=not args.no_gif)

    fields = out_dir / "les_sphere_fields.npz"
    solver.save_fields(str(fields))
    mid = solver.plot_midplane(str(out_dir / "flow_past_sphere_midplane.png"))
    streams = solver.plot_streamlines(str(out_dir / "flow_past_sphere_streamlines.png"))
    wake = solver.plot_wake_profile(str(out_dir / "flow_past_sphere_wake.png"))
    nut = solver.plot_nu_t(str(out_dir / "flow_past_sphere_nut.png"))

    paths = [fields, Path(mid), Path(streams), Path(wake), Path(nut)]
    if solver.frames:
        gif = solver.save_gif(str(out_dir / "flow_past_sphere.gif"))
        paths.append(Path(gif))

    print("\nWorking folder outputs:")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
