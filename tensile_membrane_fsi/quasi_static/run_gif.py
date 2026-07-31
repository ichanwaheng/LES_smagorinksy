#!/usr/bin/env python3
"""Quasi-static UWM + PISO/LES over a time interval → animated GIFs.

Advances the fluid in time; at each sample the membrane form is updated
with the Updated Weight Method under the current pressure, then frames
are rendered (3D membrane + mid-plane |u| slice).

Writes **two** GIFs by default:
  - ``membrane_quasi_static_original.gif`` — physical (unamplified) deflection
  - ``membrane_quasi_static_amplified.gif`` — visually scaled deflection

    python quasi_static/run_gif.py --quick
    python quasi_static/run_gif.py --t-end 1.0 --fps 8
    python quasi_static/run_gif.py --out-dir quasi_static/output
"""

from __future__ import annotations

import argparse
import csv
import sys
import time as walltime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(QS))

from src.utils.io import ensure_dir, load_config, save_snapshot, write_membrane_vtk
from src.utils.viz import (
    plot_history,
    plot_membrane_3d,
    plot_membrane_and_slice,
    render_flutter_frame,
    save_gif,
)

from coupling import QuasiStaticFSI
from excel_export import DeformationRecorder


def parse_args():
    p = argparse.ArgumentParser(
        description="Quasi-static membrane FSI → original + amplified GIFs"
    )
    p.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(QS / "config" / "quasi_static.yaml"),
    )
    p.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="Physical end time [s] (default: from config time.t_end)",
    )
    p.add_argument("--fps", type=int, default=8, help="GIF frames per second")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for GIF outputs (default: config simulation.output_dir)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Deprecated alias: if set, amplified GIF path; original is "
        "written beside it as *_original.gif",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Coarse mesh, short interval smoke GIF",
    )
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=5,
        help="Frames between NPZ/VTK writes (0 disables)",
    )
    p.add_argument(
        "--disp-scale",
        type=float,
        default=None,
        help="Visual amplification of out-of-plane deflection vs the flat "
        "mounting plane (physics unchanged). Default: auto-scale so the "
        "peak |Δz| fills ~40%% of the membrane span, or config disp_scale.",
    )
    p.add_argument(
        "--target-amp",
        type=float,
        default=None,
        help="Target visual peak |Δz| [m] for auto scaling (default: 0.4×span)",
    )
    p.add_argument(
        "--excel",
        type=str,
        default=None,
        help="Output Excel path for nodal x,y,z at each time step "
        "(default: <output_dir>/membrane_deformations.xlsx)",
    )
    p.add_argument(
        "--no-excel",
        action="store_true",
        help="Skip writing the deformations Excel workbook",
    )
    return p.parse_args()


def _amplify_z(
    nodes: np.ndarray,
    z_ref: float,
    scale: float,
) -> np.ndarray:
    """Amplify out-of-plane deflection about the flat mounting plane (GIF only)."""
    out = np.asarray(nodes, dtype=float).copy()
    out[:, 2] = float(z_ref) + float(scale) * (out[:, 2] - float(z_ref))
    return out


def _auto_disp_scale(nodes: np.ndarray, z_ref: float, target_amp: float) -> float:
    phys = float(np.max(np.abs(nodes[:, 2] - z_ref)))
    return float(target_amp) / max(phys, 1e-9)


def _render_frame(
    nodes_plot: np.ndarray,
    elements: np.ndarray,
    nodes_flat: np.ndarray,
    speed: np.ndarray,
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    time: float,
    mesh_nx: int,
    mesh_ny: int,
    speed_max: float,
    disp_max: float,
    z_limits: tuple,
    title: str,
):
    return render_flutter_frame(
        nodes_plot,
        elements,
        nodes_flat,
        speed,
        grid_x,
        grid_z,
        time,
        mesh_nx,
        mesh_ny,
        speed_max,
        disp_max,
        z_limits,
        title=title,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.quick:
        cfg["membrane"]["nx"] = 8
        cfg["membrane"]["ny"] = 6
        cfg["fluid"]["nx"] = 16
        cfg["fluid"]["ny"] = 8
        cfg["fluid"]["nz"] = 8
        cfg["fluid"]["nu"] = 5.0e-3
        cfg["time"]["dt"] = 0.01
        cfg["time"]["t_end"] = 0.4
        cfg["quasi_static"]["fluid_substeps"] = 4
        cfg["quasi_static"]["max_iters"] = 40
        cfg["quasi_static"]["load_scale"] = 1.5
        cfg["quasi_static"]["shape_tol"] = 0.0
        cfg["les"]["enabled"] = True

    if args.t_end is not None:
        cfg["time"]["t_end"] = args.t_end

    cfg.setdefault("quasi_static", {})
    cfg["quasi_static"]["shape_tol"] = 0.0

    out = ensure_dir(
        Path(args.out_dir)
        if args.out_dir
        else ROOT / cfg["simulation"].get("output_dir", "quasi_static/output")
    )
    if args.out:
        amp_gif_path = Path(args.out)
        orig_gif_path = amp_gif_path.with_name(
            amp_gif_path.stem + "_original" + amp_gif_path.suffix
        )
    else:
        orig_gif_path = out / "membrane_quasi_static_original.gif"
        amp_gif_path = out / "membrane_quasi_static_amplified.gif"

    sim = QuasiStaticFSI(cfg)
    z0 = float(cfg["fluid"]["membrane_z0"])
    nodes_flat = sim.x_bc.copy()
    nodes_flat[:, 2] = z0
    mesh_nx, mesh_ny = sim.mesh.nx, sim.mesh.ny
    j_slice = sim.grid.ny // 2
    span = float(cfg["membrane"]["length"])
    target_amp = float(
        args.target_amp
        if args.target_amp is not None
        else cfg.get("quasi_static", {}).get("target_amp", 0.4 * span)
    )

    if args.disp_scale is not None:
        disp_scale = float(args.disp_scale)
    elif cfg.get("quasi_static", {}).get("disp_scale", None) not in (None, "auto"):
        disp_scale = float(cfg["quasi_static"]["disp_scale"])
    else:
        disp_scale = _auto_disp_scale(sim.nodes, z0, target_amp)

    U = float(cfg["fluid"]["U_inlet"])
    speed_max = 1.6 * U
    # Physical GIF: tight window around the true membrane motion
    phys_disp_max = max(
        0.02,
        1.5 * float(np.max(np.abs(sim.nodes[:, 2] - z0))) + 1e-3,
    )
    phys_z_span = max(1.5 * phys_disp_max, 0.08)
    phys_z_limits = (z0 - phys_z_span, z0 + phys_z_span)

    # Amplified GIF scales
    amp_disp_max = max(0.15, 1.15 * target_amp)
    amp_z_span = max(1.25 * amp_disp_max, 0.35)
    amp_z_limits = (z0 - amp_z_span, z0 + amp_z_span)

    frames_orig = []
    frames_amp = []
    t0 = walltime.time()
    t_end = float(cfg["time"]["t_end"])
    recorder = DeformationRecorder(reference_nodes=nodes_flat, fixed=sim.mesh.fixed)

    def on_frame(simulation, info, k):
        nonlocal disp_scale, phys_disp_max, phys_z_limits, amp_disp_max, amp_z_limits
        st = simulation.fluid.state
        speed = np.sqrt(
            st.u[:, j_slice, :] ** 2
            + st.v[:, j_slice, :] ** 2
            + st.w[:, j_slice, :] ** 2
        )
        if args.disp_scale is None and cfg.get("quasi_static", {}).get(
            "disp_scale", "auto"
        ) in (None, "auto"):
            disp_scale = _auto_disp_scale(simulation.nodes, z0, target_amp)

        phys_peak = float(np.max(np.abs(simulation.nodes[:, 2] - z0)))
        phys_disp_max = max(0.02, 1.5 * phys_peak + 1e-3, phys_disp_max)
        phys_z_span_now = max(1.5 * phys_disp_max, 0.08)
        phys_z_limits = (z0 - phys_z_span_now, z0 + phys_z_span_now)

        nodes_amp = _amplify_z(simulation.nodes, z0, disp_scale)
        vis_peak = float(np.max(np.abs(nodes_amp[:, 2] - z0)))
        amp_disp_max = max(0.15, 1.15 * vis_peak, 1.15 * target_amp)
        amp_z_span_now = max(1.25 * amp_disp_max, 0.35)
        amp_z_limits = (z0 - amp_z_span_now, z0 + amp_z_span_now)

        recorder.record(
            time=float(info["time"]),
            nodes=simulation.nodes,
            iteration=int(info["iteration"]),
        )

        frames_orig.append(
            _render_frame(
                simulation.nodes,
                simulation.mesh.elements,
                nodes_flat,
                speed,
                simulation.grid.x,
                simulation.grid.z,
                info["time"],
                mesh_nx,
                mesh_ny,
                speed_max,
                phys_disp_max,
                phys_z_limits,
                title="Quasi-static UWM membrane  (original Δz)",
            )
        )
        frames_amp.append(
            _render_frame(
                nodes_amp,
                simulation.mesh.elements,
                nodes_flat,
                speed,
                simulation.grid.x,
                simulation.grid.z,
                info["time"],
                mesh_nx,
                mesh_ny,
                speed_max,
                amp_disp_max,
                amp_z_limits,
                title=f"Quasi-static UWM membrane  (×{disp_scale:.0f} Δz vs flat)",
            )
        )
        print(
            f"  frame {len(frames_orig):3d}  t={info['time']:6.3f}s  "
            f"Δz_phys={phys_peak:.4e} m  "
            f"×{disp_scale:.0f} → {vis_peak:.3f} m  "
            f"|p|_max={info['pressure_max']:.2f} Pa  "
            f"[{walltime.time() - t0:6.1f}s wall]"
        )
        if args.snapshot_every > 0 and (len(frames_orig) - 1) % args.snapshot_every == 0:
            save_snapshot(
                out,
                step=int(info["iteration"]),
                time=float(info["time"]),
                membrane_nodes=simulation.nodes,
                membrane_elements=simulation.mesh.elements,
                fluid_u=st.u,
                fluid_v=st.v,
                fluid_w=st.w,
                fluid_p=st.p,
                meta=info,
            )
            write_membrane_vtk(
                out / f"membrane_{int(info['iteration']):06d}.vtk",
                simulation.nodes,
                simulation.mesh.elements,
            )
            plot_membrane_and_slice(
                simulation.nodes,
                simulation.mesh.elements,
                st.u,
                simulation.grid.x,
                simulation.grid.z,
                j_slice,
                out / f"slice_{int(info['iteration']):06d}.png",
                title=f"Quasi-static UWM  t={info['time']:.3f}s",
            )

    print(
        f"[QS-GIF] fluid {sim.grid.nx}x{sim.grid.ny}x{sim.grid.nz}, "
        f"membrane {mesh_nx}x{mesh_ny}, dt={sim.dt}, "
        f"substeps={sim.fluid_substeps}, t_end={t_end}"
    )
    hist = sim.run_timed(t_end=t_end, callback=on_frame)

    csv_path = out / "history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "iteration",
                "time",
                "max_disp",
                "shape_residual",
                "uwm_residual",
                "pressure_max",
                "cfl",
            ]
        )
        dt_block = sim.fluid_substeps * sim.dt
        for i in range(len(hist.iteration)):
            w.writerow(
                [
                    hist.iteration[i],
                    hist.iteration[i] * dt_block,
                    hist.max_disp[i],
                    hist.shape_residual[i],
                    hist.uwm_residual[i],
                    hist.pressure_max[i],
                    hist.cfl[i],
                ]
            )

    class _H:
        pass

    h = _H()
    h.time = [i * dt_block for i in hist.iteration]
    h.max_disp = hist.max_disp
    h.kinetic = hist.pressure_max
    h.cfl = hist.cfl
    h.residual = hist.shape_residual
    plot_history(h, out / "history.png")
    plot_membrane_3d(
        sim.nodes,
        sim.mesh.elements,
        out / "membrane_final.png",
        displacement=sim.nodes - nodes_flat,
        title="Final quasi-static UWM form",
    )

    if not frames_orig:
        raise SystemExit("no frames recorded — check t_end / fluid_substeps")
    save_gif(frames_orig, orig_gif_path, fps=args.fps)
    save_gif(frames_amp, amp_gif_path, fps=args.fps)
    print(f"[QS-GIF] {len(frames_orig)} frames → {orig_gif_path}  (original)")
    print(f"[QS-GIF] {len(frames_amp)} frames → {amp_gif_path}  (amplified)")

    # keep legacy filename as a copy of the amplified GIF for older docs/links
    legacy = out / "membrane_quasi_static.gif"
    if amp_gif_path.resolve() != legacy.resolve():
        try:
            legacy.write_bytes(amp_gif_path.read_bytes())
        except OSError:
            pass

    if not args.no_excel:
        excel_path = (
            Path(args.excel)
            if args.excel
            else out / "membrane_deformations.xlsx"
        )
        per_step = len(recorder.times) <= 40
        xlsx = recorder.write_xlsx(excel_path, per_step_sheets=per_step)
        print(
            f"[QS-GIF] deformations Excel ({len(recorder.times)} steps × "
            f"{sim.mesh.n_nodes} nodes) → {xlsx}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
