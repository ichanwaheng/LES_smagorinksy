#!/usr/bin/env python3
"""Entry point: quasi-static UWM ↔ PISO/LES membrane FSI."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(QS))

from src.utils.io import ensure_dir, load_config, save_snapshot, write_membrane_vtk
from src.utils.viz import plot_history, plot_membrane_3d, plot_membrane_and_slice

from coupling import QuasiStaticFSI
from excel_export import DeformationRecorder


def parse_args():
    p = argparse.ArgumentParser(
        description="Quasi-static FSI: Updated Weight Method + PISO/LES"
    )
    p.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(QS / "config" / "quasi_static.yaml"),
        help="Path to YAML config",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Coarse mesh / few iterations smoke run",
    )
    p.add_argument(
        "--excel",
        type=str,
        default=None,
        help="Output Excel path (default: <output_dir>/membrane_deformations.xlsx)",
    )
    return p.parse_args()


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
        cfg["quasi_static"]["max_iters"] = 3
        cfg["quasi_static"]["fluid_substeps"] = 5
        cfg["quasi_static"]["load_scale"] = 0.15
        cfg["les"]["enabled"] = True
        cfg["simulation"]["save_interval"] = 1

    out_rel = cfg["simulation"].get("output_dir", "quasi_static/output")
    out = ensure_dir(ROOT / out_rel)
    print(f"[QS-FSI] output → {out}")
    print(
        f"[QS-FSI] UWM membrane {cfg['membrane']['nx']}x{cfg['membrane']['ny']}, "
        f"PISO+LES fluid {cfg['fluid']['nx']}x{cfg['fluid']['ny']}x{cfg['fluid']['nz']}, "
        f"max_iters={cfg['quasi_static']['max_iters']}"
    )

    sim = QuasiStaticFSI(cfg)
    z0 = float(cfg["fluid"]["membrane_z0"])
    nodes_flat = sim.x_bc.copy()
    nodes_flat[:, 2] = z0
    recorder = DeformationRecorder(reference_nodes=nodes_flat, fixed=sim.mesh.fixed)

    def on_iter(simulation, info, k):
        print(
            f"  iter={int(info['iteration']):02d}  "
            f"disp={info['max_disp']:.4e} m  "
            f"shape_res={info['shape_residual']:.2e}  "
            f"uwm_res={info['uwm_residual']:.2e}  "
            f"|p|_max={info['pressure_max']:.2f} Pa  "
            f"CFL={info['cfl']:.3f}"
        )
        recorder.record(
            time=float(info.get("time", info["iteration"])),
            nodes=simulation.nodes,
            iteration=int(info["iteration"]),
        )
        save_every = int(cfg["simulation"].get("save_interval", 1))
        if k % save_every == 0:
            save_snapshot(
                out,
                step=int(info["iteration"]),
                time=float(info["iteration"]),
                membrane_nodes=simulation.nodes,
                membrane_elements=simulation.mesh.elements,
                fluid_u=simulation.fluid.state.u,
                fluid_v=simulation.fluid.state.v,
                fluid_w=simulation.fluid.state.w,
                fluid_p=simulation.fluid.state.p,
                meta={
                    "shape_residual": info["shape_residual"],
                    "max_disp": info["max_disp"],
                },
            )
            write_membrane_vtk(
                out / f"membrane_{int(info['iteration']):06d}.vtk",
                simulation.nodes,
                simulation.mesh.elements,
            )

    hist = sim.run(callback=on_iter)

    # history CSV
    csv_path = out / "history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "iteration",
                "max_disp",
                "shape_residual",
                "uwm_residual",
                "pressure_max",
                "cfl",
            ]
        )
        for i in range(len(hist.iteration)):
            w.writerow(
                [
                    hist.iteration[i],
                    hist.max_disp[i],
                    hist.shape_residual[i],
                    hist.uwm_residual[i],
                    hist.pressure_max[i],
                    hist.cfl[i],
                ]
            )

    # reuse plot_history with synthetic time = iteration index
    class _H:
        pass

    h = _H()
    h.time = [float(v) for v in hist.iteration]
    h.max_disp = hist.max_disp
    h.kinetic = hist.pressure_max  # show |p|_max in the KE panel slot
    h.cfl = hist.cfl
    h.residual = hist.shape_residual
    if cfg["simulation"].get("plot", True):
        plot_history(h, out / "history.png")
        plot_membrane_3d(
            sim.nodes,
            sim.mesh.elements,
            out / "final_membrane.png",
            title="Quasi-static UWM form under fluid load",
        )
        j = sim.grid.ny // 2
        plot_membrane_and_slice(
            sim.nodes,
            sim.mesh.elements,
            sim.fluid.state.u,
            sim.grid.x,
            sim.grid.z,
            j,
            out / "final_slice.png",
            title="Quasi-static UWM + PISO/LES",
        )

    print(f"[QS-FSI] done — {len(hist.iteration)} outer iterations → {out}")
    excel_path = Path(args.excel) if args.excel else out / "membrane_deformations.xlsx"
    xlsx = recorder.write_xlsx(
        excel_path, per_step_sheets=len(recorder.times) <= 40
    )
    print(f"[QS-FSI] deformations Excel → {xlsx}")


if __name__ == "__main__":
    main()
