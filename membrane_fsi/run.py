#!/usr/bin/env python3
"""Build mesh (if needed) and run the flexible-membrane LES–FSI demo."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ensure_mesh(coarse: bool = True) -> tuple[Path, Path]:
    msh = ROOT / "data" / "fluid_mesh_membrane.msh"
    meta = ROOT / "data" / "fluid_mesh_membrane.npz"
    npz = ROOT / "data" / "processed_mesh_membrane.npz"
    if not msh.exists():
        cmd = [sys.executable, str(ROOT / "mesh" / "generate_membrane_mesh.py"), "-o", str(msh)]
        if coarse:
            cmd.append("--coarse")
        print("Generating membrane mesh...")
        subprocess.check_call(cmd)
    if not npz.exists():
        print("Processing mesh → npz...")
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "mesh" / "process_mesh.py"),
                "--msh",
                str(msh),
                "-o",
                str(npz),
            ]
        )
    return npz, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Run flexible membrane LES–FSI")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--skip-mesh", action="store_true")
    args = ap.parse_args()

    if not args.skip_mesh:
        npz, meta = ensure_mesh(coarse=True)
    else:
        npz = ROOT / "data" / "processed_mesh_membrane.npz"
        meta = ROOT / "data" / "fluid_mesh_membrane.npz"

    cmd = [
        sys.executable,
        str(ROOT / "les_membrane_fsi.py"),
        "--mesh",
        str(npz),
        "--geom",
        str(meta),
        "--outdir",
        str(ROOT / "outputs"),
    ]
    if args.quick:
        cmd.append("--quick")
    if args.steps is not None:
        cmd += ["--steps", str(args.steps)]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
