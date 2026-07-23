#!/usr/bin/env python
# coding: utf-8
"""Render an animated GIF of the fluid flow developing (LES solver).

Captures the mid-plane velocity field at successive SIMPLE iterations and
animates it. Works for the flow past the sphere (default) or any mesh.

    python animate_les.py                       # flow past sphere -> les_flow.gif
    python animate_les.py --nu 0.01 --out les_flow.gif
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import griddata

import les_solver


def make_gif(nu=0.01, U=1.0, iters=180, snap_every=3, beta=0.9, limiter="none",
             mesh_path="processed_mesh.npz", sphere=(3.0, 2.5, 0.5),
             z0=2.5, out="les_flow.gif", fps=12):
    frames = []

    def cb(it, u, p):
        frames.append((it, np.linalg.norm(u, axis=1)))

    print("Running solver and capturing frames ...")
    _, _, mesh, _, _ = les_solver.run(
        nu=nu, U=U, iters=iters, beta=beta, limiter=limiter,
        alpha_u=0.4, alpha_p=0.25, avg_last=1, tol=0.0,
        mesh_path=mesh_path, out_path="/tmp/_anim.npz", log_every=iters,
        callback=cb, snap_every=snap_every)

    cc = mesh["cell_centroids"]
    sl = np.abs(cc[:, 2] - z0) < 0.25
    x, y = cc[sl, 0], cc[sl, 1]
    xi = np.linspace(1, 8, 260); yi = np.linspace(0.2, 4.8, 170)
    Xi, Yi = np.meshgrid(xi, yi)
    inside = None
    if sphere is not None:
        cx, cy, r = sphere
        inside = (Xi - cx) ** 2 + (Yi - cy) ** 2 < r ** 2

    grids = []
    for it, umag in frames:
        G = griddata((x, y), umag[sl], (Xi, Yi), method="linear")
        if inside is not None:
            G[inside] = np.nan
        grids.append((it, G))
    vmax = max(1.2, np.nanmax([np.nanmax(g) for _, g in grids]))

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("0.85")                       # sphere interior (NaN) shown grey
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(grids[0][1], origin="lower", extent=[xi[0], xi[-1], yi[0], yi[-1]],
                   cmap=cmap, vmin=0, vmax=vmax, aspect="equal", interpolation="bilinear")
    fig.colorbar(im, ax=ax, label="|u| [m/s]")
    if sphere is not None:
        ax.add_patch(plt.Circle(sphere[:2], sphere[2], color="0.85", ec="k", zorder=6))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    title = ax.set_title("")

    def update(k):
        it, G = grids[k]
        im.set_data(G)
        title.set_text(f"LES flow past sphere (Re={U/nu:.0f}) - iteration {it}")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(grids), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved {out}  ({len(grids)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--U", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=180)
    ap.add_argument("--snap_every", type=int, default=3)
    ap.add_argument("--out", type=str, default="les_flow.gif")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    make_gif(nu=args.nu, U=args.U, iters=args.iters, snap_every=args.snap_every,
             out=args.out, fps=args.fps)
