#!/usr/bin/env python
# coding: utf-8
"""Render an animated GIF of the fluid flow past the tensile membrane structure.

Form-finds the membrane, deflects it to its wind equilibrium (a few FSI cycles),
then captures the flow developing past the deflected membrane and animates the
mid-plane velocity field with the membrane cross-section overlaid.

    python animate_fsi.py --wind_axis 0 --out fsi_flow.gif
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import griddata

import les_solver
import fsi_membrane as fsi


def make_gif(wind_axis=0, nu=0.02, u_wind=1.0, load_scale=1.0, relax=0.5,
             n_pre=4, pre_iters=120, anim_iters=150, snap_every=3,
             out="fsi_flow.gif", fps=12):
    flow = fsi.ChannelFlow()
    mem = fsi.form_find_membrane(wind_axis=wind_axis)

    # deflect the membrane to its wind equilibrium (partitioned FSI cycles)
    print("Deflecting membrane to wind equilibrium ...")
    for k in range(n_pre):
        blocked = flow.blocked_mask(mem)
        _, p, _, g, _ = flow.solve(blocked, nu=nu, u_wind=u_wind,
                                   iters=pre_iters, wind_axis=wind_axis)
        nodal_load, _ = fsi.transfer_load(g, p, mem, load_scale)
        fsi.deflect_membrane(mem, nodal_load, relax=relax)
    defl = np.linalg.norm(mem["coords"] - mem["coords0"], axis=1).max()
    print(f"  deflected {defl:.3f} m")

    # capture the flow developing past the deflected membrane
    blocked = flow.blocked_mask(mem)
    frames = []
    flow.solve(blocked, nu=nu, u_wind=u_wind, iters=anim_iters, wind_axis=wind_axis,
               callback=lambda it, u, p: frames.append((it, np.linalg.norm(u, axis=1))),
               snap_every=snap_every)

    w = wind_axis
    a, b = fsi.span_axes(w)
    names = "xyz"
    cc = flow.mesh["cell_centroids"]
    mid_b = fsi.DOMAIN[b] * 0.5
    sl = np.abs(cc[:, b] - mid_b) < 0.25
    hx, vy = cc[sl, w], cc[sl, a]
    xi = np.linspace(0.3, fsi.DOMAIN[w] - 0.3, 300)
    yi = np.linspace(0.2, fsi.DOMAIN[a] - 0.2, 190)
    Xi, Yi = np.meshgrid(xi, yi)
    grids = [(it, griddata((hx, vy), um[sl], (Xi, Yi), method="linear"))
             for it, um in frames]
    # robust colour scale (ignore localised edge jets from the staircase baffle)
    vmax = float(np.clip(np.nanpercentile(grids[-1][1], 99), 1.2, 2.0))

    ms = np.abs(mem["coords"][:, b] - mid_b) < 0.35
    mem_w, mem_a = mem["coords"][ms, w], mem["coords"][ms, a]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    im = ax.imshow(grids[0][1], origin="lower", extent=[xi[0], xi[-1], yi[0], yi[-1]],
                   cmap="turbo", vmin=0, vmax=vmax, aspect="equal",
                   interpolation="bilinear")
    fig.colorbar(im, ax=ax, label="|u| [m/s]")
    ax.scatter(mem_w, mem_a, s=10, c="white", edgecolors="k", linewidths=0.4, zorder=5)
    ax.set_xlabel(f"{names[w]} [m] (wind ->)"); ax.set_ylabel(f"{names[a]} [m]")
    title = ax.set_title("")

    def update(k):
        it, G = grids[k]
        im.set_data(G)
        title.set_text(f"Flow past tensile membrane (Re={u_wind/nu:.0f}, "
                       f"wind {names[w]}) - iteration {it}")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(grids), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved {out}  ({len(grids)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wind_axis", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--nu", type=float, default=0.02)
    ap.add_argument("--u_wind", type=float, default=1.0)
    ap.add_argument("--load_scale", type=float, default=1.0)
    ap.add_argument("--n_pre", type=int, default=4)
    ap.add_argument("--anim_iters", type=int, default=150)
    ap.add_argument("--snap_every", type=int, default=3)
    ap.add_argument("--out", type=str, default="fsi_flow.gif")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    make_gif(wind_axis=args.wind_axis, nu=args.nu, u_wind=args.u_wind,
             load_scale=args.load_scale, n_pre=args.n_pre,
             anim_iters=args.anim_iters, snap_every=args.snap_every,
             out=args.out, fps=args.fps)
