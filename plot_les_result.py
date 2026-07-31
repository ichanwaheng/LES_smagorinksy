#!/usr/bin/env python
# coding: utf-8
"""Visualise a les_solver result on the mid-plane (z = 2.5) of the sphere channel.

Produces smooth contour + streamline plots of the velocity magnitude and
pressure fields, clearly showing the flow accelerating around the sphere and
the recirculating wake behind it.

    python plot_les_result.py les_result.npz les_flow_past_sphere.png
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

SPHERE_C = (3.0, 2.5)
SPHERE_R = 0.5


def plot_result(npz_path="les_result.npz", out_path="les_flow_past_sphere.png",
                z0=2.5, band=0.35):
    d = np.load(npz_path, allow_pickle=True)
    cc = d["cell_centroids"]
    u = d["velocity"]
    p = d["pressure"]
    nu = float(d["nu"])
    U = float(d["U"])
    Re = U / nu

    sl = np.abs(cc[:, 2] - z0) < band
    x, y = cc[sl, 0], cc[sl, 1]
    ux, uy = u[sl, 0], u[sl, 1]
    umag = np.linalg.norm(u[sl], axis=1)
    ps = p[sl]

    # interpolate the scattered slice onto a regular grid
    xi = np.linspace(1.0, 8.0, 360)
    yi = np.linspace(0.2, 4.8, 240)
    Xi, Yi = np.meshgrid(xi, yi)
    def grid(v):
        return griddata((x, y), v, (Xi, Yi), method="linear")
    Umag = grid(umag); Ux = grid(ux); Uy = grid(uy); Pg = grid(ps)

    # mask the sphere interior
    inside = (Xi - SPHERE_C[0]) ** 2 + (Yi - SPHERE_C[1]) ** 2 < SPHERE_R ** 2
    for arr in (Umag, Ux, Uy, Pg):
        arr[inside] = np.nan

    fig, axs = plt.subplots(2, 1, figsize=(13, 9))

    cf0 = axs[0].contourf(Xi, Yi, Umag, levels=40, cmap="turbo")
    axs[0].streamplot(Xi, Yi, Ux, Uy, color="k", density=1.6, linewidth=0.6,
                      arrowsize=0.7)
    fig.colorbar(cf0, ax=axs[0], label="|u| [m/s]")
    axs[0].set_title(f"LES Smagorinsky - flow past a sphere  (Re = {Re:.0f})\n"
                     f"velocity magnitude + streamlines (recirculating wake behind sphere)")

    cf1 = axs[1].contourf(Xi, Yi, Pg, levels=40, cmap="coolwarm")
    fig.colorbar(cf1, ax=axs[1], label="p [Pa]")
    axs[1].set_title("pressure field (high stagnation pressure upstream, low-pressure wake)")

    for ax in axs:
        ax.add_patch(plt.Circle(SPHERE_C, SPHERE_R, color="0.85", ec="k", zorder=6))
        ax.set_xlim(1, 8); ax.set_ylim(0.2, 4.8); ax.set_aspect("equal")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Re={Re:.0f}  |u| max={umag.max():.3f}  reversed-flow cells (u_x<0)="
          f"{int(np.sum(u[:,0] < -1e-3))}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else "les_result.npz"
    out = sys.argv[2] if len(sys.argv) > 2 else "les_flow_past_sphere.png"
    plot_result(npz, out)
