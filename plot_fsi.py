#!/usr/bin/env python
# coding: utf-8
"""Visualise the FSI result from ``fsi_membrane.py``:
  (a) the form-found vs wind-deflected membrane (3-D),
  (b) the flow field on a mid-height slice past the deflected membrane,
  (c) the FSI deflection-convergence history.

    python plot_fsi.py fsi_result.npz fsi_membrane_wind.png
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata


def plot(path="fsi_result.npz", out="fsi_membrane_wind.png"):
    d = np.load(path, allow_pickle=True)
    c0 = d["mem_coords0"]; c = d["mem_coords"]; tris = d["triangles"]
    cc = d["cell_centroids"]; u = d["velocity"]; p = d["pressure"]
    hist = d["history"]; U = float(d["u_wind"])
    defl = np.linalg.norm(c - c0, axis=1)

    fig = plt.figure(figsize=(16, 5.5))

    # (a) 3-D membrane: reference (grey) vs deflected (colored by deflection)
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.add_collection3d(Poly3DCollection(c0[tris], facecolor="0.8", edgecolor="none",
                                         alpha=0.35))
    polys = Poly3DCollection(c[tris], edgecolor="k", linewidths=0.1)
    polys.set_array(defl[tris].mean(axis=1))
    polys.set_cmap("plasma")
    ax.add_collection3d(polys)
    ax.quiver(3.2, 2.5, 2.5, 0.8, 0, 0, color="blue", lw=2)
    ax.text(3.0, 2.5, 3.2, "wind", color="blue")
    ax.set_xlim(3.5, 7); ax.set_ylim(1, 4); ax.set_zlim(1, 4)
    ax.set_xlabel("x (wind)"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"Membrane: form-found (grey) vs\nwind-deflected (max {defl.max():.2f} m)")
    ax.view_init(elev=18, azim=-70)

    # (b) flow field on z ~ 2.5 slice
    ax2 = fig.add_subplot(1, 3, 2)
    sl = np.abs(cc[:, 2] - 2.5) < 0.25
    x, y = cc[sl, 0], cc[sl, 1]
    umag = np.linalg.norm(u[sl], axis=1)
    xi = np.linspace(1, 9, 320); yi = np.linspace(0.2, 4.8, 200)
    Xi, Yi = np.meshgrid(xi, yi)
    Um = griddata((x, y), umag, (Xi, Yi), method="linear")
    cf = ax2.contourf(Xi, Yi, Um, levels=40, cmap="turbo")
    fig.colorbar(cf, ax=ax2, label="|u| [m/s]")
    # membrane trace near this slice
    ms = np.abs(c[:, 2] - 2.5) < 0.25
    ax2.scatter(c[ms, 0], c[ms, 1], s=8, c="white", edgecolors="k", linewidths=0.3)
    ax2.set_title("Flow past deflected membrane (z=2.5 slice)")
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]"); ax2.set_aspect("equal")
    ax2.set_xlim(1, 9); ax2.set_ylim(0.2, 4.8)

    # (c) convergence
    ax3 = fig.add_subplot(1, 3, 3)
    it = hist[:, 0]
    ax3.plot(it, hist[:, 3], "-o", label="max deflection [m]")
    ax3.plot(it, hist[:, 4], "-s", label="update / FSI iter [m]")
    ax3.set_xlabel("FSI iteration"); ax3.set_title("Two-way FSI convergence")
    ax3.grid(True, alpha=0.3); ax3.legend()

    fig.suptitle("FSI: wind load (LES) on a form-found tensile membrane (UWM)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"max deflection = {defl.max():.3f} m   net Fx (last) = {hist[-1,2]:.3f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "fsi_result.npz"
    o = sys.argv[2] if len(sys.argv) > 2 else "fsi_membrane_wind.png"
    plot(p, o)
