"""Visualization helpers (matplotlib)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def plot_membrane_and_slice(
    nodes: np.ndarray,
    elements: np.ndarray,
    fluid_u: np.ndarray,
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    j_slice: int,
    out_path: str | Path,
    title: str = "Membrane FSI",
) -> None:
    """Save a 2D plot: membrane side-view + mid-plane |U| contour."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    speed = np.sqrt(
        fluid_u[:, j_slice, :] ** 2
        + 0.0  # v not shown
    )
    # full speed if only u passed — recompute properly if 3 components unavailable
    # here fluid_u is u-component; use abs(u) as proxy for streamwise field
    field = np.abs(fluid_u[:, j_slice, :])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    Xg, Zg = np.meshgrid(grid_x, grid_z, indexing="ij")
    cf = ax.contourf(Xg, Zg, field, levels=24, cmap="YlOrBr")
    fig.colorbar(cf, ax=ax, label="|u| [m/s]")

    # membrane projected to xz (average y or all nodes)
    tri = Triangulation(nodes[:, 0], nodes[:, 2], elements)
    ax.triplot(tri, color="#1a1a1a", lw=0.6, alpha=0.85)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_history(history, out_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    t = history.time
    axes[0, 0].plot(t, history.max_disp, color="#0b3d2e")
    axes[0, 0].set_ylabel("max |u| [m]")
    axes[0, 0].set_title("Membrane displacement")
    axes[0, 1].plot(t, history.kinetic, color="#8b3a2a")
    axes[0, 1].set_ylabel("KE [J]")
    axes[0, 1].set_title("Kinetic energy")
    axes[1, 0].plot(t, history.cfl, color="#1f4e79")
    axes[1, 0].set_ylabel("CFL")
    axes[1, 0].set_title("Fluid CFL")
    axes[1, 1].plot(t, history.residual, color="#5c4a1f")
    axes[1, 1].set_ylabel("residual")
    axes[1, 1].set_title("FSI sub-iter residual")
    for ax in axes.ravel():
        ax.set_xlabel("t [s]")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_membrane_3d(
    nodes: np.ndarray,
    elements: np.ndarray,
    out_path: str | Path,
    displacement: Optional[np.ndarray] = None,
    title: str = "Tensile membrane",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    tris = nodes[elements]
    coll = Poly3DCollection(tris, alpha=0.85, edgecolor="#222222", linewidths=0.2)
    if displacement is not None:
        mag = np.linalg.norm(displacement, axis=1)
        # color by average nodal magnitude per triangle
        face_mag = mag[elements].mean(axis=1)
        coll.set_array(face_mag)
        coll.set_cmap("cividis")
    else:
        coll.set_facecolor("#c4a574")
    ax.add_collection3d(coll)
    ax.auto_scale_xyz(nodes[:, 0], nodes[:, 1], nodes[:, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
