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


def animate_membrane_3d(
    frames_nodes,
    elements: np.ndarray,
    x0: np.ndarray,
    out_path: str | Path,
    times=None,
    fps: int = 12,
    title: str = "Tensile membrane flutter",
) -> None:
    """Animated 3-D GIF of the membrane moving/fluttering over time.

    ``frames_nodes`` is a list of (n_nodes, 3) node-position arrays; the surface
    is coloured by displacement magnitude from the reference shape ``x0``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    allc = np.vstack(frames_nodes)
    lo = allc.min(axis=0) - 0.05
    hi = allc.max(axis=0) + 0.05
    dmax = max(1e-6, max(np.linalg.norm(f - x0, axis=1).max() for f in frames_nodes))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw(k):
        ax.clear()
        c = frames_nodes[k]
        d = np.linalg.norm(c - x0, axis=1)
        coll = Poly3DCollection(c[elements], edgecolor="k", linewidths=0.15, alpha=0.9)
        coll.set_array(d[elements].mean(axis=1))
        coll.set_cmap("plasma")
        coll.set_clim(0, dmax)
        ax.add_collection3d(coll)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        t = f"  t={times[k]:.3f}s" if times is not None else f"  frame {k}"
        ax.set_title(title + t)
        ax.view_init(elev=22, azim=-60)

    anim = FuncAnimation(fig, draw, frames=len(frames_nodes), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def animate_flow_slice(
    frames_speed,
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    membrane_traces,
    out_path: str | Path,
    times=None,
    fps: int = 12,
    title: str = "Flow past tensile membrane",
) -> None:
    """Animated GIF of the mid-plane speed field with the membrane cross-section.

    ``frames_speed`` is a list of (nx, nz) speed slices; ``membrane_traces`` is a
    list of (m, 2) arrays of membrane (x, z) points near the slice."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vmax = float(np.clip(np.nanpercentile(frames_speed[-1], 99.0), 1e-6, None))
    extent = [grid_x[0], grid_x[-1], grid_z[0], grid_z[-1]]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(frames_speed[0].T, origin="lower", extent=extent, cmap="turbo",
                   vmin=0, vmax=vmax, aspect="equal", interpolation="bilinear")
    fig.colorbar(im, ax=ax, label="|U| [m/s]")
    tr0 = membrane_traces[0]
    (line,) = ax.plot(tr0[:, 0], tr0[:, 1], "w.", ms=4, mec="k", mew=0.4)
    ax.set_xlabel("x [m] (wind ->)"); ax.set_ylabel("z [m]")
    ttl = ax.set_title(title)

    def draw(k):
        im.set_data(frames_speed[k].T)
        tr = membrane_traces[k]
        line.set_data(tr[:, 0], tr[:, 1])
        t = f"  t={times[k]:.3f}s" if times is not None else f"  frame {k}"
        ttl.set_text(title + t)
        return [im, line, ttl]

    anim = FuncAnimation(fig, draw, frames=len(frames_speed), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
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
