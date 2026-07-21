#!/usr/bin/env python3
"""
LES (Smagorinsky) finite-volume simulation of turbulent flow past a sphere.

Boundary conditions chosen so inlet/outlet do not disturb the sphere wake:
  - Inlet  : fixed freestream U_inf + mild turbulence intensity (Dirichlet)
  - Outlet : convective Orlanski outflow for velocity + soft pressure
             (non-reflecting exit; wake structures leave without bouncing back)
  - Walls  : free-slip on outer box (avoids strong channel blockage)
  - Object : no-slip on the sphere
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image
from scipy import sparse
from scipy.interpolate import griddata


INLET, OUTLET, WALL, OBJECT = 1, 2, 3, 4


@dataclass
class SimConfig:
    rho: float = 1.0
    nu: float = 1.0e-3
    Cs: float = 0.1
    U_inf: float = 1.0
    dt: float = 2.0e-4
    t_end: float = 0.5
    inlet_ti: float = 0.01
    seed: int = 7
    print_every: int = 50
    frame_every: int = 20  # capture mid-plane frame every N steps for GIF
    urf_u: float = 0.5
    urf_p: float = 0.3
    L: float = 10.0
    W: float = 5.0
    H: float = 5.0
    sphere_c: tuple[float, float, float] = (3.0, 2.5, 2.5)
    sphere_r: float = 0.5
    gif_duration_ms: int = 120
    gif_vmax: float = 1.6  # fixed color scale for smooth animation



class SphereLESSolver:
    def __init__(self, mesh_path: str = "processed_mesh.npz", cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        data = np.load(mesh_path)

        self.points = data["points"]
        self.tetra = data["tetra"]
        self.xc = data["cell_centroids"].astype(np.float64)
        self.V = data["cell_volumes"].astype(np.float64)
        self.owner = data["owner"].astype(np.int64)
        self.neigh = data["neighbour"].astype(np.int64)
        self.xf = data["face_centroids"].astype(np.float64)
        self.Sf = data["face_area_vectors"].astype(np.float64)
        self.magSf = data["face_areas"].astype(np.float64)
        self.nf = data["face_normals"].astype(np.float64)
        self.tag = data["boundary_tag"].astype(np.int32)

        self.n_c = len(self.V)
        self.n_f = len(self.owner)

        self.inlet = np.where(self.tag == INLET)[0]
        self.outlet = np.where(self.tag == OUTLET)[0]
        self.walls = np.where(self.tag == WALL)[0]
        self.object = np.where(self.tag == OBJECT)[0]
        self.internal = np.where(self.neigh >= 0)[0]
        self.boundary = np.where(self.neigh < 0)[0]

        self.d_owner = self.xf - self.xc[self.owner]
        neigh_safe = np.where(self.neigh >= 0, self.neigh, 0)
        self.d_neigh = self.xf - self.xc[neigh_safe]

        P = self.owner[self.internal]
        N = self.neigh[self.internal]
        dPN = self.xc[N] - self.xc[P]
        # a = |Sf| / |d·n|  (stable two-point coefficient)
        dn = np.einsum("ij,ij->i", dPN, self.nf[self.internal])
        self.a_int = self.magSf[self.internal] / np.maximum(np.abs(dn), 1e-6)

        delta_b = np.abs(
            np.einsum("ij,ij->i", self.d_owner[self.boundary], self.nf[self.boundary])
        )
        self.a_bnd = self.magSf[self.boundary] / np.maximum(delta_b, 1e-6)

        dP = np.linalg.norm(self.d_owner[self.internal], axis=1) + 1e-30
        dN = np.linalg.norm(self.d_neigh[self.internal], axis=1) + 1e-30
        self.wN = dP / (dP + dN)
        self.wP = 1.0 - self.wN

        self.rng = np.random.default_rng(self.cfg.seed)
        self.U = np.zeros((self.n_c, 3), dtype=np.float64)
        self.U[:, 0] = self.cfg.U_inf
        self.p = np.zeros(self.n_c, dtype=np.float64)
        self.nu_t = np.zeros(self.n_c, dtype=np.float64)
        self.Uf_out = np.tile(np.array([self.cfg.U_inf, 0.0, 0.0]), (len(self.outlet), 1))
        self.object_cells = np.unique(self.owner[self.object])
        self.history: list[dict] = []
        self.frames: list[Image.Image] = []

        self._build_pressure_matrix()
        # Precompute mid-plane sample mask / grid for fast GIF frames
        z0 = self.cfg.sphere_c[2]
        self._mid_mask = np.abs(self.xc[:, 2] - z0) < 0.12
        self._gx = np.linspace(0.0, self.cfg.L, 160)
        self._gy = np.linspace(0.0, self.cfg.W, 100)
        self._GX, self._GY = np.meshgrid(self._gx, self._gy)
        self._inside = (self._GX - self.cfg.sphere_c[0]) ** 2 + (
            self._GY - self.cfg.sphere_c[1]
        ) ** 2 <= self.cfg.sphere_r**2

    def _build_pressure_matrix(self) -> None:
        """Optional Laplacian (kept for diagnostics); AC scheme does not require it each step."""
        P = self.owner[self.internal]
        N = self.neigh[self.internal]
        a = np.abs(self.a_int)
        self.a_int = a
        rows = np.concatenate([P, N, P, N])
        cols = np.concatenate([N, P, P, N])
        vals = np.concatenate([-a, -a, a, a])
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(self.n_c, self.n_c)).tocsr()
        pin = int(np.argmax(self.xc[:, 0]))
        self._pin = pin
        A = A.tolil()
        A[pin, :] = 0.0
        A[pin, pin] = 1.0
        self.A_p = A.tocsr()
        self._solve_p = None  # unused by AC stepper

    def _interp_int(self, phi: np.ndarray) -> np.ndarray:
        P = self.owner[self.internal]
        N = self.neigh[self.internal]
        if phi.ndim == 1:
            return self.wP * phi[P] + self.wN * phi[N]
        return self.wP[:, None] * phi[P] + self.wN[:, None] * phi[N]

    def _apply_velocity_bcs(self, U_cell: np.ndarray, Uf: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        Uf = Uf.copy()
        Uf[self.boundary] = U_cell[self.owner[self.boundary]]

        # Inlet: freestream + mild TI (small, so it does not dominate the sphere wake)
        noise = cfg.inlet_ti * cfg.U_inf * self.rng.normal(size=(len(self.inlet), 3))
        noise[:, 0] *= 0.5
        Uf[self.inlet] = np.array([cfg.U_inf, 0.0, 0.0]) + noise

        # Outlet: convective Orlanski — non-reflecting / soft
        Uc = max(cfg.U_inf, 1e-6)
        P = self.owner[self.outlet]
        n = self.nf[self.outlet]
        dx = np.abs(np.einsum("ij,ij->i", self.d_owner[self.outlet], n)) + 1e-6
        Uf_out = self.Uf_out - cfg.dt * Uc * (U_cell[P] - self.Uf_out) / dx[:, None]
        un = np.einsum("ij,ij->i", Uf_out, n)
        Uf_out = Uf_out - np.minimum(un, 0.0)[:, None] * n
        # Blend with interior extrapolation for robustness
        Uf_out = 0.5 * Uf_out + 0.5 * U_cell[P]
        un = np.einsum("ij,ij->i", Uf_out, n)
        Uf_out = Uf_out - np.minimum(un, 0.0)[:, None] * n
        Uf[self.outlet] = Uf_out
        self.Uf_out = Uf_out

        # Outer walls: free-slip
        n = self.nf[self.walls]
        u = U_cell[self.owner[self.walls]]
        Uf[self.walls] = u - np.einsum("ij,ij->i", u, n)[:, None] * n

        # Sphere: no-slip
        Uf[self.object] = 0.0
        return Uf

    def _face_velocity_from_cells(self, U_cell: np.ndarray) -> np.ndarray:
        Uf = np.zeros((self.n_f, 3), dtype=np.float64)
        Uf[self.internal] = self._interp_int(U_cell)
        return self._apply_velocity_bcs(U_cell, Uf)

    def _divergence(self, phi: np.ndarray) -> np.ndarray:
        div = np.zeros(self.n_c, dtype=np.float64)
        np.add.at(div, self.owner, phi)
        np.add.at(div, self.neigh[self.internal], -phi[self.internal])
        return div

    def _grad_from_faces(self, phi_f: np.ndarray) -> np.ndarray:
        grad = np.zeros((self.n_c, 3), dtype=np.float64)
        contrib = phi_f[:, None] * self.Sf
        np.add.at(grad, self.owner, contrib)
        np.add.at(grad, self.neigh[self.internal], -contrib[self.internal])
        return grad / self.V[:, None]

    def _smagorinsky(self, Uf: np.ndarray) -> None:
        gradU = np.zeros((self.n_c, 3, 3), dtype=np.float64)
        for i in range(3):
            contrib = Uf[:, i : i + 1] * self.Sf
            np.add.at(gradU[:, i, :], self.owner, contrib)
            np.add.at(gradU[:, i, :], self.neigh[self.internal], -contrib[self.internal])
        gradU /= self.V[:, None, None]
        S = 0.5 * (gradU + np.transpose(gradU, (0, 2, 1)))
        S_mag = np.sqrt(2.0 * np.sum(S * S, axis=(1, 2)) + 1e-30)
        delta = np.minimum(self.V ** (1.0 / 3.0), 0.5)
        self.nu_t = np.clip((self.cfg.Cs * delta) ** 2 * S_mag, 0.0, 50.0 * self.cfg.nu)

    def step(self) -> None:
        cfg = self.cfg
        rho, nu, dt = cfg.rho, cfg.nu, cfg.dt

        Uf = self._face_velocity_from_cells(self.U)
        self._smagorinsky(Uf)
        nu_eff = nu + self.nu_t
        phi = np.einsum("ij,ij->i", Uf, self.Sf)

        # Momentum (explicit convection + diffusion)
        conv = np.zeros((self.n_c, 3), dtype=np.float64)
        P = self.owner[self.internal]
        N = self.neigh[self.internal]
        flux = phi[self.internal]
        U_up = np.where((flux >= 0.0)[:, None], self.U[P], self.U[N])
        contrib = flux[:, None] * U_up
        np.add.at(conv, P, contrib)
        np.add.at(conv, N, -contrib)
        Pb = self.owner[self.boundary]
        np.add.at(conv, Pb, phi[self.boundary][:, None] * Uf[self.boundary])

        diff = np.zeros((self.n_c, 3), dtype=np.float64)
        nu_f = 0.5 * (nu_eff[P] + nu_eff[N])
        diff_f = (nu_f * self.a_int)[:, None] * (self.U[N] - self.U[P])
        np.add.at(diff, P, diff_f)
        np.add.at(diff, N, -diff_f)
        diff_b = (nu_eff[Pb] * self.a_bnd)[:, None] * (Uf[self.boundary] - self.U[Pb])
        np.add.at(diff, Pb, diff_b)

        # Existing pressure gradient (Green–Gauss), soft at outlet
        pf = np.zeros(self.n_f, dtype=np.float64)
        pf[self.internal] = self._interp_int(self.p)
        pf[self.boundary] = self.p[self.owner[self.boundary]]
        pf[self.outlet] *= 0.25  # soft outlet pressure (non-reflecting)
        grad_p = self._grad_from_faces(pf)

        U = self.U + dt * ((-conv + diff) / self.V[:, None] - grad_p / rho)

        # Artificial compressibility iterations (stable projection on unstructured mesh)
        # p_t = -c^2 (∇·U), with flux-consistent velocity adjustment
        c2 = 8.0
        for _ in range(4):
            Uf = self._face_velocity_from_cells(U)
            phi = np.einsum("ij,ij->i", Uf, self.Sf)
            div_raw = self._divergence(phi)
            self.p = self.p - dt * c2 * (div_raw / self.V)
            self.p -= self.p.mean()

            dp = self.p[N] - self.p[P]
            dphi = (dt / rho) * self.a_int * dp
            dU = -(dphi / (self.magSf[self.internal] ** 2 + 1e-30))[:, None] * self.Sf[self.internal]
            np.add.at(U, P, 0.25 * dU)
            np.add.at(U, N, 0.25 * dU)

        U[self.object_cells] *= 0.85
        self.U = cfg.urf_u * U + (1.0 - cfg.urf_u) * self.U

        speed = np.linalg.norm(self.U, axis=1)
        cap = 2.5 * cfg.U_inf
        too_fast = speed > cap
        if np.any(too_fast):
            self.U[too_fast] *= (cap / speed[too_fast])[:, None]

    def capture_frame(self, t: float, step: int) -> None:
        """Capture one mid-plane |U| frame for the GIF."""
        cfg = self.cfg
        c = self.xc[self._mid_mask]
        speed = np.linalg.norm(self.U[self._mid_mask], axis=1) / cfg.U_inf
        speed_g = griddata(c[:, :2], speed, (self._GX, self._GY), method="linear")
        speed_g = np.ma.array(speed_g, mask=self._inside | ~np.isfinite(speed_g))

        fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=100)
        cf = ax.contourf(
            self._GX,
            self._GY,
            speed_g,
            levels=24,
            cmap="viridis",
            vmin=0.0,
            vmax=cfg.gif_vmax,
        )
        ax.add_patch(
            Circle(
                (cfg.sphere_c[0], cfg.sphere_c[1]),
                cfg.sphere_r,
                facecolor="white",
                edgecolor="k",
                lw=1.5,
                zorder=5,
            )
        )
        ax.set_aspect("equal")
        ax.set_xlim(0, cfg.L)
        ax.set_ylim(0, cfg.W)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"LES flow past sphere  |  step {step}  t={t:.4f}")
        ax.axvline(0.0, color="tab:green", ls="--", lw=1)
        ax.axvline(cfg.L, color="tab:red", ls="--", lw=1)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label=r"$|U|/U_\infty$")
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        self.frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    def save_gif(self, path: str = "flow_past_sphere.gif") -> str:
        if not self.frames:
            raise RuntimeError("No frames captured. Run with frame capturing enabled.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.frames[0].save(
            out,
            save_all=True,
            append_images=self.frames[1:],
            duration=self.cfg.gif_duration_ms,
            loop=0,
            optimize=False,
        )
        print(f"Saved GIF → {out}  ({len(self.frames)} frames)")
        return str(out)

    def run(
        self,
        t_end: float | None = None,
        max_steps: int | None = None,
        capture_gif: bool = True,
    ) -> None:
        cfg = self.cfg
        t_end = cfg.t_end if t_end is None else t_end
        n_steps = int(np.ceil(t_end / cfg.dt))
        if max_steps is not None:
            n_steps = min(n_steps, max_steps)

        print(
            f"LES flow past sphere | cells={self.n_c} faces={self.n_f} "
            f"Re≈{cfg.U_inf * 2 * cfg.sphere_r / cfg.nu:.0f} dt={cfg.dt} steps={n_steps}"
        )
        print(
            "BCs: inlet=U∞(+mild TI), outlet=convective/soft-p, "
            "outer walls=free-slip, sphere=no-slip"
        )
        if capture_gif:
            self.frames = []
            print(f"GIF frames every {cfg.frame_every} steps")

        t = 0.0
        for step in range(1, n_steps + 1):
            self.step()
            t += cfg.dt
            if capture_gif and (step == 1 or step % cfg.frame_every == 0 or step == n_steps):
                self.capture_frame(t, step)
            if step % cfg.print_every == 0 or step == 1 or step == n_steps:
                speed = np.linalg.norm(self.U, axis=1)
                if not np.isfinite(speed).all():
                    raise RuntimeError(f"Non-finite solution at step {step}")
                print(
                    f"step {step:5d}/{n_steps}  t={t:.4f}  "
                    f"|U| mean={speed.mean():.4f} max={speed.max():.4f}  "
                    f"nu_t mean={self.nu_t.mean():.3e}  "
                    f"p range=[{self.p.min():.3e},{self.p.max():.3e}]"
                )
                self.history.append(
                    {
                        "step": step,
                        "t": t,
                        "U_mean": float(speed.mean()),
                        "U_max": float(speed.max()),
                        "nu_t_mean": float(self.nu_t.mean()),
                    }
                )

    def save_fields(self, path: str = "les_sphere_fields.npz") -> None:
        np.savez_compressed(
            path,
            cell_centroids=self.xc,
            U=self.U,
            p=self.p,
            nu_t=self.nu_t,
            points=self.points,
            tetra=self.tetra,
            sphere_c=np.array(self.cfg.sphere_c),
            sphere_r=self.cfg.sphere_r,
            U_inf=self.cfg.U_inf,
            nu=self.cfg.nu,
            Cs=self.cfg.Cs,
        )
        print(f"Saved fields → {path}")

    def plot_midplane(self, path: str = "flow_past_sphere_midplane.png", band: float = 0.12) -> str:
        from scipy.interpolate import griddata

        cfg = self.cfg
        z0 = cfg.sphere_c[2]
        mask = np.abs(self.xc[:, 2] - z0) < band
        c = self.xc[mask]
        U = self.U[mask]
        speed = np.linalg.norm(U, axis=1) / cfg.U_inf
        p = self.p[mask]

        gx = np.linspace(0.0, cfg.L, 220)
        gy = np.linspace(0.0, cfg.W, 140)
        GX, GY = np.meshgrid(gx, gy)
        speed_g = griddata(c[:, :2], speed, (GX, GY), method="linear")
        p_g = griddata(c[:, :2], p, (GX, GY), method="linear")
        # Mask interior of sphere
        inside = (GX - cfg.sphere_c[0]) ** 2 + (GY - cfg.sphere_c[1]) ** 2 <= cfg.sphere_r**2
        speed_g = np.ma.array(speed_g, mask=inside | ~np.isfinite(speed_g))
        p_g = np.ma.array(p_g, mask=inside | ~np.isfinite(p_g))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for ax, field, title, cmap in [
            (axes[0], speed_g, r"Velocity magnitude $|U|/U_\infty$", "viridis"),
            (axes[1], p_g, "Pressure $p$", "coolwarm"),
        ]:
            cf = ax.contourf(GX, GY, field, levels=28, cmap=cmap)
            ax.add_patch(
                Circle(
                    (cfg.sphere_c[0], cfg.sphere_c[1]),
                    cfg.sphere_r,
                    facecolor="white",
                    edgecolor="k",
                    lw=1.8,
                    zorder=5,
                )
            )
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title)
            ax.set_xlim(0, cfg.L)
            ax.set_ylim(0, cfg.W)
            ax.axvline(0.0, color="tab:green", ls="--", lw=1.2, label="inlet (U∞)")
            ax.axvline(cfg.L, color="tab:red", ls="--", lw=1.2, label="outlet (convective)")
            fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

        # Streamlines + velocity vectors on the speed panel
        Ux = griddata(c[:, :2], U[:, 0], (GX, GY), method="linear")
        Uy = griddata(c[:, :2], U[:, 1], (GX, GY), method="linear")
        Ux = np.nan_to_num(np.where(inside, np.nan, Ux), nan=0.0)
        Uy = np.nan_to_num(np.where(inside, np.nan, Uy), nan=0.0)
        axes[0].streamplot(
            gx,
            gy,
            Ux,
            Uy,
            color="white",
            density=1.2,
            linewidth=0.8,
            arrowsize=0.9,
            zorder=3,
        )

        # Velocity vectors around the sphere (zoom-friendly subsample)
        near = (
            (c[:, 0] > 1.0)
            & (c[:, 0] < 7.0)
            & (c[:, 1] > 0.5)
            & (c[:, 1] < 4.5)
        )
        idx = np.where(near)[0]
        if len(idx) > 400:
            idx = idx[:: max(1, len(idx) // 400)]
        axes[0].quiver(
            c[idx, 0],
            c[idx, 1],
            U[idx, 0],
            U[idx, 1],
            color="white",
            alpha=0.35,
            scale=35,
            width=0.0025,
            zorder=4,
        )
        axes[0].legend(loc="upper right", fontsize=8)
        fig.suptitle(
            "Turbulent LES flow past a sphere — mid-plane\n"
            "Inlet: fixed U∞ + mild TI · Outlet: convective (non-reflecting) · "
            "Outer walls: free-slip · Sphere: no-slip",
            fontsize=11,
        )
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"Saved visualization → {path}")
        return path

    def plot_streamlines(self, path: str = "flow_past_sphere_streamlines.png", band: float = 0.12) -> str:
        """Draw mid-plane streamlines of the in-plane velocity field around the sphere."""
        cfg = self.cfg
        z0 = cfg.sphere_c[2]
        mask = np.abs(self.xc[:, 2] - z0) < band
        c = self.xc[mask]
        U = self.U[mask]
        speed = np.linalg.norm(U, axis=1) / cfg.U_inf

        gx = np.linspace(0.05, cfg.L - 0.05, 200)
        gy = np.linspace(0.05, cfg.W - 0.05, 120)
        GX, GY = np.meshgrid(gx, gy)
        Ux = griddata(c[:, :2], U[:, 0], (GX, GY), method="linear")
        Uy = griddata(c[:, :2], U[:, 1], (GX, GY), method="linear")
        speed_g = griddata(c[:, :2], speed, (GX, GY), method="linear")

        inside = (GX - cfg.sphere_c[0]) ** 2 + (GY - cfg.sphere_c[1]) ** 2 <= (cfg.sphere_r * 1.02) ** 2
        bad = inside | ~np.isfinite(Ux) | ~np.isfinite(Uy) | ~np.isfinite(speed_g)
        Ux = np.array(Ux, dtype=np.float64)
        Uy = np.array(Uy, dtype=np.float64)
        speed_g = np.array(speed_g, dtype=np.float64)
        Ux[bad] = np.nan
        Uy[bad] = np.nan
        speed_g[bad] = np.nan

        # streamplot needs finite values; fill masked region with 0 then mask visually with sphere patch
        Ux_s = np.nan_to_num(Ux, nan=0.0)
        Uy_s = np.nan_to_num(Uy, nan=0.0)
        speed_plot = np.ma.array(speed_g, mask=~np.isfinite(speed_g))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        # Full domain streamlines
        ax = axes[0]
        cf = ax.contourf(GX, GY, speed_plot, levels=28, cmap="viridis")
        ax.streamplot(
            gx,
            gy,
            Ux_s,
            Uy_s,
            color="white",
            density=1.35,
            linewidth=0.9,
            arrowsize=1.0,
            minlength=0.1,
        )
        ax.add_patch(
            Circle(
                (cfg.sphere_c[0], cfg.sphere_c[1]),
                cfg.sphere_r,
                facecolor="white",
                edgecolor="k",
                lw=1.8,
                zorder=5,
            )
        )
        ax.set_aspect("equal")
        ax.set_xlim(0, cfg.L)
        ax.set_ylim(0, cfg.W)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Streamlines — full domain")
        ax.axvline(0.0, color="tab:green", ls="--", lw=1.2, label="inlet")
        ax.axvline(cfg.L, color="tab:red", ls="--", lw=1.2, label="outlet")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label=r"$|U|/U_\infty$")

        # Zoom around sphere / near wake
        ax = axes[1]
        cf2 = ax.contourf(GX, GY, speed_plot, levels=28, cmap="viridis")
        ax.streamplot(
            gx,
            gy,
            Ux_s,
            Uy_s,
            color="white",
            density=2.0,
            linewidth=1.0,
            arrowsize=1.1,
            minlength=0.08,
        )
        ax.add_patch(
            Circle(
                (cfg.sphere_c[0], cfg.sphere_c[1]),
                cfg.sphere_r,
                facecolor="white",
                edgecolor="k",
                lw=1.8,
                zorder=5,
            )
        )
        ax.set_aspect("equal")
        ax.set_xlim(1.5, 6.5)
        ax.set_ylim(0.8, 4.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Streamlines — zoom near sphere")
        fig.colorbar(cf2, ax=ax, fraction=0.046, pad=0.04, label=r"$|U|/U_\infty$")

        fig.suptitle(
            "Streamlines of turbulent LES flow past a sphere (mid-plane)\n"
            "Inlet: U∞ · Outlet: convective · Outer walls: free-slip · Sphere: no-slip",
            fontsize=11,
        )
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"Saved streamlines → {path}")
        return path

    def plot_wake_profile(self, path: str = "flow_past_sphere_wake.png") -> str:
        cfg = self.cfg
        cy, cz = cfg.sphere_c[1], cfg.sphere_c[2]
        mask = (
            (np.abs(self.xc[:, 1] - cy) < 0.2)
            & (np.abs(self.xc[:, 2] - cz) < 0.2)
            & (self.xc[:, 0] > cfg.sphere_c[0])
        )
        x = self.xc[mask, 0]
        u = self.U[mask, 0]
        order = np.argsort(x)
        x, u = x[order], u[order]

        # Bin-average for a cleaner wake curve while keeping scatter of raw samples
        bins = np.linspace(cfg.sphere_c[0] + cfg.sphere_r, cfg.L, 60)
        dig = np.digitize(x, bins)
        xb, ub = [], []
        for i in range(1, len(bins)):
            sel = dig == i
            if np.any(sel):
                xb.append(0.5 * (bins[i - 1] + bins[i]))
                ub.append(np.mean(u[sel]) / cfg.U_inf)

        fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
        ax.plot(x, u / cfg.U_inf, ".", ms=2.5, alpha=0.25, color="tab:blue", label="samples")
        if xb:
            ax.plot(xb, ub, "-", lw=2.2, color="tab:red", label="binned mean")
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.axvline(cfg.sphere_c[0] + cfg.sphere_r, color="gray", ls="--", label="sphere rear")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$U_x / U_\infty$")
        ax.set_title("Wake centerline velocity behind the sphere")
        ax.set_ylim(-0.5, 1.4)
        ax.legend()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"Saved wake profile → {path}")
        return path

    def plot_nu_t(self, path: str = "flow_past_sphere_nut.png", band: float = 0.12) -> str:
        from scipy.interpolate import griddata

        cfg = self.cfg
        z0 = cfg.sphere_c[2]
        mask = np.abs(self.xc[:, 2] - z0) < band
        c = self.xc[mask]
        nut = self.nu_t[mask] / max(cfg.nu, 1e-30)

        gx = np.linspace(0.0, cfg.L, 220)
        gy = np.linspace(0.0, cfg.W, 140)
        GX, GY = np.meshgrid(gx, gy)
        nut_g = griddata(c[:, :2], nut, (GX, GY), method="linear")
        inside = (GX - cfg.sphere_c[0]) ** 2 + (GY - cfg.sphere_c[1]) ** 2 <= cfg.sphere_r**2
        nut_g = np.ma.array(nut_g, mask=inside | ~np.isfinite(nut_g))

        fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
        cf = ax.contourf(GX, GY, nut_g, levels=24, cmap="magma")
        ax.add_patch(
            Circle(
                (cfg.sphere_c[0], cfg.sphere_c[1]),
                cfg.sphere_r,
                facecolor="white",
                edgecolor="k",
                lw=1.8,
                zorder=5,
            )
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(r"Smagorinsky eddy viscosity $\nu_t/\nu$ (mid-plane)")
        ax.set_xlim(0, cfg.L)
        ax.set_ylim(0, cfg.W)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"Saved eddy-viscosity map → {path}")
        return path

    def write_all_outputs(self, prefix: str = "flow_past_sphere") -> dict[str, str]:
        fields = "les_sphere_fields.npz"
        self.save_fields(fields)
        outs = {
            "fields": fields,
            "midplane": self.plot_midplane(f"{prefix}_midplane.png"),
            "streamlines": self.plot_streamlines(f"{prefix}_streamlines.png"),
            "wake": self.plot_wake_profile(f"{prefix}_wake.png"),
            "nut": self.plot_nu_t(f"{prefix}_nut.png"),
        }
        if self.frames:
            outs["gif"] = self.save_gif(f"{prefix}.gif")
        return outs


def main() -> None:
    parser = argparse.ArgumentParser(description="LES turbulent flow past a sphere")
    parser.add_argument("--mesh", default="processed_mesh.npz")
    parser.add_argument("--t-end", type=float, default=0.2)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", action="store_true", help="Longer run to generate final plots/GIF")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    cfg = SimConfig(t_end=args.t_end, dt=args.dt, print_every=50, inlet_ti=0.015)
    if args.quick:
        cfg.t_end = 0.02
        cfg.dt = 1e-4
        cfg.print_every = 20
        cfg.frame_every = 10
        max_steps = 100
    elif args.output:
        cfg.t_end = 0.2
        cfg.dt = 1e-4
        cfg.print_every = 100
        cfg.frame_every = 25
        max_steps = 800
    else:
        max_steps = args.max_steps

    solver = SphereLESSolver(mesh_path=args.mesh, cfg=cfg)
    solver.run(max_steps=max_steps, capture_gif=not args.no_gif)
    outs = solver.write_all_outputs()
    print("Generated outputs:", outs)


if __name__ == "__main__":
    main()
