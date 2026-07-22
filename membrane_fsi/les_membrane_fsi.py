#!/usr/bin/env python3
"""
LES flow past a FLEXIBLE rectangular membrane (FSI).

Keeps the sphere package (working/) unchanged. This is a new code path:
  - Fluid: same Smagorinsky LES FV approach as the sphere solver
  - Solid: cantilever flexible membrane (inertia + bending + damping)
  - Interface: no-slip with moving-wall velocity from membrane motion
  - Coupling: partitioned weak FSI each time step

Boundary conditions (fluid):
  inlet  : U = U∞ + mild TI
  outlet : convective Orlanski + soft pressure
  walls  : free-slip
  membrane: no-slip with U_wall = (v_membrane(z), 0, 0)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from solid.membrane_model import CantileverMembrane, MembraneSolidConfig  # noqa: E402


INLET, OUTLET, WALL, MEMBRANE = 1, 2, 3, 4


@dataclass
class FluidConfig:
    rho: float = 1.0
    nu: float = 1.0e-3
    Cs: float = 0.1
    U_inf: float = 1.0
    dt: float = 1.0e-4
    t_end: float = 0.1
    inlet_ti: float = 0.01
    seed: int = 11
    print_every: int = 50
    frame_every: int = 20
    urf_u: float = 0.5
    L: float = 10.0
    W: float = 5.0
    H: float = 5.0
    gif_duration_ms: int = 120
    gif_vmax: float = 1.6
    fsi_relax: float = 0.5  # under-relax wall velocity


class MembraneFSISolver:
    def __init__(
        self,
        mesh_path: str,
        fluid_cfg: FluidConfig | None = None,
        solid_cfg: MembraneSolidConfig | None = None,
        geom_meta: str | None = None,
    ):
        self.cfg = fluid_cfg or FluidConfig()
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
        self.membrane = np.where(self.tag == MEMBRANE)[0]
        self.internal = np.where(self.neigh >= 0)[0]
        self.boundary = np.where(self.neigh < 0)[0]

        if len(self.membrane) == 0:
            raise RuntimeError("No membrane faces found (boundary_tag==4). Reprocess mesh.")

        self.d_owner = self.xf - self.xc[self.owner]
        neigh_safe = np.where(self.neigh >= 0, self.neigh, 0)
        self.d_neigh = self.xf - self.xc[neigh_safe]

        P = self.owner[self.internal]
        N = self.neigh[self.internal]
        dPN = self.xc[N] - self.xc[P]
        dn = np.einsum("ij,ij->i", dPN, self.nf[self.internal])
        self.a_int = self.magSf[self.internal] / np.maximum(np.abs(dn), 1e-6)
        delta_b = np.abs(np.einsum("ij,ij->i", self.d_owner[self.boundary], self.nf[self.boundary]))
        self.a_bnd = self.magSf[self.boundary] / np.maximum(delta_b, 1e-6)

        dP = np.linalg.norm(self.d_owner[self.internal], axis=1) + 1e-30
        dN = np.linalg.norm(self.d_neigh[self.internal], axis=1) + 1e-30
        self.wN = dP / (dP + dN)
        self.wP = 1.0 - self.wN

        # Solid model
        scfg = solid_cfg or MembraneSolidConfig()
        if geom_meta and Path(geom_meta).exists():
            meta = np.load(geom_meta)
            scfg.plate_x = float(meta["plate_x"])
            scfg.plate_t = float(meta["plate_t"])
            scfg.plate_w = float(meta["plate_w"])
            scfg.plate_h = float(meta["plate_h"])
            scfg.plate_yc = float(meta["plate_yc"])
            scfg.plate_zc = float(meta["plate_zc"])
            scfg.thickness = float(meta["plate_t"])
        self.solid = CantileverMembrane(scfg)
        self.scfg = scfg

        # Map membrane faces → nearest solid node (by z)
        self.mem_z = self.xf[self.membrane, 2]
        self.mem_node = np.clip(
            np.searchsorted(self.solid.z, self.mem_z) - 1, 0, len(self.solid.z) - 1
        )
        # Prefer closer of the two neighbours
        for i, z in enumerate(self.mem_z):
            j = self.mem_node[i]
            if j < len(self.solid.z) - 1 and abs(self.solid.z[j + 1] - z) < abs(self.solid.z[j] - z):
                self.mem_node[i] = j + 1

        # Front vs back faces by normal x-component (for pressure difference)
        self.mem_nx = self.nf[self.membrane, 0]

        self.rng = np.random.default_rng(self.cfg.seed)
        self.U = np.zeros((self.n_c, 3), dtype=np.float64)
        self.U[:, 0] = self.cfg.U_inf
        self.p = np.zeros(self.n_c, dtype=np.float64)
        self.nu_t = np.zeros(self.n_c, dtype=np.float64)
        self.Uf_out = np.tile(np.array([self.cfg.U_inf, 0.0, 0.0]), (len(self.outlet), 1))
        self.u_wall = np.zeros(len(self.membrane), dtype=np.float64)  # streamwise wall speed
        self.frames: list[Image.Image] = []
        self.history: list[dict] = []

        # Midplane viz helpers (z = plate centre)
        z0 = scfg.plate_zc
        self._mid_mask = np.abs(self.xc[:, 2] - z0) < 0.15
        self._gx = np.linspace(0.0, self.cfg.L, 160)
        self._gy = np.linspace(0.0, self.cfg.W, 100)
        self._GX, self._GY = np.meshgrid(self._gx, self._gy)

    # ----- FV helpers (same philosophy as sphere solver) -----
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

        noise = cfg.inlet_ti * cfg.U_inf * self.rng.normal(size=(len(self.inlet), 3))
        noise[:, 0] *= 0.5
        Uf[self.inlet] = np.array([cfg.U_inf, 0.0, 0.0]) + noise

        # Outlet Orlanski
        Uc = max(cfg.U_inf, 1e-6)
        P = self.owner[self.outlet]
        n = self.nf[self.outlet]
        dx = np.abs(np.einsum("ij,ij->i", self.d_owner[self.outlet], n)) + 1e-6
        Uf_out = self.Uf_out - cfg.dt * Uc * (U_cell[P] - self.Uf_out) / dx[:, None]
        un = np.einsum("ij,ij->i", Uf_out, n)
        Uf_out = Uf_out - np.minimum(un, 0.0)[:, None] * n
        Uf_out = 0.5 * Uf_out + 0.5 * U_cell[P]
        un = np.einsum("ij,ij->i", Uf_out, n)
        Uf_out = Uf_out - np.minimum(un, 0.0)[:, None] * n
        Uf[self.outlet] = Uf_out
        self.Uf_out = Uf_out

        # Outer walls free-slip
        n = self.nf[self.walls]
        u = U_cell[self.owner[self.walls]]
        Uf[self.walls] = u - np.einsum("ij,ij->i", u, n)[:, None] * n

        # Membrane: NO-SLIP with moving wall velocity from solid
        # U_wall = (v_x(z), 0, 0)  — flexible solid kinematics
        Uf[self.membrane] = 0.0
        Uf[self.membrane, 0] = self.u_wall
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

    def _fluid_forces_on_membrane_nodes(self) -> np.ndarray:
        """Integrate -p n_x * A (+ viscous normal approx) onto solid nodes."""
        cfg = self.cfg
        f_nodes = np.zeros(len(self.solid.z), dtype=np.float64)
        # Pressure at owner cells of membrane faces
        p_f = self.p[self.owner[self.membrane]]
        # Traction ~ -p n ; streamwise component (-p n_x) * area
        # For a thin plate, opposite faces have opposite n_x → net Δp force
        fx = (-p_f * self.mem_nx) * self.magSf[self.membrane]
        np.add.at(f_nodes, self.mem_node, fx)

        # Mild viscous contribution using relative wall shear proxy
        # tau ~ mu * |U_owner - U_wall| / delta_n
        Pb = self.owner[self.membrane]
        delta_n = np.maximum(
            np.abs(np.einsum("ij,ij->i", self.d_owner[self.membrane], self.nf[self.membrane])),
            1e-4,
        )
        u_rel = self.U[Pb, 0] - self.u_wall
        tau = (cfg.nu + self.nu_t[Pb]) * cfg.rho * u_rel / delta_n
        # shear traction along wall tangential; project to x
        fx_v = -tau * self.magSf[self.membrane] * np.sign(self.mem_nx + 1e-30) * 0.1
        np.add.at(f_nodes, self.mem_node, fx_v)
        f_nodes[0] = 0.0  # clamped
        return f_nodes

    def step(self) -> None:
        cfg = self.cfg
        rho, nu, dt = cfg.rho, cfg.nu, cfg.dt

        # --- Solid → Fluid: update moving no-slip wall velocity ---
        v_face = self.solid.velocity_at_z(self.mem_z)
        self.u_wall = cfg.fsi_relax * v_face + (1.0 - cfg.fsi_relax) * self.u_wall

        Uf = self._face_velocity_from_cells(self.U)
        self._smagorinsky(Uf)
        nu_eff = nu + self.nu_t
        phi = np.einsum("ij,ij->i", Uf, self.Sf)

        # Momentum
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

        pf = np.zeros(self.n_f, dtype=np.float64)
        pf[self.internal] = self._interp_int(self.p)
        pf[self.boundary] = self.p[self.owner[self.boundary]]
        pf[self.outlet] *= 0.25
        grad_p = self._grad_from_faces(pf)

        U = self.U + dt * ((-conv + diff) / self.V[:, None] - grad_p / rho)

        # Artificial compressibility
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

        self.U = cfg.urf_u * U + (1.0 - cfg.urf_u) * self.U
        speed = np.linalg.norm(self.U, axis=1)
        cap = 2.5 * cfg.U_inf
        too_fast = speed > cap
        if np.any(too_fast):
            self.U[too_fast] *= (cap / speed[too_fast])[:, None]

        # --- Fluid → Solid: nodal forces, structural step ---
        f_nodes = self._fluid_forces_on_membrane_nodes()
        self.solid.step(f_nodes, dt)

    def run(self, max_steps: int | None = None, capture_gif: bool = True) -> None:
        cfg = self.cfg
        n_steps = int(np.ceil(cfg.t_end / cfg.dt))
        if max_steps is not None:
            n_steps = min(n_steps, max_steps)

        print(
            f"LES–FSI flexible membrane | cells={self.n_c} membrane_faces={len(self.membrane)} "
            f"Re≈{cfg.U_inf * self.scfg.plate_w / cfg.nu:.0f} (based on plate width) "
            f"dt={cfg.dt} steps={n_steps}"
        )
        print(
            "BCs: inlet=U∞(+TI), outlet=convective, walls=free-slip, "
            "membrane=moving no-slip (flexible solid)"
        )
        print(
            f"Solid: cantilever membrane E={self.scfg.E:.1e} rho_s={self.scfg.rho_s} "
            f"nodes={self.scfg.n_nodes}"
        )
        if capture_gif:
            self.frames = []

        t = 0.0
        for step in range(1, n_steps + 1):
            self.step()
            t += cfg.dt
            if capture_gif and (step == 1 or step % cfg.frame_every == 0 or step == n_steps):
                self.capture_frame(t, step)
            if step % cfg.print_every == 0 or step == 1 or step == n_steps:
                speed = np.linalg.norm(self.U, axis=1)
                tip = self.solid.tip_deflection()
                print(
                    f"step {step:5d}/{n_steps} t={t:.4f} "
                    f"|U|mean={speed.mean():.3f} tip_η={tip:+.6f} "
                    f"tip_v={self.solid.v[-1]:+.6f} νt_mean={self.nu_t.mean():.2e}"
                )
                self.history.append(
                    {"step": step, "t": t, "tip_eta": tip, "U_mean": float(speed.mean())}
                )

    def capture_frame(self, t: float, step: int) -> None:
        cfg = self.cfg
        c = self.xc[self._mid_mask]
        speed = np.linalg.norm(self.U[self._mid_mask], axis=1) / cfg.U_inf
        speed_g = griddata(c[:, :2], speed, (self._GX, self._GY), method="linear")
        # Mask membrane rectangle in midplane (approx)
        sc = self.scfg
        x0, y0 = sc.plate_x - 0.5 * sc.plate_t, sc.plate_yc - 0.5 * sc.plate_w
        # Visual deflection: shift drawn rectangle by tip/2
        x_draw = x0 + 0.5 * self.solid.tip_deflection()
        inside = (
            (self._GX >= x_draw)
            & (self._GX <= x_draw + sc.plate_t)
            & (self._GY >= y0)
            & (self._GY <= y0 + sc.plate_w)
        )
        speed_g = np.ma.array(speed_g, mask=inside | ~np.isfinite(speed_g))

        fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=100)
        cf = ax.contourf(
            self._GX, self._GY, speed_g, levels=24, cmap="viridis", vmin=0.0, vmax=cfg.gif_vmax
        )
        ax.add_patch(
            Rectangle(
                (x_draw, y0),
                sc.plate_t,
                sc.plate_w,
                facecolor="white",
                edgecolor="crimson",
                lw=1.6,
                zorder=5,
            )
        )
        ax.set_aspect("equal")
        ax.set_xlim(0, cfg.L)
        ax.set_ylim(0, cfg.W)
        ax.set_title(f"LES–FSI flexible membrane | step {step} t={t:.4f} tipη={self.solid.tip_deflection():+.3f}")
        ax.axvline(0.0, color="tab:green", ls="--", lw=1)
        ax.axvline(cfg.L, color="tab:red", ls="--", lw=1)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label=r"$|U|/U_\infty$")
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        self.frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    def save_gif(self, path: str) -> str:
        if not self.frames:
            raise RuntimeError("No GIF frames captured")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.frames[0].save(
            out,
            save_all=True,
            append_images=self.frames[1:],
            duration=self.cfg.gif_duration_ms,
            loop=0,
        )
        print(f"Saved GIF → {out}")
        return str(out)

    def plot_midplane(self, path: str) -> str:
        cfg, sc = self.cfg, self.scfg
        c = self.xc[self._mid_mask]
        speed = np.linalg.norm(self.U[self._mid_mask], axis=1) / cfg.U_inf
        gx = np.linspace(0, cfg.L, 200)
        gy = np.linspace(0, cfg.W, 120)
        GX, GY = np.meshgrid(gx, gy)
        speed_g = griddata(c[:, :2], speed, (GX, GY), method="linear")
        x_draw = sc.plate_x - 0.5 * sc.plate_t + 0.5 * self.solid.tip_deflection()
        y0 = sc.plate_yc - 0.5 * sc.plate_w
        inside = (GX >= x_draw) & (GX <= x_draw + sc.plate_t) & (GY >= y0) & (GY <= y0 + sc.plate_w)
        speed_g = np.ma.array(speed_g, mask=inside | ~np.isfinite(speed_g))

        fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
        cf = ax.contourf(GX, GY, speed_g, levels=28, cmap="viridis")
        ax.add_patch(
            Rectangle((x_draw, y0), sc.plate_t, sc.plate_w, facecolor="white", edgecolor="crimson", lw=1.8)
        )
        ax.set_aspect("equal")
        ax.set_xlim(0, cfg.L)
        ax.set_ylim(0, cfg.W)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            "LES flow past flexible rectangular membrane (mid-plane)\n"
            "Membrane: moving no-slip | tip deflection shown"
        )
        fig.colorbar(cf, ax=ax, label=r"$|U|/U_\infty$")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Saved → {path}")
        return path

    def plot_membrane_response(self, path: str) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        ax = axes[0]
        ax.plot(self.solid.eta, self.solid.z, "o-", color="crimson")
        ax.axvline(0.0, color="k", ls=":", lw=1)
        ax.set_xlabel("streamwise deflection η")
        ax.set_ylabel("z")
        ax.set_title("Membrane deflection shape (clamped at bottom)")

        ax = axes[1]
        if self.history:
            t = [h["t"] for h in self.history]
            tip = [h["tip_eta"] for h in self.history]
            ax.plot(t, tip, "-", color="tab:blue")
        ax.set_xlabel("t")
        ax.set_ylabel("tip η")
        ax.set_title("Tip deflection vs time")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Saved → {path}")
        return path

    def save_fields(self, path: str) -> None:
        np.savez_compressed(
            path,
            U=self.U,
            p=self.p,
            nu_t=self.nu_t,
            cell_centroids=self.xc,
            membrane_z=self.solid.z,
            membrane_eta=self.solid.eta,
            membrane_v=self.solid.v,
            tip_eta=self.solid.tip_deflection(),
        )
        print(f"Saved fields → {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="LES–FSI flexible rectangular membrane")
    ap.add_argument("--mesh", default=str(ROOT / "data" / "processed_mesh_membrane.npz"))
    ap.add_argument("--geom", default=str(ROOT / "data" / "fluid_mesh_membrane.npz"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--outdir", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    fcfg = FluidConfig()
    if args.quick:
        fcfg.t_end = 0.02
        fcfg.dt = 1e-4
        fcfg.print_every = 20
        fcfg.frame_every = 10
        max_steps = args.steps or 80
    else:
        fcfg.t_end = 0.1
        fcfg.dt = 1e-4
        fcfg.print_every = 50
        fcfg.frame_every = 25
        max_steps = args.steps or 400

    solver = MembraneFSISolver(args.mesh, fluid_cfg=fcfg, geom_meta=args.geom)
    solver.run(max_steps=max_steps, capture_gif=True)
    solver.save_fields(str(out / "les_membrane_fields.npz"))
    solver.plot_midplane(str(out / "flow_past_membrane_midplane.png"))
    solver.plot_membrane_response(str(out / "membrane_deflection.png"))
    if solver.frames:
        solver.save_gif(str(out / "flow_past_membrane.gif"))
    print("Done. Outputs in", out)


if __name__ == "__main__":
    main()
