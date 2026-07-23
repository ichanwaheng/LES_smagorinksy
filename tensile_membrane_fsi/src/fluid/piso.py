"""Incompressible Navier–Stokes on collocated Cartesian grid (PISO-like)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve

from .les import smagorinsky_viscosity
from .mesh import FluidGrid


@dataclass
class FluidState:
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    p: np.ndarray
    nu_eff: np.ndarray


def _diff_central(phi: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """Central difference with one-sided edges (no periodic wrap)."""
    out = np.zeros_like(phi)
    sl_c = [slice(None)] * phi.ndim
    sl_m = [slice(None)] * phi.ndim
    sl_p = [slice(None)] * phi.ndim
    sl_c[axis] = slice(1, -1)
    sl_m[axis] = slice(0, -2)
    sl_p[axis] = slice(2, None)
    out[tuple(sl_c)] = (phi[tuple(sl_p)] - phi[tuple(sl_m)]) / (2.0 * dx)

    sl0 = [slice(None)] * phi.ndim
    sl1 = [slice(None)] * phi.ndim
    sl0[axis] = 0
    sl1[axis] = 1
    out[tuple(sl0)] = (phi[tuple(sl1)] - phi[tuple(sl0)]) / dx

    sln = [slice(None)] * phi.ndim
    slnm = [slice(None)] * phi.ndim
    sln[axis] = -1
    slnm[axis] = -2
    out[tuple(sln)] = (phi[tuple(sln)] - phi[tuple(slnm)]) / dx
    return out


def _laplacian(phi: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """5/7-point Laplacian with Neumann (copy) boundaries — no wrap."""
    lap = np.zeros_like(phi)
    # x
    lap[1:-1] += (phi[2:] - 2 * phi[1:-1] + phi[0:-2]) / dx**2
    lap[0] += (phi[1] - phi[0]) / dx**2
    lap[-1] += (phi[-2] - phi[-1]) / dx**2
    # y
    lap[:, 1:-1] += (phi[:, 2:] - 2 * phi[:, 1:-1] + phi[:, 0:-2]) / dy**2
    lap[:, 0] += (phi[:, 1] - phi[:, 0]) / dy**2
    lap[:, -1] += (phi[:, -2] - phi[:, -1]) / dy**2
    # z
    lap[:, :, 1:-1] += (phi[:, :, 2:] - 2 * phi[:, :, 1:-1] + phi[:, :, 0:-2]) / dz**2
    lap[:, :, 0] += (phi[:, :, 1] - phi[:, :, 0]) / dz**2
    lap[:, :, -1] += (phi[:, :, -2] - phi[:, :, -1]) / dz**2
    return lap


class FluidSolver:
    """Simplified 3D incompressible solver with fractional-step / PISO flavour.

    - Convective terms: first-order upwind
    - Viscous terms: Laplacian with ν_eff (molecular + optional Smagorinsky)
    - Pressure: Poisson with Neumann walls (pinned cell)
    - Immersed membrane: velocity forced in masked cells
    """

    def __init__(
        self,
        grid: FluidGrid,
        rho: float = 1.225,
        nu: float = 1.5e-5,
        U_inlet: float = 10.0,
        use_les: bool = True,
        Cs: float = 0.17,
        u_clip: Optional[float] = None,
    ) -> None:
        self.grid = grid
        self.rho = float(rho)
        self.nu = float(nu)
        self.U_inlet = float(U_inlet)
        self.use_les = bool(use_les)
        self.Cs = float(Cs)
        self.u_clip = float(u_clip) if u_clip is not None else 5.0 * abs(self.U_inlet)

        sh = grid.shape
        self.state = FluidState(
            u=np.full(sh, self.U_inlet * 0.5),
            v=np.zeros(sh),
            w=np.zeros(sh),
            p=np.zeros(sh),
            nu_eff=np.full(sh, self.nu),
        )
        self.state.u[0, :, :] = self.U_inlet
        self._obstacle = np.zeros(sh, dtype=bool)
        self._u_solid = np.zeros(sh)
        self._v_solid = np.zeros(sh)
        self._w_solid = np.zeros(sh)
        self._lap_cache: Optional[sparse.csr_matrix] = None

    def set_immersed_boundary(
        self,
        mask: np.ndarray,
        u_s: Optional[np.ndarray] = None,
        v_s: Optional[np.ndarray] = None,
        w_s: Optional[np.ndarray] = None,
    ) -> None:
        self._obstacle = mask.astype(bool)
        z = np.zeros(self.grid.shape)
        self._u_solid = z if u_s is None else np.asarray(u_s, dtype=float)
        self._v_solid = z if v_s is None else np.asarray(v_s, dtype=float)
        self._w_solid = z if w_s is None else np.asarray(w_s, dtype=float)

    def _apply_bc(self, u, v, w) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # side walls (no-slip)
        for arr in (u, v, w):
            arr[:, 0, :] = 0.0
            arr[:, -1, :] = 0.0
            arr[:, :, 0] = 0.0
            arr[:, :, -1] = 0.0
        # outlet (zero gradient)
        u[-1, :, :] = u[-2, :, :]
        v[-1, :, :] = v[-2, :, :]
        w[-1, :, :] = w[-2, :, :]
        # inlet last
        u[0, :, :] = self.U_inlet
        v[0, :, :] = 0.0
        w[0, :, :] = 0.0
        # immersed membrane (soft: only force if mask)
        m = self._obstacle
        if np.any(m):
            u[m] = self._u_solid[m]
            v[m] = self._v_solid[m]
            w[m] = self._w_solid[m]
        return u, v, w

    def _sanitize(self, *fields: np.ndarray) -> Tuple[np.ndarray, ...]:
        out = []
        for f in fields:
            g = np.nan_to_num(f, nan=0.0, posinf=self.u_clip, neginf=-self.u_clip)
            out.append(np.clip(g, -self.u_clip, self.u_clip))
        return tuple(out)

    def _advect_diffuse(
        self,
        phi: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        nu_eff: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        g = self.grid
        dx, dy, dz = g.dx, g.dy, g.dz
        dudx = np.zeros_like(phi)
        dudy = np.zeros_like(phi)
        dudz = np.zeros_like(phi)

        dudx[1:-1] = np.where(
            u[1:-1] >= 0,
            (phi[1:-1] - phi[:-2]) / dx,
            (phi[2:] - phi[1:-1]) / dx,
        )
        dudy[:, 1:-1] = np.where(
            v[:, 1:-1] >= 0,
            (phi[:, 1:-1] - phi[:, :-2]) / dy,
            (phi[:, 2:] - phi[:, 1:-1]) / dy,
        )
        dudz[:, :, 1:-1] = np.where(
            w[:, :, 1:-1] >= 0,
            (phi[:, :, 1:-1] - phi[:, :, :-2]) / dz,
            (phi[:, :, 2:] - phi[:, :, 1:-1]) / dz,
        )
        conv = u * dudx + v * dudy + w * dudz
        lap = _laplacian(phi, dx, dy, dz)
        # CFL-friendly explicit update
        phi_new = phi + dt * (-conv + nu_eff * lap)
        return phi_new

    def _build_laplacian(self) -> sparse.csr_matrix:
        g = self.grid
        nx, ny, nz = g.shape
        n = nx * ny * nz
        dx2, dy2, dz2 = g.dx**2, g.dy**2, g.dz**2
        rows, cols, data = [], [], []

        def add(r, c, v):
            rows.append(r)
            cols.append(c)
            data.append(v)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = i * ny * nz + j * nz + k
                    diag = 0.0
                    if i > 0:
                        add(idx, idx - ny * nz, 1.0 / dx2)
                        diag -= 1.0 / dx2
                    else:
                        # Neumann → ghost = boundary ⇒ no neighbour contrib
                        pass
                    if i < nx - 1:
                        add(idx, idx + ny * nz, 1.0 / dx2)
                        diag -= 1.0 / dx2
                    if j > 0:
                        add(idx, idx - nz, 1.0 / dy2)
                        diag -= 1.0 / dy2
                    if j < ny - 1:
                        add(idx, idx + nz, 1.0 / dy2)
                        diag -= 1.0 / dy2
                    if k > 0:
                        add(idx, idx - 1, 1.0 / dz2)
                        diag -= 1.0 / dz2
                    if k < nz - 1:
                        add(idx, idx + 1, 1.0 / dz2)
                        diag -= 1.0 / dz2
                    add(idx, idx, diag if abs(diag) > 0 else 1.0)

        A = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tolil()
        A[0, :] = 0.0
        A[0, 0] = 1.0
        return A.tocsr()

    def _pressure_poisson(self, div: np.ndarray, dt: float) -> np.ndarray:
        if self._lap_cache is None:
            self._lap_cache = self._build_laplacian()
        rhs = (self.rho / max(dt, 1e-12)) * div.ravel(order="C")
        rhs = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)
        rhs[0] = 0.0
        p_flat, info = cg(self._lap_cache, rhs, rtol=1e-5, maxiter=300)
        if info != 0 or not np.all(np.isfinite(p_flat)):
            p_flat = spsolve(self._lap_cache, rhs)
        p = np.asarray(p_flat, dtype=float).reshape(self.grid.shape, order="C")
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        # remove mean for Neumann system
        p -= p.mean()
        return p

    def step(self, dt: float) -> FluidState:
        g = self.grid
        u, v, w = self.state.u, self.state.v, self.state.w
        u, v, w = self._sanitize(u, v, w)

        if self.use_les:
            nu_eff = smagorinsky_viscosity(u, v, w, g, self.Cs, self.nu)
            nu_eff = np.clip(nu_eff, self.nu, 50.0 * self.nu + 1.0)
        else:
            nu_eff = np.full(g.shape, self.nu)
        # floor viscosity for stability on coarse grids
        nu_eff = np.maximum(nu_eff, self.nu)

        u_s = self._advect_diffuse(u, u, v, w, nu_eff, dt)
        v_s = self._advect_diffuse(v, u, v, w, nu_eff, dt)
        w_s = self._advect_diffuse(w, u, v, w, nu_eff, dt)
        u_s, v_s, w_s = self._sanitize(u_s, v_s, w_s)
        u_s, v_s, w_s = self._apply_bc(u_s, v_s, w_s)

        dudx = _diff_central(u_s, g.dx, 0)
        dvdy = _diff_central(v_s, g.dy, 1)
        dwdz = _diff_central(w_s, g.dz, 2)
        div = dudx + dvdy + dwdz
        div[self._obstacle] = 0.0

        p_new = self._pressure_poisson(div, dt)

        dpdx = _diff_central(p_new, g.dx, 0)
        dpdy = _diff_central(p_new, g.dy, 1)
        dpdz = _diff_central(p_new, g.dz, 2)
        u_n = u_s - (dt / self.rho) * dpdx
        v_n = v_s - (dt / self.rho) * dpdy
        w_n = w_s - (dt / self.rho) * dpdz
        u_n, v_n, w_n = self._sanitize(u_n, v_n, w_n)
        u_n, v_n, w_n = self._apply_bc(u_n, v_n, w_n)

        self.state = FluidState(u=u_n, v=v_n, w=w_n, p=p_new, nu_eff=nu_eff)
        return self.state

    def sample_at(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g = self.grid
        pts = np.asarray(points, dtype=float)
        pts = np.nan_to_num(pts, nan=0.0)
        fi = np.clip(pts[:, 0] / g.dx - 0.5, 0.0, g.nx - 1.001)
        fj = np.clip(pts[:, 1] / g.dy - 0.5, 0.0, g.ny - 1.001)
        fk = np.clip(pts[:, 2] / g.dz - 0.5, 0.0, g.nz - 1.001)
        i0 = np.floor(fi).astype(int)
        j0 = np.floor(fj).astype(int)
        k0 = np.floor(fk).astype(int)
        i1 = np.minimum(i0 + 1, g.nx - 1)
        j1 = np.minimum(j0 + 1, g.ny - 1)
        k1 = np.minimum(k0 + 1, g.nz - 1)
        wx, wy, wz = fi - i0, fj - j0, fk - k0

        def trilin(field):
            field = np.nan_to_num(field, nan=0.0)
            c000 = field[i0, j0, k0]
            c100 = field[i1, j0, k0]
            c010 = field[i0, j1, k0]
            c110 = field[i1, j1, k0]
            c001 = field[i0, j0, k1]
            c101 = field[i1, j0, k1]
            c011 = field[i0, j1, k1]
            c111 = field[i1, j1, k1]
            c00 = c000 * (1 - wx) + c100 * wx
            c01 = c001 * (1 - wx) + c101 * wx
            c10 = c010 * (1 - wx) + c110 * wx
            c11 = c011 * (1 - wx) + c111 * wx
            c0 = c00 * (1 - wy) + c10 * wy
            c1 = c01 * (1 - wy) + c11 * wy
            return c0 * (1 - wz) + c1 * wz

        vel = np.column_stack(
            [trilin(self.state.u), trilin(self.state.v), trilin(self.state.w)]
        )
        pressure = trilin(self.state.p)
        return vel, pressure

    def max_cfl(self, dt: float) -> float:
        g = self.grid
        umax = max(
            float(np.max(np.abs(self.state.u))),
            float(np.max(np.abs(self.state.v))),
            float(np.max(np.abs(self.state.w))),
            1e-12,
        )
        hmin = min(g.dx, g.dy, g.dz)
        return umax * dt / hmin
