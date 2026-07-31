#!/usr/bin/env python
# coding: utf-8
"""
LES_smagorinksy -- incompressible flow past a sphere.

A cell-centred (collocated) unstructured finite-volume solver for the steady
incompressible Navier-Stokes equations, using the SIMPLE algorithm with
implicit, under-relaxed momentum, Rhie-Chow face-flux interpolation for
pressure-velocity coupling, and the Smagorinsky sub-grid-scale (LES)
turbulence model for the effective viscosity.

Mesh comes from ``processed_mesh.npz`` (produced by
``read_mesh_from_the_generated_mesh.ipynb``): a 10 x 5 x 5 m channel with a
sphere of radius 0.5 m centred at (3, 2.5, 2.5).

Boundary conditions
-------------------
    inlet  (x = 0)  : Dirichlet velocity  u = (U, 0, 0),  dp/dn = 0
    outlet (x = L)  : convective outflow (du/dn = 0),      p = 0
    walls  (box)    : free-slip (no penetration, no shear)
    object (sphere) : no-slip  u = 0,                       dp/dn = 0

Run:
    python les_solver.py --iters 400 --nu 0.02
"""
import argparse
import time
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg, bicgstab, LinearOperator

L, W, H = 10.0, 5.0, 5.0
SPHERE_CENTRE = np.array([3.0, 2.5, 2.5])
SPHERE_R = 0.5
INLET, OUTLET, WALL, OBJECT = 0, 1, 2, 3


def load_mesh(path="processed_mesh.npz"):
    d = np.load(path, allow_pickle=True)
    return {
        "cell_centroids": d["cell_centroids"],
        "cell_volumes": d["cell_volumes"],
        "owner": d["owner"].astype(np.int64),
        "neighbour": d["neighbour"].astype(np.int64),
        "face_centroids": d["face_centroids"],
        "face_area_vectors": d["face_area_vectors"],
        "face_areas": d["face_areas"],
        "n_cells": len(d["cell_volumes"]),
        "n_faces": len(d["owner"]),
    }


def classify_boundaries(mesh, wind_axis=0):
    """Inlet/outlet are the min/max faces along ``wind_axis``; other box faces are
    free-slip walls; faces on the sphere (if present) are the no-slip object."""
    fc = mesh["face_centroids"]
    b = np.where(mesh["neighbour"] == -1)[0]
    coord = fc[b, wind_axis]
    cmin, cmax = coord.min(), coord.max()
    dist_sphere = np.linalg.norm(fc[b] - SPHERE_CENTRE, axis=1)
    btype = np.full(len(b), WALL, dtype=np.int64)
    btype[np.abs(coord - cmin) < 1e-2] = INLET
    btype[np.abs(coord - cmax) < 1e-2] = OUTLET
    btype[dist_sphere < (SPHERE_R + 0.3)] = OBJECT
    return b, btype


def build_geometry(mesh, blocked_internal=None, wind_axis=0):
    """Build FV geometry. ``blocked_internal`` is an optional boolean mask over the
    internal faces (in the order returned by ``np.where(neighbour != -1)``); those
    faces are turned into a thin no-slip immersed baffle (a membrane): each becomes
    a no-slip OBJECT wall for BOTH adjacent cells and is removed from the internal
    (through-flow / pressure) coupling."""
    owner, neigh = mesh["owner"], mesh["neighbour"]
    cc, fc = mesh["cell_centroids"], mesh["face_centroids"]
    Sf, fa = mesh["face_area_vectors"], mesh["face_areas"]

    internal = np.where(neigh != -1)[0]
    P, N = owner[internal], neigh[internal]
    d_len = np.linalg.norm(cc[N] - cc[P], axis=1)
    dP = np.linalg.norm(fc[internal] - cc[P], axis=1)
    dN = np.linalg.norm(fc[internal] - cc[N], axis=1)
    wP = dN / (dP + dN)
    a_geom = fa[internal] / d_len

    b, btype = classify_boundaries(mesh, wind_axis=wind_axis)
    b_owner = owner[b]; b_Sf = Sf[b]; b_fa = fa[b]
    b_db = np.linalg.norm(fc[b] - cc[b_owner], axis=1)

    g = {}
    if blocked_internal is not None and np.any(blocked_internal):
        blk = np.asarray(blocked_internal, dtype=bool)
        keep = ~blk
        iP, iN = P[blk], N[blk]
        iSf, ifc, ifa = Sf[internal][blk], fc[internal][blk], fa[internal][blk]
        # membrane face map (for pressure-jump load transfer): normal points P->N
        g["mem_P"], g["mem_N"] = iP, iN
        g["mem_Sf"], g["mem_fc"], g["mem_fa"] = iSf.copy(), ifc, ifa
        # append two no-slip OBJECT boundary faces per blocked face (one per side)
        db_P = np.linalg.norm(ifc - cc[iP], axis=1)
        db_N = np.linalg.norm(ifc - cc[iN], axis=1)
        b_owner = np.concatenate([b_owner, iP, iN])
        b_Sf = np.concatenate([b_Sf, iSf, -iSf])            # outward from each side
        b_fa = np.concatenate([b_fa, ifa, ifa])
        b_db = np.concatenate([b_db, db_P, db_N])
        btype = np.concatenate([btype, np.full(2 * blk.sum(), OBJECT, dtype=np.int64)])
        # restrict internal set to unblocked faces
        P, N = P[keep], N[keep]
        Sf_int = Sf[internal][keep]; wP = wP[keep]; a_geom = a_geom[keep]
    else:
        Sf_int = Sf[internal]

    g.update({"internal": internal, "P": P, "N": N, "Sf_int": Sf_int,
              "wP": wP, "a_geom": a_geom,
              "b": b, "btype": btype, "b_owner": b_owner, "b_Sf": b_Sf,
              "b_fa": b_fa, "b_db": b_db})
    g["b_a_geom"] = g["b_fa"] / g["b_db"]
    return g


def green_gauss(phi, mesh, g, phi_bc):
    """Green-Gauss cell gradient. phi:(n,) scalar -> grad (n,3)."""
    n = mesh["n_cells"]
    cv = mesh["cell_volumes"]
    P, N, Sf, wP = g["P"], g["N"], g["Sf_int"], g["wP"]
    phi_f = wP * phi[P] + (1 - wP) * phi[N]
    contrib = phi_f[:, None] * Sf
    grad = np.zeros((n, 3))
    np.add.at(grad, P, contrib)
    np.add.at(grad, N, -contrib)
    np.add.at(grad, g["b_owner"], phi_bc[:, None] * g["b_Sf"])
    return grad / cv[:, None]


def velocity_gradient(u, mesh, g, U_in):
    """Cell velocity gradient tensor grad_u[c,i,j] = du_i/dx_j (for LES)."""
    n = mesh["n_cells"]
    cv = mesh["cell_volumes"]
    P, N, Sf, wP = g["P"], g["N"], g["Sf_int"], g["wP"]
    btype, b_owner = g["btype"], g["b_owner"]
    ub = np.empty((len(btype), 3))
    ub[btype == INLET] = U_in
    ub[btype == OBJECT] = 0.0
    zg = (btype == OUTLET) | (btype == WALL)
    ub[zg] = u[b_owner[zg]]
    grad = np.zeros((n, 3, 3))
    uf = wP[:, None] * u[P] + (1 - wP)[:, None] * u[N]
    for i in range(3):
        contrib = uf[:, i][:, None] * Sf
        np.add.at(grad[:, i, :], P, contrib)
        np.add.at(grad[:, i, :], N, -contrib)
        np.add.at(grad[:, i, :], b_owner, ub[:, i][:, None] * g["b_Sf"])
    return grad / cv[:, None, None]


def smagorinsky_nu_t(u, mesh, g, U_in, Cs=0.17):
    grad_u = velocity_gradient(u, mesh, g, U_in)
    S = 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))
    S_mag = np.sqrt(2.0 * np.sum(S * S, axis=(1, 2)))
    delta = np.cbrt(mesh["cell_volumes"])
    return (Cs * delta) ** 2 * S_mag


def run(nu=0.02, U=1.0, rho=1.0, iters=400, alpha_u=0.7, alpha_p=0.3,
        Cs=0.17, beta=1.0, limiter="vanleer", avg_last=None, tol=1e-4,
        mesh_path="processed_mesh.npz", out_path="les_result.npz", log_every=10,
        blocked_internal=None, mesh=None, wind_axis=0,
        callback=None, snap_every=0):
    t0 = time.time()
    if mesh is None:
        mesh = load_mesh(mesh_path)
    g = build_geometry(mesh, blocked_internal=blocked_internal, wind_axis=wind_axis)
    n = mesh["n_cells"]
    cv = mesh["cell_volumes"]
    P, N = g["P"], g["N"]
    Sf = g["Sf_int"]
    wP = g["wP"]
    a_geom = g["a_geom"]
    btype = g["btype"]
    b_owner = g["b_owner"]
    b_Sf = g["b_Sf"]
    b_a_geom = g["b_a_geom"]

    inlet_m = btype == INLET
    outlet_m = btype == OUTLET
    wall_m = btype == WALL
    object_m = btype == OBJECT
    U_in = np.zeros(3); U_in[wind_axis] = U

    print(f"Mesh: {n} cells, {mesh['n_faces']} faces "
          f"(internal {len(P)}, boundary {len(btype)})")
    print(f"Boundaries -> inlet {inlet_m.sum()}, outlet {outlet_m.sum()}, "
          f"walls {wall_m.sum()}, object {object_m.sum()}")
    print(f"nu={nu}  Re(D=1)={U/nu:.0f}  iters={iters}  alpha_u={alpha_u} alpha_p={alpha_p}")

    # boundary Dirichlet mass fluxes (constant)
    F_in = np.sum(U_in * b_Sf[inlet_m], axis=1)     # < 0 (inflow)

    def pressure_bc(pc):
        return np.where(outlet_m, 0.0, pc[b_owner])

    def cell_divergence(Fint, Fb):
        div = np.zeros(n)
        np.add.at(div, P, Fint)
        np.add.at(div, N, -Fint)
        np.add.at(div, b_owner, Fb)
        return div

    # COO index scaffolding for the momentum matrix (structure is constant)
    rows_int = np.concatenate([P, N, P, N])
    cols_int = np.concatenate([P, N, N, P])

    # initial fields
    u = np.tile(U_in, (n, 1)).astype(np.float64)
    p = np.zeros(n)
    # initial face mass flux from free stream
    uf0 = wP[:, None] * u[P] + (1 - wP)[:, None] * u[N]
    Fint = np.sum(uf0 * Sf, axis=1)
    Fb = np.zeros(len(btype))
    Fb[inlet_m] = F_in

    cc = mesh["cell_centroids"]
    d_PN = cc[N] - cc[P]

    # LES-style averaging window: report the mean field over the last iterations
    # (removes any residual limit-cycle oscillation of the high-order scheme).
    if avg_last is None:
        avg_last = max(1, iters // 3)
    avg_start = iters - avg_last
    u_acc = np.zeros((n, 3)); p_acc = np.zeros(n); n_acc = 0

    hist = []
    for it in range(1, iters + 1):
        # velocity gradient (reused for LES eddy viscosity and the TVD limiter)
        grad_u = velocity_gradient(u, mesh, g, U_in)
        S = 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))
        S_mag = np.sqrt(2.0 * np.sum(S * S, axis=(1, 2)))
        nu_t = (Cs * np.cbrt(cv)) ** 2 * S_mag
        nu_eff = nu + nu_t
        nuf = wP * nu_eff[P] + (1 - wP) * nu_eff[N]

        # ---------------- momentum matrix (implicit upwind + diffusion) ----------------
        D_f = nuf * a_geom
        Fpos = np.maximum(Fint, 0.0)
        Fneg = np.maximum(-Fint, 0.0)
        aN_P = D_f + Fneg          # coeff of neighbour in owner row
        aP_N = D_f + Fpos          # coeff of owner in neighbour row
        diag = np.zeros(n)
        np.add.at(diag, P, D_f + Fpos)
        np.add.at(diag, N, D_f + Fneg)

        # boundary diagonal + explicit source contributions
        b_src = np.zeros((n, 3))
        # inlet (Dirichlet U_in): D_b + inflow
        D_in = nu_eff[b_owner[inlet_m]] * b_a_geom[inlet_m]
        np.add.at(diag, b_owner[inlet_m], D_in + np.maximum(F_in, 0.0))
        coef_in = D_in + np.maximum(-F_in, 0.0)
        np.add.at(b_src, b_owner[inlet_m], coef_in[:, None] * U_in)
        # object (no-slip u=0)
        D_ob = nu_eff[b_owner[object_m]] * b_a_geom[object_m]
        np.add.at(diag, b_owner[object_m], D_ob)
        # outlet (convective outflow)
        F_out = np.sum(u[b_owner[outlet_m]] * b_Sf[outlet_m], axis=1)
        np.add.at(diag, b_owner[outlet_m], np.maximum(F_out, 0.0))
        # walls: slip -> nothing

        # deferred-correction higher-order convection to cut the false diffusion
        # of first-order upwind (so the recirculating wake can form).
        #   limiter='none'    -> central differencing (sharpest, may ring slightly)
        #   limiter='vanleer' -> bounded (TVD), most robust, more diffusive
        #   limiter='superbee'-> bounded (TVD), sharpest of the limited schemes
        if beta > 0.0:
            fpos = Fint >= 0
            up = np.where(fpos, P, N)               # upwind cell per face
            dn = np.where(fpos, N, P)               # downwind cell per face
            u_up = np.where(fpos[:, None], u[P], u[N])
            delta = u[dn] - u[up]                   # (nf,3)
            if limiter == "none":
                u_ho = wP[:, None] * u[P] + (1 - wP)[:, None] * u[N]   # central
                ho = u_ho - u_up
            else:
                dvec = np.where(fpos[:, None], d_PN, -d_PN)            # upwind->downwind
                gdotd = np.einsum("fij,fj->fi", grad_u[up], dvec)
                safe = np.abs(delta) > 1e-12
                r = np.where(safe, 2.0 * gdotd / np.where(safe, delta, 1.0) - 1.0, 0.0)
                if limiter == "superbee":
                    psi = np.maximum.reduce([np.zeros_like(r),
                                             np.minimum(2.0 * r, 1.0),
                                             np.minimum(r, 2.0)])
                else:  # van Leer
                    psi = (r + np.abs(r)) / (1.0 + np.abs(r))
                ho = np.where(safe, 0.5 * psi * delta, 0.0)
            corr = beta * Fint[:, None] * ho
            np.add.at(b_src, P, -corr)
            np.add.at(b_src, N, corr)

        diag_ur = diag / alpha_u                      # under-relaxed diagonal
        # pressure gradient source
        gradp = green_gauss(p, mesh, g, pressure_bc(p))
        b_src += -gradp * cv[:, None]
        # under-relaxation deferred source
        b_src += (diag_ur - diag)[:, None] * u

        vals_int = np.concatenate([np.full(len(P), 0.0), np.full(len(N), 0.0),
                                   -aN_P, -aP_N])
        rows = np.concatenate([rows_int, np.arange(n)])
        cols = np.concatenate([cols_int, np.arange(n)])
        vals = np.concatenate([vals_int, diag_ur])
        A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

        u_prev = u.copy()
        Aprec = LinearOperator((n, n), matvec=lambda x, dd=diag_ur: x / dd)
        u_new = np.empty_like(u)
        for i in range(3):
            sol, _ = bicgstab(A, b_src[:, i], x0=u[:, i], rtol=1e-3, maxiter=200, M=Aprec)
            u_new[:, i] = sol

        # ---------------- Rhie-Chow face flux ----------------
        D_P = cv / diag_ur                            # V / a_P
        D_face = wP * D_P[P] + (1 - wP) * D_P[N]
        gradp = green_gauss(p, mesh, g, pressure_bc(p))
        uf = wP[:, None] * u_new[P] + (1 - wP)[:, None] * u_new[N]
        gpf = wP[:, None] * gradp[P] + (1 - wP)[:, None] * gradp[N]
        Fint_s = np.sum(uf * Sf, axis=1) - D_face * (a_geom * (p[N] - p[P])
                                                     - np.sum(gpf * Sf, axis=1))
        Fb_s = np.zeros(len(btype))
        Fb_s[inlet_m] = F_in
        Fb_s[outlet_m] = np.sum(u_new[b_owner[outlet_m]] * b_Sf[outlet_m], axis=1)

        # ---------------- pressure correction ----------------
        c_f = D_face * a_geom
        c_out = D_P[b_owner[outlet_m]] * b_a_geom[outlet_m]
        # build Lp: graph Laplacian with coeff c_f, plus outlet Dirichlet diagonal
        rP = np.concatenate([P, N, P, N])
        cP = np.concatenate([P, N, N, P])
        vP = np.concatenate([c_f, c_f, -c_f, -c_f])
        rP = np.concatenate([rP, b_owner[outlet_m]])
        cP = np.concatenate([cP, b_owner[outlet_m]])
        vP = np.concatenate([vP, c_out])
        Lp = coo_matrix((vP, (rP, cP)), shape=(n, n)).tocsr()
        Lp_diag = Lp.diagonal()
        Pprec = LinearOperator((n, n), matvec=lambda x, dd=Lp_diag: x / dd)
        rhs_p = -cell_divergence(Fint_s, Fb_s)
        pcorr, _ = cg(Lp, rhs_p, rtol=1e-5, maxiter=2000, M=Pprec)

        # ---------------- corrections ----------------
        Fint = Fint_s - c_f * (pcorr[N] - pcorr[P])
        Fb = Fb_s.copy()
        Fb[outlet_m] = Fb_s[outlet_m] - c_out * (0.0 - pcorr[b_owner[outlet_m]])

        gradpc = green_gauss(pcorr, mesh, g, pressure_bc(pcorr))
        u = u_new - D_P[:, None] * gradpc
        p = p + alpha_p * pcorr

        if it > avg_start:
            u_acc += u; p_acc += p; n_acc += 1

        if callback is not None and snap_every > 0 and (it == 1 or it % snap_every == 0):
            callback(it, u.copy(), p.copy())

        # ---------------- diagnostics ----------------
        du_max = np.max(np.abs(u - u_prev)) / U       # steadiness metric
        if it % log_every == 0 or it == 1:
            cont = np.max(np.abs(cell_divergence(Fint, Fb)))
            umax = np.max(np.linalg.norm(u, axis=1))
            hist.append((it, cont, umax, float(nu_t.max()), du_max))
            print(f"iter {it:4d}  d|u|/U={du_max:.3e}  max|div|={cont:.2e}  "
                  f"umax={umax:.3f}  nu_t_max={nu_t.max():.2e}")
            if not np.isfinite(umax):
                print("DIVERGED -- aborting")
                break
        if du_max < tol and it > 20:
            print(f"Converged (velocity change < {tol}) at iter {it}")
            break

    elapsed = time.time() - t0
    # mean (time-averaged) fields -- the reported LES result
    if n_acc > 0:
        u_mean = u_acc / n_acc
        p_mean = p_acc / n_acc
    else:
        u_mean, p_mean = u, p
    print(f"Done in {elapsed:.1f}s  (mean field averaged over last {n_acc} iterations)")
    np.savez(out_path, velocity=u_mean, pressure=p_mean,
             velocity_inst=u, pressure_inst=p,
             cell_centroids=mesh["cell_centroids"], cell_volumes=cv,
             nu=nu, U=U, iters=it, history=np.array(hist))
    print(f"Saved {out_path}")
    return u_mean, p_mean, mesh, g, np.array(hist)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=0.02)
    ap.add_argument("--U", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--alpha_u", type=float, default=0.7)
    ap.add_argument("--alpha_p", type=float, default=0.3)
    ap.add_argument("--Cs", type=float, default=0.17)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="deferred-correction blend (0=upwind, 1=full high-order)")
    ap.add_argument("--limiter", type=str, default="vanleer",
                    choices=["none", "vanleer", "superbee"])
    ap.add_argument("--avg_last", type=int, default=None,
                    help="average the reported field over the last N iterations")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--mesh", type=str, default="processed_mesh.npz")
    ap.add_argument("--out", type=str, default="les_result.npz")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--wind_axis", type=int, default=0, choices=[0, 1, 2])
    args = ap.parse_args()
    run(nu=args.nu, U=args.U, iters=args.iters, alpha_u=args.alpha_u,
        alpha_p=args.alpha_p, Cs=args.Cs, beta=args.beta, limiter=args.limiter,
        avg_last=args.avg_last, tol=args.tol, mesh_path=args.mesh,
        out_path=args.out, log_every=args.log_every, wind_axis=args.wind_axis)
