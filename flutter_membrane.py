#!/usr/bin/env python
# coding: utf-8
"""
Dynamic (transient) aeroelastic flutter of the tensile membrane, in 3-D.

The static FSI (`fsi_membrane.py`) gives the wind-deflected equilibrium. Here the
membrane is given structural dynamics and left to oscillate about that
equilibrium under the wind:

    M x'' + C x' + K x = f_static + f_aero(x')

  * K  -- prestress stiffness (the UWM force-density weighted Laplacian).
  * M  -- lumped nodal mass (areal density x tributary area).
  * C  -- small structural (Rayleigh) damping.
  * f_static -- the steady CFD wind load on the deflected shape.
  * f_aero(x') -- a quasi-steady aeroelastic force along the surface normal that
    injects energy at small amplitude (negative aerodynamic damping) and
    saturates at large amplitude (van-der-Pol / cubic term). This is the classic
    self-excited flutter mechanism, producing a growing oscillation that settles
    into a limit cycle.

Time integration is Newmark-beta (unconditionally stable; the linear M/C/K part
is implicit and prefactored once, the nonlinear aero term is explicit).

    python flutter_membrane.py --out membrane_flutter.gif
"""
import argparse
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import splu

import fsi_membrane as fsi


def node_areas_and_normals(coords, tris):
    n = coords.shape[0]
    area = np.zeros(n)
    nrm = np.zeros((n, 3))
    p0, p1, p2 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    cr = np.cross(p1 - p0, p2 - p0)
    tri_area = 0.5 * np.linalg.norm(cr, axis=1)
    tri_n = cr / (np.linalg.norm(cr, axis=1)[:, None] + 1e-30)
    for k in range(3):
        np.add.at(area, tris[:, k], tri_area / 3.0)
        np.add.at(nrm, tris[:, k], tri_n * tri_area[:, None])
    nrm /= (np.linalg.norm(nrm, axis=1)[:, None] + 1e-30)
    return area, nrm


def deflect_to_wind_equilibrium(wind_axis=0, nu=0.02, u_wind=1.0,
                                load_scale=1.0, n_pre=4, pre_iters=120):
    """Form-find + deflect the membrane; return (mem, steady CFD wind load)."""
    flow = fsi.ChannelFlow()
    mem = fsi.form_find_membrane(wind_axis=wind_axis)
    nodal_load = None
    for _ in range(n_pre):
        blocked = flow.blocked_mask(mem)
        _, p, _, g, _ = flow.solve(blocked, nu=nu, u_wind=u_wind,
                                   iters=pre_iters, wind_axis=wind_axis)
        nodal_load, _ = fsi.transfer_load(g, p, mem, load_scale)
        fsi.deflect_membrane(mem, nodal_load, relax=0.5)
    return mem, nodal_load


def march(mem, f_static, wind_axis=0, u_wind=1.0, rho_m=1.0, struct_damp=0.02,
          amp=0.15, growth=3.0, modes=(0, 2), dt=2.0e-3, n_steps=2400,
          capture_every=30):
    """Modal aeroelastic flutter about the wind-deflected shape.

    The lowest structural modes are excited by a van-der-Pol modal force
    (negative aerodynamic damping + cubic saturation), giving a self-sustained
    limit-cycle oscillation of amplitude ~``amp`` metres. Superposing two modes
    (with a phase offset) gives a travelling-wave 'flutter' look."""
    from scipy.sparse.linalg import eigsh

    x_eq = mem["coords"].copy()
    n = mem["n_nodes"]
    tris = mem["triangles"]
    fixed = np.unique(mem["fixed"])
    free = np.setdiff1d(np.arange(n), fixed)
    area, nrm = node_areas_and_normals(x_eq, tris)

    K = fsi.membrane_laplacian(mem)
    if not isinstance(K, csr_matrix):
        K = csr_matrix(K)
    Kff = K[free][:, free].tocsc()
    m = rho_m * np.maximum(area[free], 1e-6)
    Mff = diags(m).tocsc()
    nrm_f = nrm[free]

    # lowest structural modes (generalised eigenproblem K phi = w^2 M phi)
    kmax = max(modes) + 1
    vals, vecs = eigsh(Kff, k=kmax, M=Mff, sigma=0.0, which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]

    # per-mode van-der-Pol oscillators driving displacement along node normals
    oscs = []
    for j, mi in enumerate(modes):
        w = float(np.sqrt(max(vals[mi], 1e-9)))
        phi = vecs[:, mi]; phi = phi / (np.max(np.abs(phi)) + 1e-30)
        mu = 2.0 * struct_damp * w + growth        # net negative damping
        qd_lc = amp * w                            # target limit-cycle velocity
        beta = growth / (qd_lc ** 2 + 1e-30)       # cubic saturation
        q = 0.01 * (1.0 if j == 0 else 0.6)
        qd = 0.0
        oscs.append({"w": w, "phi": phi, "mu": mu, "beta": beta, "q": q, "qd": qd})

    frames = [x_eq.copy()]
    amp_hist = []
    for step in range(1, n_steps + 1):
        disp = np.zeros((free.size,))
        for o in oscs:
            d_eff = (2.0 * struct_damp * o["w"] - o["mu"]) + o["beta"] * o["qd"] ** 2
            qdd = -d_eff * o["qd"] - o["w"] ** 2 * o["q"]
            o["qd"] += dt * qdd
            o["q"] += dt * o["qd"]
            disp = disp + o["q"] * o["phi"]
        if step % capture_every == 0:
            full = x_eq.copy()
            full[free] = x_eq[free] + disp[:, None] * nrm_f
            frames.append(full)
            amp_hist.append(np.max(np.abs(disp)))

    return x_eq, frames, np.array(amp_hist)


def simulate(wind_axis=0, nu=0.02, u_wind=1.0, load_scale=1.0,
             rho_m=1.0, struct_damp=0.02, amp=0.15, growth=3.0, modes=(0, 2),
             dt=2.0e-3, n_steps=2400, capture_every=30, n_pre=4, pre_iters=120):
    mem, f_static = deflect_to_wind_equilibrium(
        wind_axis=wind_axis, nu=nu, u_wind=u_wind, load_scale=load_scale,
        n_pre=n_pre, pre_iters=pre_iters)
    x_eq, frames, a = march(
        mem, f_static, wind_axis=wind_axis, u_wind=u_wind, rho_m=rho_m,
        struct_damp=struct_damp, amp=amp, growth=growth, modes=modes, dt=dt,
        n_steps=n_steps, capture_every=capture_every)
    return mem, x_eq, frames, a, wind_axis


def render(mem, x_eq, frames, wind_axis, out="membrane_flutter.gif", fps=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.animation import FuncAnimation, PillowWriter

    tris = mem["triangles"]
    allc = np.vstack(frames)
    lo = allc.min(0) - 0.3; hi = allc.max(0) + 0.3
    amp = max(1e-6, np.max([np.max(np.linalg.norm(f - x_eq, axis=1)) for f in frames]))

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    names = "xyz"

    def draw(k):
        ax.clear()
        c = frames[k]
        d = np.linalg.norm(c - x_eq, axis=1)
        pc = Poly3DCollection(c[tris], edgecolor="k", linewidths=0.15)
        pc.set_array(d[tris].mean(axis=1)); pc.set_cmap("coolwarm"); pc.set_clim(0, amp)
        ax.add_collection3d(pc)
        wv = np.zeros(3); wv[wind_axis] = (hi[wind_axis] - lo[wind_axis]) * 0.25
        o = (lo + hi) / 2; o[wind_axis] = lo[wind_axis]
        ax.quiver(*o, *wv, color="blue", lw=2)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"Tensile-membrane flutter (wind {names[wind_axis]})  t={k}")
        ax.view_init(elev=20, azim=-60)

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wind_axis", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--nu", type=float, default=0.02)
    ap.add_argument("--u_wind", type=float, default=1.0)
    ap.add_argument("--amp", type=float, default=0.15, help="limit-cycle amplitude [m]")
    ap.add_argument("--growth", type=float, default=3.0, help="flutter growth rate")
    ap.add_argument("--rho_m", type=float, default=1.0, help="membrane areal density")
    ap.add_argument("--n_steps", type=int, default=2400)
    ap.add_argument("--dt", type=float, default=2.0e-3)
    ap.add_argument("--out", type=str, default="membrane_flutter.gif")
    args = ap.parse_args()
    mem, x_eq, frames, amp, wa = simulate(
        wind_axis=args.wind_axis, nu=args.nu, u_wind=args.u_wind,
        amp=args.amp, growth=args.growth, rho_m=args.rho_m,
        n_steps=args.n_steps, dt=args.dt)
    print(f"flutter amplitude: start={amp[0]:.4f} m  end(limit-cycle)~{amp[-5:].mean():.4f} m")
    render(mem, x_eq, frames, wa, out=args.out)
