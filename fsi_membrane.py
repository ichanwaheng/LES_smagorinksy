#!/usr/bin/env python
# coding: utf-8
"""
Fluid-structure interaction (FSI): wind load on a tensile membrane.

Couples the two projects:
  * `les_solver.py`   -- incompressible LES (Smagorinsky) finite-volume flow solver.
  * `uwm_membrane.py` -- Updated-Weight-Method tensile-membrane form-finding.

Workflow (partitioned, two-way FSI on a fixed background CFD mesh):
  1. Form-find a hypar membrane "canopy" (UWM) and place it across the channel,
     facing the wind (single-valued surface x = X(y, z)).
  2. Immerse it in the LES channel flow as a thin no-slip baffle: internal CFD
     faces the membrane surface crosses are flagged as blocked (no-slip) walls.
  3. Run the LES flow -> pressure field. The net pressure force on each blocked
     face, (p[P] - p[N]) * Sf, is the aerodynamic (wind) load.
  4. Transfer that load to the membrane nodes and solve the loaded membrane
     equilibrium (force-density with the prestress stiffness) -> the membrane
     deflects downwind.
  5. Re-blank with the deflected shape and repeat (under-relaxed) until the
     membrane deflection converges.

    python fsi_membrane.py --n_fsi 5 --u_wind 1.0 --load_scale 8
"""
import argparse
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import les_solver
import uwm_membrane as uwm

CHANNEL_MESH = "channel_mesh.npz"


# --------------------------------------------------------------------------- #
# Membrane
# --------------------------------------------------------------------------- #
def form_find_membrane(span=3.0, rise=1.5, n=16, X0=5.0, Y0=1.0, Z0=1.0,
                       sigma=3.0, cable=30.0, iters=25):
    """Form-find a hypar canopy and place it in the channel facing +x wind.

    The UWM hypar is built over a local square; its local 'height' becomes the
    global x (windward/leeward bulge) and the square footprint maps to the (y, z)
    channel cross-section, so the surface is single-valued in x = X(y, z)."""
    coords, tris, cables, corners, _ = uwm.build_structured_rectangle_mesh(
        span, span, n, n)
    lx, ly = coords[:, 0], coords[:, 1]
    # local hypar height field (saddle) in [0, rise]
    lz = rise * (lx / span) * (1 - ly / span) + rise * (1 - lx / span) * (ly / span)
    coords[:, 2] = lz
    corner_ids = np.unique([int(v[0]) for v in corners.values()])

    s = uwm.UWMSettings(target_sigma_fill=sigma, target_sigma_warp=sigma,
                        target_cable_force=cable, max_outer_iterations=iters,
                        stress_tolerance=1e-4)
    res = uwm.run_uwm(coords.copy(), tris, cables, corner_ids, s,
                      np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    fc = res["coords"]
    # map local (lx, ly, lz) -> global (x, y, z): height -> x, footprint -> (y, z)
    glob = np.zeros_like(fc)
    glob[:, 0] = X0 + (fc[:, 2] - rise * 0.5)
    glob[:, 1] = Y0 + fc[:, 0]
    glob[:, 2] = Z0 + fc[:, 1]

    mem = {
        "coords0": glob.copy(), "coords": glob.copy(),
        "triangles": tris, "cables": cables, "fixed": corner_ids,
        "membrane_weights": res["membrane_weights"],
        "cable_weights": res["cable_weights"],
        "n_nodes": glob.shape[0],
    }
    return mem


def membrane_laplacian(mem):
    L = uwm.assemble_weighted_laplacian(
        mem["n_nodes"], mem["triangles"], mem["membrane_weights"],
        mem["cables"], mem["cable_weights"])
    if not isinstance(L, csr_matrix):
        L = csr_matrix(L)
    return L


def deflect_membrane(mem, nodal_load, relax=1.0):
    """Solve the loaded force-density equilibrium  L x = f  (corners fixed)."""
    L = membrane_laplacian(mem)
    n = mem["n_nodes"]
    fixed = np.unique(mem["fixed"])
    free = np.setdiff1d(np.arange(n), fixed)
    L_ff = L[free][:, free]
    L_fb = L[free][:, fixed]
    x = mem["coords0"].copy()          # reference (form-found) shape
    new = mem["coords0"].copy()
    for d in range(3):
        rhs = nodal_load[free, d] - L_fb @ x[fixed, d]
        new[free, d] = spsolve(L_ff, rhs)
    # under-relax the geometry update between FSI iterations
    mem["coords"] = (1 - relax) * mem["coords"] + relax * new
    return mem["coords"]


# --------------------------------------------------------------------------- #
# Fluid <-> structure coupling
# --------------------------------------------------------------------------- #
def ensure_channel_mesh(mesh_path=CHANNEL_MESH):
    import os
    if not os.path.exists(mesh_path):
        print(f"{mesh_path} not found -- generating channel mesh with gmsh ...")
        import channel_mesh
        channel_mesh.generate(out_npz=mesh_path)
    return mesh_path


class ChannelFlow:
    def __init__(self, mesh_path=CHANNEL_MESH):
        ensure_channel_mesh(mesh_path)
        self.mesh = les_solver.load_mesh(mesh_path)
        neigh = self.mesh["neighbour"]
        owner = self.mesh["owner"]
        self.internal = np.where(neigh != -1)[0]
        self.P = owner[self.internal]
        self.N = neigh[self.internal]
        self.cc = self.mesh["cell_centroids"]
        self.fc_int = self.mesh["face_centroids"][self.internal]

    def blocked_mask(self, mem):
        """Internal faces crossed by the (single-valued x=X(y,z)) membrane surface."""
        c = mem["coords"]
        interp = LinearNDInterpolator(c[:, 1:3], c[:, 0])   # (y,z) -> x
        XP = interp(self.cc[self.P, 1], self.cc[self.P, 2])
        XN = interp(self.cc[self.N, 1], self.cc[self.N, 2])
        valid = ~(np.isnan(XP) | np.isnan(XN))
        sP = self.cc[self.P, 0] - np.where(np.isnan(XP), 0.0, XP)
        sN = self.cc[self.N, 0] - np.where(np.isnan(XN), 0.0, XN)
        return valid & (np.sign(sP) != np.sign(sN)) & (sP != sN)

    def solve(self, blocked, nu=0.02, u_wind=1.0, iters=150, **kw):
        return les_solver.run(mesh=self.mesh, blocked_internal=blocked,
                              nu=nu, U=u_wind, iters=iters,
                              limiter="vanleer", alpha_u=0.5, alpha_p=0.3,
                              avg_last=max(1, iters // 3), tol=1e-4,
                              out_path="/tmp/fsi_flow.npz", log_every=max(iters, 1),
                              **kw)


def transfer_load(g, p, mem, load_scale):
    """Net pressure force per blocked face -> membrane nodal load (nearest node)."""
    Fface = (p[g["mem_P"]] - p[g["mem_N"]])[:, None] * g["mem_Sf"]   # (nblk, 3)
    tree = cKDTree(mem["coords"][:, 1:3])
    _, node = tree.query(g["mem_fc"][:, 1:3])
    f = np.zeros((mem["n_nodes"], 3))
    np.add.at(f, node, Fface)
    return load_scale * f, Fface.sum(axis=0)


# --------------------------------------------------------------------------- #
# FSI driver
# --------------------------------------------------------------------------- #
def run_fsi(n_fsi=5, nu=0.02, u_wind=1.0, load_scale=8.0, relax=0.5,
            flow_iters=150, out="fsi_result.npz"):
    flow = ChannelFlow()
    mem = form_find_membrane()
    print(f"Membrane: {mem['n_nodes']} nodes, {len(mem['triangles'])} triangles, "
          f"span in y,z; form-found prestress ~3 kN/m")
    print(f"FSI: n_fsi={n_fsi}, u_wind={u_wind}, load_scale={load_scale}, relax={relax}")

    history = []
    p = None
    g = None
    for k in range(1, n_fsi + 1):
        blocked = flow.blocked_mask(mem)
        u, p, mesh, g, _ = flow.solve(blocked, nu=nu, u_wind=u_wind, iters=flow_iters)
        nodal_load, net_F = transfer_load(g, p, mem, load_scale)

        x_before = mem["coords"].copy()
        deflect_membrane(mem, nodal_load, relax=relax)
        # deflection of the free surface relative to the form-found reference
        free = np.setdiff1d(np.arange(mem["n_nodes"]), np.unique(mem["fixed"]))
        defl = np.linalg.norm(mem["coords"] - mem["coords0"], axis=1)
        move = np.linalg.norm(mem["coords"] - x_before, axis=1).max()
        max_defl = defl.max()
        history.append((k, int(blocked.sum()), float(net_F[0]), float(max_defl), float(move)))
        print(f"[FSI {k}] blocked_faces={int(blocked.sum())}  net_Fx={net_F[0]:.3f}  "
              f"max_deflection={max_defl:.3f} m  update={move:.3e} m")

    np.savez(out,
             mem_coords0=mem["coords0"], mem_coords=mem["coords"],
             triangles=mem["triangles"], cables=mem["cables"],
             fixed=mem["fixed"],
             cell_centroids=mesh["cell_centroids"], velocity=u, pressure=p,
             history=np.array(history),
             u_wind=u_wind, nu=nu, load_scale=load_scale)
    print(f"Saved {out}")
    return mem, u, p, mesh, g, np.array(history)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_fsi", type=int, default=5)
    ap.add_argument("--nu", type=float, default=0.02)
    ap.add_argument("--u_wind", type=float, default=1.0)
    ap.add_argument("--load_scale", type=float, default=8.0)
    ap.add_argument("--relax", type=float, default=0.5)
    ap.add_argument("--flow_iters", type=int, default=150)
    ap.add_argument("--out", type=str, default="fsi_result.npz")
    args = ap.parse_args()
    run_fsi(n_fsi=args.n_fsi, nu=args.nu, u_wind=args.u_wind,
            load_scale=args.load_scale, relax=args.relax,
            flow_iters=args.flow_iters, out=args.out)
