"""Updated Weight Method (UWM) form-finding for tensile membranes.

Implements the optimisation-based form-finding of Marbaniang et al.
(Struct. Multidisc. Optim., 2022): minimise a weighted sum of squared
edge lengths whose weights are tied to a target prestress and are
updated from the current geometry.

With nodal loads F (e.g. fluid pressure), the same stationarity condition
is solved with a nonzero right-hand side — this is the quasi-static
“new form under pressure” step used by the iterative FSI driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


EdgeKey = Tuple[int, int]


def _undirected(a: int, b: int) -> EdgeKey:
    return (a, b) if a < b else (b, a)


def triangle_area(nodes: np.ndarray, conn: np.ndarray) -> float:
    x0, x1, x2 = nodes[conn[0]], nodes[conn[1]], nodes[conn[2]]
    return 0.5 * float(np.linalg.norm(np.cross(x1 - x0, x2 - x0)))


def edge_side_force(N_pre: float, area: float, length: float) -> float:
    """Side force t from isotropic prestress resultant N_pre [N/m].

    For a triangle, t = N_pre * A / L along each side (standard conversion
    used to build UWM weights from target prestress).
    """
    return N_pre * area / max(length, 1e-14)


def compute_edge_weights(
    nodes: np.ndarray,
    elements: np.ndarray,
    N_pre: float,
) -> Dict[EdgeKey, float]:
    """UWM weights W = t / (2 L) assembled over all element sides.

    Shared edges accumulate contributions from adjacent triangles, matching
    the element-wise sum in the UWM objective.
    """
    weights: Dict[EdgeKey, float] = {}
    for conn in elements:
        area = triangle_area(nodes, conn)
        if area < 1e-16:
            continue
        for a, b in ((conn[0], conn[1]), (conn[1], conn[2]), (conn[2], conn[0])):
            key = _undirected(int(a), int(b))
            L = float(np.linalg.norm(nodes[b] - nodes[a]))
            t = edge_side_force(N_pre, area, L)
            w = t / (2.0 * max(L, 1e-14))
            weights[key] = weights.get(key, 0.0) + w
    return weights


def assemble_force_density_system(
    nodes: np.ndarray,
    fixed: np.ndarray,
    weights: Dict[EdgeKey, float],
    f_ext: np.ndarray,
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Build 2 W (x_i - x_j) = F on free DOFs (force-density / UWM stationarity).

    Fixed nodes are eliminated (Dirichlet). Returns (A, b, free_node_ids).
    """
    free = np.where(~fixed)[0]
    n_free = free.size
    if n_free == 0:
        return sparse.csr_matrix((0, 0)), np.zeros((0, 3)), free

    idx = -np.ones(nodes.shape[0], dtype=int)
    idx[free] = np.arange(n_free)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = f_ext[free].copy()
    diag = np.zeros(n_free)

    for (i, j), W in weights.items():
        coef = 2.0 * float(W)
        ia, ib = int(idx[i]), int(idx[j])
        if ia >= 0 and ib >= 0:
            diag[ia] += coef
            diag[ib] += coef
            rows.extend([ia, ib])
            cols.extend([ib, ia])
            data.extend([-coef, -coef])
        elif ia >= 0 and ib < 0:
            diag[ia] += coef
            rhs[ia] += coef * nodes[j]
        elif ib >= 0 and ia < 0:
            diag[ib] += coef
            rhs[ib] += coef * nodes[i]

    for ia in range(n_free):
        rows.append(ia)
        cols.append(ia)
        if diag[ia] > 0.0:
            data.append(diag[ia])
        else:
            data.append(1.0)
            rhs[ia] = nodes[free[ia]]

    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_free, n_free)).tocsr()
    return A, rhs, free


def solve_force_density(
    nodes: np.ndarray,
    fixed: np.ndarray,
    weights: Dict[EdgeKey, float],
    f_ext: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve one linear force-density / UWM step with fixed weights."""
    f = np.zeros_like(nodes) if f_ext is None else np.asarray(f_ext, dtype=float)
    A, rhs, free = assemble_force_density_system(nodes, fixed, weights, f)
    x = nodes.copy()
    if free.size == 0:
        return x
    for d in range(3):
        x[free, d] = spsolve(A, rhs[:, d])
    x[fixed] = nodes[fixed]
    return x


@dataclass
class UWMResult:
    nodes: np.ndarray
    n_weight_updates: int
    residual: float
    objective: float


def uwm_objective(nodes: np.ndarray, weights: Dict[EdgeKey, float]) -> float:
    total = 0.0
    for (i, j), W in weights.items():
        L2 = float(np.sum((nodes[j] - nodes[i]) ** 2))
        total += W * L2
    return total


def updated_weight_form_find(
    nodes: np.ndarray,
    elements: np.ndarray,
    fixed: np.ndarray,
    N_pre: float,
    f_ext: Optional[np.ndarray] = None,
    support_nodes: Optional[np.ndarray] = None,
    max_weight_updates: int = 25,
    tol: float = 1e-7,
    under_relaxation: float = 1.0,
) -> UWMResult:
    """Updated Weight Method: alternate weight update and FD solve.

    Parameters
    ----------
    nodes :
        Initial nodal coordinates (N, 3).
    elements :
        Triangle connectivity (M, 3).
    fixed :
        Boolean Dirichlet mask (N,).
    N_pre :
        Target isotropic prestress resultant [N/m].
    f_ext :
        Optional nodal external forces (N, 3) from fluid pressure.
        ``None`` → classical prestress form-finding.
    support_nodes :
        Coordinates held on fixed DOFs (defaults to ``nodes``).
    max_weight_updates :
        Outer UWM iterations (recompute W from geometry).
    tol :
        Relative nodal change stopping criterion.
    under_relaxation :
        Blend between previous and new coordinates (0–1].
    """
    x = np.asarray(nodes, dtype=float).copy()
    x_support = (
        np.asarray(nodes, dtype=float).copy()
        if support_nodes is None
        else np.asarray(support_nodes, dtype=float).copy()
    )
    elements = np.asarray(elements, dtype=int)
    fixed = np.asarray(fixed, dtype=bool)
    alpha = float(np.clip(under_relaxation, 1e-3, 1.0))
    residual = np.inf
    obj = 0.0
    it = 0

    for it in range(1, max_weight_updates + 1):
        weights = compute_edge_weights(x, elements, N_pre)
        x_new = solve_force_density(x, fixed, weights, f_ext)
        x_new[fixed] = x_support[fixed]
        dx = x_new - x
        residual = float(np.linalg.norm(dx) / (np.linalg.norm(x) + 1e-12))
        x = (1.0 - alpha) * x + alpha * x_new
        x[fixed] = x_support[fixed]
        obj = uwm_objective(x, weights)
        if residual < tol:
            break

    return UWMResult(
        nodes=x, n_weight_updates=it, residual=residual, objective=obj
    )
