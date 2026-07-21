# LES_smagorinksy

A Python/Jupyter CFD project: incompressible flow past a sphere in a 10x5x5 m channel,
using a finite-volume solver with the Smagorinsky LES turbulence model.

## Pipeline

1. `computational_grid_gmsh_visualized.ipynb` — Gmsh mesh generation -> `fluid_mesh_3d.msh`.
2. `read_mesh_from_the_generated_mesh.ipynb` — build finite-volume connectivity/geometry -> `processed_mesh.npz`.
3. `les_solver.py` / `LES_flow_past_sphere.ipynb` — solve incompressible Navier-Stokes
   (SIMPLE + Rhie-Chow + Smagorinsky LES) and visualise the flow past the sphere.
4. `plot_les_result.py` — mid-plane velocity/pressure/streamline plots of a result.

Pre-generated `fluid_mesh_3d.msh` and `processed_mesh.npz` are committed, so the solver
can be run directly without regenerating the mesh.

## Running

- Solver (CLI): `python les_solver.py --nu 0.01 --iters 300 --beta 0.9 --limiter none --avg_last 150`
- Plot a result: `python plot_les_result.py les_result.npz les_flow_past_sphere.png`
- Notebook: run `LES_flow_past_sphere.ipynb` (imports `les_solver`).

## Cursor Cloud specific instructions

- Dependencies are Python-only; the update script installs them via
  `pip install --break-system-packages -r requirements.txt` (system Python 3.12 is
  externally managed, hence the flag). No virtualenv is used.
- `gmsh` (used only to (re)generate the mesh) needs system OpenGL libs that are NOT pip
  packages: `libglu1-mesa` (and it pulls `libopengl0`). These are installed via `apt` and
  persist in the VM snapshot; they are intentionally NOT in the update script. Without them
  `import gmsh` fails with `libGLU.so.1: cannot open shared object file`.
- Run gmsh headless: `python computational_grid_gmsh_visualized.py -nopopup` avoids the
  FLTK GUI. NOTE: the exported `.py` lost its indentation (module-level code runs before
  `gmsh.initialize()` and errors); use the `.ipynb` for mesh generation, which is correct.
- The notebooks declare a custom kernel `fenics-system` that does not exist here. Execute
  them with the default kernel, e.g.
  `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 <nb>.ipynb`.
- `processed_mesh.npz` does NOT store boundary-face tags (inlet/outlet/walls/object); the
  preprocessing notebook computes them but omits them from `np.savez`. `les_solver.py`
  therefore re-derives boundary faces geometrically (matches the Gmsh counts exactly:
  inlet 66, outlet 66, walls 526, object 3068).

### Solver numerics notes (non-obvious)

- Pressure-velocity coupling uses Rhie-Chow, so the face mass flux is divergence-free by
  construction (`max|div|` ~ 1e-6 every iteration); it is NOT a convergence indicator.
  Use `d|u|/U` (velocity change per iteration) and steadiness of `max|u|` instead.
- Convection scheme trade-off on this (wake-coarse) mesh: bounded limiters
  (`--limiter vanleer|superbee`) converge cleanly but over-diffuse and suppress wake
  separation; near-central (`--limiter none --beta 0.9`) captures the recirculating wake
  but sits in a mild dispersive limit cycle. Report the iteration-averaged mean field
  (`--avg_last`), which is standard for LES and removes the limit-cycle oscillation.
- `les_result.npz` and generated PNGs are regeneratable and are git-ignored.
