# LES_smagorinksy

LES (Smagorinsky) finite-volume simulation of **turbulent flow past a sphere**.

## Working folder (recommended)

Use the self-contained package:

```powershell
cd LES_smagorinksy\working
py -m pip install -r requirements.txt
py run.py --quick
```

Full output generation:

```powershell
py run.py
```

Results are written to `working/outputs/`.

See `working/README.md` for details.

## Quick start (repo root)

```powershell
cd LES_smagorinksy
py -m pip install -r requirements.txt
py computational_grid_gmsh_visualized.py -nopopup
py -m jupyter lab
```

Then open and run **`PISO_SOLVER.ipynb`**.

## Pipeline

1. `computational_grid_gmsh_visualized.py` → `fluid_mesh_3d.msh`
2. `read_mesh_from_the_generated_mesh.ipynb` → `processed_mesh.npz` (with boundary tags)
3. `working/` or `PISO_SOLVER.ipynb` / `les_sphere_flow.py` → LES run + wake visualization

## Boundary conditions

- **Inlet**: fixed \(U_\infty\) + mild turbulence intensity
- **Outlet**: convective (Orlanski) outflow + soft pressure (non-reflecting)
- **Outer walls**: free-slip (reduces blockage)
- **Sphere**: no-slip

## Outputs

- `les_sphere_fields.npz`
- `flow_past_sphere_midplane.png`
- `flow_past_sphere_wake.png`
- `flow_past_sphere_nut.png`

Generate them with:

```powershell
py les_sphere_flow.py --output
```

or:

```powershell
cd working
py run.py
```


## Flexible membrane FSI (new)

The sphere package in `working/` is unchanged. New code lives in `membrane_fsi/`:

```powershell
cd membrane_fsi
py -m pip install -r requirements.txt
py run.py --quick
```

See `membrane_fsi/README.md`.
