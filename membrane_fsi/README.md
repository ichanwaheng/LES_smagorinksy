# membrane_fsi/

New code for **LES flow past a flexible rectangular membrane** (FSI).

The existing sphere package in `working/` is left unchanged.

## What changed physically

| Rigid sphere (`working/`) | Flexible membrane (`membrane_fsi/`) |
|---|---|
| Rigid body | Flexible cantilever plate |
| Fixed no-slip `U=0` | Moving no-slip `U = U_wall(η̇)` |
| No solid DOFs | Inertia + bending stiffness + damping |
| Force not needed for motion | Fluid pressure/shear → structural motion |

## Boundary conditions (fluid)

- Inlet: `U∞` + mild TI
- Outlet: convective Orlanski + soft pressure
- Outer walls: free-slip
- Membrane: **no-slip with wall velocity from the solid**

## Run

```powershell
cd membrane_fsi
py -m pip install -r requirements.txt
py run.py --quick
```

Outputs → `membrane_fsi/outputs/`:
- `flow_past_membrane_midplane.png`
- `membrane_deflection.png`
- `flow_past_membrane.gif`
- `les_membrane_fields.npz`

## Layout

- `mesh/generate_membrane_mesh.py` — Gmsh box − thin rectangular plate
- `mesh/process_mesh.py` — FV connectivity + boundary tags
- `solid/membrane_model.py` — cantilever flexible membrane solid
- `les_membrane_fsi.py` — LES fluid + partitioned FSI coupling
- `run.py` — one-command driver
