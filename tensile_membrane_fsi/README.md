# Tensile Membrane Structures under Fluid Flow (FSI)

Python package for **fluid–structure interaction** of prestressed tensile membranes
(sails, canopies, fabric roofs) in incompressible flow.

This folder is self-contained and sits alongside the existing LES / PISO notebooks
in the repository root.

## What’s inside

```
tensile_membrane_fsi/
├── config/default.yaml          # physics & numerics
├── main.py                      # full FSI entry point
├── requirements.txt
├── src/
│   ├── membrane/                # CST membrane FEM + prestress / dynamics
│   ├── fluid/                   # Cartesian NS + Smagorinsky LES + immersed membrane
│   ├── fsi/                     # partitioned serial-staggered coupling
│   └── utils/                   # YAML I/O, VTK/NPZ, plots
├── examples/
│   ├── run_membrane_only.py     # structure-only gust response
│   ├── run_fsi_demo.py          # coarse coupled demo
│   └── generate_gmsh_channel.py # optional unstructured channel+membrane mesh
├── tests/
└── output/                      # created at runtime
```

## Physics model

| Part | Model |
|------|--------|
| Membrane | Constant-strain triangles, isotropic prestress + plane-stress elasticity, explicit central-difference dynamics |
| Fluid | 3D incompressible Navier–Stokes, fractional-step / PISO-like projection, optional Smagorinsky LES |
| Coupling | Serial staggered: fluid → dynamic pressure loads → membrane → immersed-boundary update |
| IB | Thin-band immersed membrane in the Cartesian fluid grid |

## Quick start

```bash
cd tensile_membrane_fsi
pip install -r requirements.txt

# fast smoke demo (~coarse mesh, short time)
python main.py --quick

# membrane-only form-finding + gust
python examples/run_membrane_only.py

# coarse FSI demo
python examples/run_fsi_demo.py

# full run from config
python main.py -c config/default.yaml

# tests
python -m pytest tests/ -q
```

## Configure a case

Edit `config/default.yaml`:

- `membrane.*` — span, mesh density, fabric E / ν / thickness / prestress, fixed edges
- `fluid.*` — channel size, resolution, ρ, ν, inlet speed, membrane placement
- `fsi.*` — under-relaxation, sub-iterations, load model (`dynamic_pressure` or `interpolated_field`)
- `les.*` — Smagorinsky `Cs`
- `time.*` — `dt`, `t_end`

## Outputs

Written under `output/` (or `simulation.output_dir`):

- `snapshot_XXXXXX.npz` — membrane nodes + fluid fields
- `membrane_XXXXXX.vtk` — open in ParaView
- `slice_XXXXXX.png` — mid-plane |u| with membrane outline
- `history.csv` / `history.png` — displacement, KE, CFL, FSI residual

## Notes / limits

- The built-in fluid solver uses a **uniform Cartesian grid** with an immersed membrane.
  For production unstructured LES, generate a Gmsh mesh with
  `examples/generate_gmsh_channel.py` and couple externally to your PISO / OpenFOAM pipeline.
- Explicit membrane time steps are limited by the membrane wave speed; reduce `dt` or
  prestress / E if the run becomes unstable.
- Load transfer uses a dynamic-pressure / incidence model suitable for teaching and
  prototyping; replace `src/fsi/load_transfer.py` for higher-fidelity pressure mapping.
