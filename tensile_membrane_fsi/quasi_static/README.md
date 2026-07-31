# Quasi-static FSI — Updated Weight Method + PISO/LES

Partitioned **quasi-static** fluid–structure interaction for prestressed tensile
membranes. The membrane form is updated with the **Updated Weight Method (UWM)**
(Marbaniang, Dutta & Ghosh, 2022); the fluid is advanced with **Smagorinsky LES**
discretisation followed by the **PISO** algorithm (Issa, 1986).

This is separate from the transient explicit-dynamics path in `../src/fsi`.

## Algorithm

```
1. UWM form-find under target prestress (no fluid load)
2. Outer iteration k = 1 … max_iters:
     a. Update immersed boundary from current form
     b. Advance fluid (PISO + LES) for fluid_substeps
     c. Transfer pressure → nodal forces
     d. Under-relax forces
     e. UWM re-form-find under prestress + fluid forces
     f. Under-relax shape; check shape residual
3. Write snapshots / VTK / history
```

No structural mass or damping is used. The “new form” after pressure arrives is
the UWM equilibrium for the current load, not a dynamic time step.

## Run

```bash
cd tensile_membrane_fsi

# quick smoke (coarse)
python quasi_static/main.py --quick

# full config
python quasi_static/main.py -c quasi_static/config/quasi_static.yaml

# animated GIF over a time interval (fluid time + UWM form updates)
python quasi_static/run_gif.py --quick
python quasi_static/run_gif.py --t-end 1.0 --fps 8

# exaggerate deflection in the GIF only (auto targets ~0.8 m peak |Δz|)
python quasi_static/run_gif.py --quick
python quasi_static/run_gif.py --quick --target-amp 1.0
python quasi_static/run_gif.py --quick --disp-scale 400
```

Outputs go to `quasi_static/output/` (NPZ, VTK, history CSV/PNG, slice plot,
`membrane_quasi_static.gif` from `run_gif.py`, and
`membrane_deformations.xlsx` with every node’s `x,y,z` / `ux,uy,uz` at each
time step).

Excel workbook sheets:
- `summary` — peak displacement per time step
- `reference` — flat-plane reference coordinates
- `deformations` — long table of all nodes × all times
- `step_XXXX` — one sheet per time step (short runs)

Each GIF frame corresponds to one outer iteration: the fluid advances by
`fluid_substeps * dt`, then UWM updates the membrane form under the new load.
Physical time on the frame is the accumulated fluid time.

GIF deflection is amplified about the **flat mounting plane** (not the sagged
seed) and auto-scaled so the peak visual `|Δz|` reaches `target_amp`. The 3D
panel uses a corrected box aspect so Z is not squashed. Saved NPZ/VTK remain
physical.

## Configure

See `config/quasi_static.yaml`:

| Block | Role |
|-------|------|
| `membrane.*` | Patch size, mesh, prestress, supports |
| `fluid.*` / `les.*` | Channel, PISO grid, Smagorinsky |
| `time.dt` | Fluid Δt inside each outer iteration |
| `quasi_static.*` | Outer iters, UWM updates, load mode, relaxation |

## Tests

```bash
python -m pytest quasi_static/tests -q
```

## References

- Marbaniang, A.L., Dutta, S. & Ghosh, S. (2022). *Updated weight method: an
  optimisation-based form-finding method of tensile membrane structures.*
  Structural and Multidisciplinary Optimization 65:169.
