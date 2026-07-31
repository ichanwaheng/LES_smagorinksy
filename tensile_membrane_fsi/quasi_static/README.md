# Quasi-static FSI — Updated Weight Method + PISO/LES

Partitioned **quasi-static** fluid–structure interaction for prestressed tensile
membranes. The membrane form is updated with the **Updated Weight Method (UWM)**
(Marbaniang, Dutta & Ghosh, 2022); the fluid is advanced with the existing
**PISO-like** incompressible solver and optional **Smagorinsky LES**.

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
```

Outputs go to `quasi_static/output/` (NPZ, VTK, history CSV/PNG, slice plot).

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
