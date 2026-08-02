# Quasi-static FSI — Updated Weight Method + PISO/LES

**Standalone** partitioned quasi-static fluid–structure interaction for
prestressed tensile membranes. This folder is self-contained inside
`LES_smagorinksy` and does **not** depend on `tensile_membrane_fsi/` or any
other repository code.

- Membrane form: **Updated Weight Method (UWM)** (Marbaniang et al., 2022)
- Fluid: **Smagorinsky LES** discretisation + **PISO** (Issa, 1986)

## Layout

```
quasi_static/
├── fluid/          # Cartesian NS, Smagorinsky LES, PISO
├── membrane/       # geometry, materials, initial sag seed
├── fsi/            # load transfer + immersed boundary
├── utils/          # YAML I/O, plots, GIF helpers
├── uwm.py          # Updated Weight Method
├── coupling.py     # outer QS-FSI loop
├── excel_export.py
├── main.py         # iterative form updates
├── run_gif.py      # timed run → original + amplified GIFs
├── config/
└── tests/
```

## Install & run

```bash
cd quasi_static
pip install -r requirements.txt

# quick smoke (coarse)
python main.py --quick

# full config
python main.py -c config/quasi_static.yaml

# animated GIFs (fluid time + UWM form updates)
python run_gif.py --quick
python run_gif.py --t-end 1.0 --fluid-substeps 5 --fps 10
```

Outputs go to `quasi_static/output/` (NPZ, VTK, history, GIFs, Excel).

## Tests

```bash
cd quasi_static
python -m pytest tests -q
```

## Configure

See `config/quasi_static.yaml` (`membrane.*`, `fluid.*`, `les.*`, `quasi_static.*`).
Default load mode is `pressure_jump` (two-sided Δp).

## References

- Marbaniang, A.L., Dutta, S. & Ghosh, S. (2022). *Updated weight method…*
  Struct. Multidisc. Optim. 65:169.
- Issa, R.I. (1986). *Solution of the implicitly discretised fluid flow
  equations by operator-splitting.* J. Comput. Phys. 62:40–65.
