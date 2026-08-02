# LES_smagorinksy

Incompressible LES (Smagorinsky) / PISO finite-volume work, plus coupled
tensile-membrane FSI packages.

## Standalone quasi-static FSI (recommended entry)

Self-contained folder — **no imports from other repo code**:

See [`quasi_static/`](quasi_static/)

```bash
cd quasi_static
pip install -r requirements.txt
python main.py --quick
python run_gif.py --quick
```

## Transient / full tensile membrane package

See [`tensile_membrane_fsi/`](tensile_membrane_fsi/) for the broader package
(transient explicit dynamics, examples, docs):

```bash
cd tensile_membrane_fsi
pip install -r requirements.txt
python main.py --quick
```
