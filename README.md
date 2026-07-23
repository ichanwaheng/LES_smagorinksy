# LES_smagorinksy

Incompressible LES (Smagorinsky) / PISO finite-volume work, plus a coupled
**tensile membrane FSI** package.

## Tensile membrane under fluid flow

See [`tensile_membrane_fsi/`](tensile_membrane_fsi/) for a full folder of codes:

- prestressed membrane FEM
- Cartesian fluid solver with Smagorinsky LES
- partitioned FSI coupling
- examples, tests, and YAML configuration

```bash
cd tensile_membrane_fsi
pip install -r requirements.txt
python main.py --quick
```
