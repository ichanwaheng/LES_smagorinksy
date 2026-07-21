# working/

Self-contained, runnable LES flow-past-sphere package.

## Setup

```powershell
cd working
py -m pip install -r requirements.txt
```

## Run (recommended)

Quick demo:

```powershell
py run.py --quick
```

Full output generation:

```powershell
py run.py
```

## Outputs

Written to `working/outputs/`:

- `flow_past_sphere_midplane.png`
- `flow_past_sphere_wake.png`
- `flow_past_sphere_nut.png`
- `les_sphere_fields.npz`

## Notebook

Open `run_les_sphere.ipynb` from this folder (so paths resolve to `data/`).
