# working/

Self-contained, runnable LES flow-past-sphere package.

## Setup

```powershell
cd working
py -m pip install -r requirements.txt
```

## Run (recommended)

Quick demo (includes GIF):

```powershell
py run.py --quick
```

Full output generation:

```powershell
py run.py
```

## Outputs

Written to `working/outputs/`:

- `flow_past_sphere.gif`  ← animation
- `flow_past_sphere_streamlines.png`  ← streamlines
- `flow_past_sphere_midplane.png`
- `flow_past_sphere_wake.png`
- `flow_past_sphere_nut.png`
- `les_sphere_fields.npz`

## Notebook

Open `run_les_sphere.ipynb` from this folder and Run All.
The GIF is displayed in the notebook.
