"""Standalone quasi-static FSI: Smagorinsky LES + PISO ↔ UWM membrane.

This package is self-contained under ``quasi_static/`` and does not import
from ``tensile_membrane_fsi`` or any other repo folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .coupling import QuasiStaticFSI, QuasiStaticHistory
from .uwm import updated_weight_form_find, UWMResult

__all__ = [
    "QuasiStaticFSI",
    "QuasiStaticHistory",
    "updated_weight_form_find",
    "UWMResult",
]
