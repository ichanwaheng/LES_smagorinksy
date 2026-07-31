"""
Tensile Membrane Structures under Fluid Flow (FSI)

Package layout
--------------
membrane/  – prestressed membrane FEM (dynamic relaxation / Newmark)
fluid/     – incompressible NS: Smagorinsky LES + Issa PISO
fsi/       – partitioned serial-staggered coupling
utils/     – I/O, visualization, helpers
"""

__version__ = "0.1.0"
__all__ = ["membrane", "fluid", "fsi", "utils"]
