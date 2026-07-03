from importlib.util import find_spec

from ...utils import make_dependency_missing

if find_spec("gmsh"):
    from ._read_gmsh import Gmsh2OPS
else:
    Gmsh2OPS = make_dependency_missing("pre.Gmsh2OPS", "gmsh", extra="gmsh")

from .tcl2py import tcl2py

__all__ = []  # Initialize __all__ to avoid linting issues
__all__ += ["Gmsh2OPS", "tcl2py"]  # Import Gmsh2OPS and tcl2py
