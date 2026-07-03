from importlib.util import find_spec

from ...utils import make_dependency_missing
from ._plot_fiber_sec import vis_fiber_sec_real
from ._plot_fiber_sec_by_cmds import fiber, layer, patch, plot_fiber_sec_cmds, section

if find_spec("sectionproperties"):
    from .sec_mesh import (
        FiberSecMesh,
        create_circle_patch,
        create_circle_points,
        create_material,
        create_patch_from_dxf,
        create_polygon_patch,
        create_polygon_points,
        line_offset,
        offset,
        poly_offset,
        set_patch_material,
    )
else:
    FiberSecMesh = make_dependency_missing("pre.section.FiberSecMesh", "sectionproperties", extra="pre")
    create_circle_patch = make_dependency_missing("pre.section.create_circle_patch", "sectionproperties", extra="pre")
    create_circle_points = make_dependency_missing("pre.section.create_circle_points", "sectionproperties", extra="pre")
    create_material = make_dependency_missing("pre.section.create_material", "sectionproperties", extra="pre")
    create_patch_from_dxf = make_dependency_missing("pre.section.create_patch_from_dxf", "sectionproperties", extra="pre")
    create_polygon_patch = make_dependency_missing("pre.section.create_polygon_patch", "sectionproperties", extra="pre")
    create_polygon_points = make_dependency_missing("pre.section.create_polygon_points", "sectionproperties", extra="pre")
    line_offset = make_dependency_missing("pre.section.line_offset", "sectionproperties", extra="pre")
    offset = make_dependency_missing("pre.section.offset", "sectionproperties", extra="pre")
    poly_offset = make_dependency_missing("pre.section.poly_offset", "sectionproperties", extra="pre")
    set_patch_material = make_dependency_missing("pre.section.set_patch_material", "sectionproperties", extra="pre")

SecMesh = FiberSecMesh

__all__ = [
    "FiberSecMesh",
    "SecMesh",
    "create_circle_patch",
    "create_circle_points",
    "create_material",
    "create_patch_from_dxf",
    "create_polygon_patch",
    "create_polygon_points",
    "line_offset",
    "offset",
    "poly_offset",
    "set_patch_material",
]

__all__ += ["fiber", "layer", "patch", "section"]

__all__ += ["plot_fiber_sec_cmds"]

__all__ += ["vis_fiber_sec_real"]
