from __future__ import annotations

import plotly.express as px
import plotly.io as pio

from ...post import load_model_data, save_model_data
from ._nicegui_utils import (
    FONT_FAMILIES,
    bind_auto_refresh,
    clean_optional_text,
    compact_button,
    compact_checkbox,
    compact_color,
    compact_input,
    compact_number,
    compact_select,
    control_row,
    export_html,
    normalize_color_value,
    page_shell,
    preserve_plotly_view,
    run_gui,
    section_title,
    set_message,
    shutdown_gui,
    subsection_title,
    update_camera_state,
    update_plot,
)
from .plot_utils import PLOT_ARGS_DEFAULT, reset_plot_props, set_plot_colors, set_plot_props
from .vis_model import PlotModelBase

try:
    from nicegui import ui
except ImportError as e:  # pragma: no cover
    msg = "NiceGUI is required for plot_model_gui(). Install it with: pip install nicegui"
    raise ImportError(msg) from e


GLOBAL_DEFAULTS = PLOT_ARGS_DEFAULT.copy()
LOCAL_DEFAULTS = {
    "style": "surface",
    "color": "",
    "show_ele_hover": True,
    "show_outline": True,
    "show_node_numbering": False,
    "show_ele_numbering": False,
    "show_bc": True,
    "bc_scale": 1.0,
    "show_link": True,
    "show_mp_constraint": True,
    "show_constraint_dofs": False,
    "show_nodal_loads": False,
    "show_ele_loads": False,
    "load_scale": 1.0,
    "show_local_axes": False,
    "local_axes_scale": 1.0,
    "export_html_path": "model_viewer.html",
}


class _ControlValue:
    def __init__(self, value):
        self.value = value


def _default_controls_proxy() -> dict:
    values = {key: _ControlValue(value) for key, value in LOCAL_DEFAULTS.items()}
    for key, value in GLOBAL_DEFAULTS.items():
        values[f"gc_{key}"] = _ControlValue(normalize_color_value(value) if "color" in key else value)
    values["_camera_state"] = {}
    return values


def _apply_global(c) -> None:
    set_plot_props(
        point_size=float(c["gc_point_size"].value or GLOBAL_DEFAULTS["point_size"]),
        line_width=float(c["gc_line_width"].value or GLOBAL_DEFAULTS["line_width"]),
        theme=c["gc_theme"].value or GLOBAL_DEFAULTS["theme"],
        scale_factor=float(c["gc_scale_factor"].value or GLOBAL_DEFAULTS["scale_factor"]),
        show_mesh_edges=bool(c["gc_show_mesh_edges"].value),
        mesh_edge_color=c["gc_mesh_edge_color"].value or GLOBAL_DEFAULTS["mesh_edge_color"],
        mesh_edge_width=float(c["gc_mesh_edge_width"].value or GLOBAL_DEFAULTS["mesh_edge_width"]),
        mesh_opacity=float(c["gc_mesh_opacity"].value or GLOBAL_DEFAULTS["mesh_opacity"]),
        font_family=c["gc_font_family"].value or GLOBAL_DEFAULTS["font_family"],
        font_size=int(c["gc_font_size"].value or GLOBAL_DEFAULTS["font_size"]),
        title_font_size=int(c["gc_title_font_size"].value or GLOBAL_DEFAULTS["title_font_size"]),
        cmap=clean_optional_text(c["gc_cmap"].value),
        cmap_model=clean_optional_text(c["gc_cmap_model"].value),
    )
    set_plot_colors(
        point=c["gc_color_point"].value or GLOBAL_DEFAULTS["color_point"],
        frame=c["gc_color_frame"].value or GLOBAL_DEFAULTS["color_frame"],
        truss=c["gc_color_truss"].value or GLOBAL_DEFAULTS["color_truss"],
        link=c["gc_color_link"].value or GLOBAL_DEFAULTS["color_link"],
        shell=c["gc_color_shell"].value or GLOBAL_DEFAULTS["color_shell"],
        plane=c["gc_color_plane"].value or GLOBAL_DEFAULTS["color_plane"],
        brick=c["gc_color_brick"].value or GLOBAL_DEFAULTS["color_brick"],
        tet=c["gc_color_tet"].value or GLOBAL_DEFAULTS["color_tet"],
        joint=c["gc_color_joint"].value or GLOBAL_DEFAULTS["color_joint"],
        contact=c["gc_color_contact"].value or GLOBAL_DEFAULTS["color_contact"],
        pfem=c["gc_color_pfem"].value or GLOBAL_DEFAULTS["color_pfem"],
        constraint=c["gc_color_constraint"].value or GLOBAL_DEFAULTS["color_constraint"],
        bc=c["gc_color_bc"].value or GLOBAL_DEFAULTS["color_bc"],
        cmap=clean_optional_text(c["gc_cmap"].value),
        cmap_model=clean_optional_text(c["gc_cmap_model"].value),
    )


def _make_figure(model_data, c):
    _apply_global(c)
    model_info, cells = model_data
    plotbase = PlotModelBase(model_info, cells)
    plotter = []
    color = clean_optional_text(c["color"].value)
    style = c["style"].value or LOCAL_DEFAULTS["style"]
    if color:
        plotbase.plot_model_one_color(plotter, color, style)
    else:
        plotbase.plot_model(plotter, style, show_ele_hover=bool(c["show_ele_hover"].value))
    if bool(c["show_node_numbering"].value):
        plotbase.plot_nodal_labels(plotter)
    if bool(c["show_ele_numbering"].value):
        plotbase.plot_ele_labels(plotter)
    if bool(c["show_bc"].value):
        plotbase.plot_bc(plotter, float(c["bc_scale"].value or LOCAL_DEFAULTS["bc_scale"]))
    if bool(c["show_mp_constraint"].value):
        plotbase.plot_mp_constraint(plotter, bool(c["show_constraint_dofs"].value))
    if bool(c["show_link"].value):
        plotbase.plot_link(plotter)
    if bool(c["show_local_axes"].value):
        scale = float(c["local_axes_scale"].value or LOCAL_DEFAULTS["local_axes_scale"])
        plotbase.plot_beam_local_axes(plotter, scale)
        plotbase.plot_link_local_axes(plotter, scale)
        plotbase.plot_shell_local_axes(plotter, scale)
    if bool(c["show_nodal_loads"].value):
        plotbase.plot_node_load(plotter, float(c["load_scale"].value or LOCAL_DEFAULTS["load_scale"]))
    if bool(c["show_ele_loads"].value):
        plotbase.plot_ele_load(plotter, float(c["load_scale"].value or LOCAL_DEFAULTS["load_scale"]))
    fig = plotbase.update_fig(plotter, bool(c["show_outline"].value))
    preserve_plotly_view(fig, c.get("_camera_state"))
    return fig


def _init_model_data(odb_tag: int | str | None):
    if odb_tag is not None:
        return load_model_data(odb_tag, resave=False)
    gui_odb_tag = "__plot_model_gui"
    save_model_data(odb_tag=gui_odb_tag)
    return load_model_data(None, resave=False)


def _reset_controls(c, include_local: bool = True) -> None:
    reset_plot_props()
    if include_local:
        for key, value in LOCAL_DEFAULTS.items():
            c[key].value = value
            c[key].update()
    for key, value in GLOBAL_DEFAULTS.items():
        name = f"gc_{key}"
        if name in c:
            c[name].value = normalize_color_value(value) if "color" in key else value
            c[name].update()


def plot_model_gui(
    odb_tag: int | str | None = None,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    auto_open: bool = True,
) -> None:
    """Launch a NiceGUI application for interactive geometric model visualization."""
    model_data = _init_model_data(odb_tag)

    def root() -> None:
        reset_plot_props()
        controls = {"_camera_state": {}}
        current = {"fig": _make_figure(model_data, _default_controls_proxy())}
        preserve_plotly_view(current["fig"], controls["_camera_state"])
        sidebar, content = page_shell("Model Viewer")

        def update() -> None:
            current["fig"] = _make_figure(model_data, controls)
            update_plot(plot, current["fig"])
            set_message(message, "Figure updated.")

        with sidebar:
            with ui.row().classes("w-full no-wrap").style("gap: 6px;"):
                compact_button("Refresh", update)
                compact_button("Exit", shutdown_gui)
            message = ui.label("").style("font-size: 11px; color: #555; min-height: 14px;")

            section_title("Export")
            with control_row():
                controls["export_html_path"] = compact_input("HTML Path", LOCAL_DEFAULTS["export_html_path"])
                compact_button(
                    "Export HTML",
                    lambda: set_message(message, export_html(current["fig"], controls["export_html_path"].value)),
                )

            section_title("Appearance")
            with control_row():
                controls["style"] = compact_select("Style", ["surface", "wireframe"], LOCAL_DEFAULTS["style"])
                controls["color"] = compact_input("Single Color", LOCAL_DEFAULTS["color"], "e.g. black / #1f77b4")
            with control_row():
                controls["show_ele_hover"] = compact_checkbox("Show element hover", LOCAL_DEFAULTS["show_ele_hover"])
                controls["show_outline"] = compact_checkbox("Show outline", LOCAL_DEFAULTS["show_outline"])

            section_title("Labels")
            with control_row():
                controls["show_node_numbering"] = compact_checkbox("Show node numbering", False)
                controls["show_ele_numbering"] = compact_checkbox("Show element numbering", False)

            section_title("Constraints")
            with control_row():
                controls["show_bc"] = compact_checkbox("Show BC", True)
                controls["bc_scale"] = compact_number("BC scale", 1.0, 0.1)
            with control_row():
                controls["show_link"] = compact_checkbox("Show link", True)
                controls["show_mp_constraint"] = compact_checkbox("Show MP constraint", True)
            with control_row():
                controls["show_constraint_dofs"] = compact_checkbox("Show constraint DOFs", False)

            section_title("Loads")
            with control_row():
                controls["show_nodal_loads"] = compact_checkbox("Show nodal loads", False)
                controls["show_ele_loads"] = compact_checkbox("Show element loads", False)
            with control_row():
                controls["load_scale"] = compact_number("Load scale", 1.0, 0.1)

            section_title("Local Axes")
            with control_row():
                controls["show_local_axes"] = compact_checkbox("Show local axes", False)
                controls["local_axes_scale"] = compact_number("Axes scale", 1.0, 0.1)

            section_title("Reset")
            with control_row():
                compact_button(
                    "Reset All Controls",
                    lambda: (_reset_controls(controls), set_message(message, "All controls have been reset.")),
                )
                compact_button(
                    "Reset Global Control",
                    lambda: (_reset_controls(controls, False), set_message(message, "Global controls have been reset.")),
                )

            section_title("Global Control")
            subsection_title("General Props")
            with control_row():
                controls["gc_point_size"] = compact_number("Point size", GLOBAL_DEFAULTS["point_size"], 0.5)
                controls["gc_line_width"] = compact_number("Line width", GLOBAL_DEFAULTS["line_width"], 0.5)
            with control_row():
                controls["gc_scale_factor"] = compact_number("Scale factor", GLOBAL_DEFAULTS["scale_factor"], 0.01)
                controls["gc_show_mesh_edges"] = compact_checkbox("Show mesh edges", GLOBAL_DEFAULTS["show_mesh_edges"])
            with control_row():
                controls["gc_mesh_edge_color"] = compact_color("Mesh edge color", GLOBAL_DEFAULTS["mesh_edge_color"])
                controls["gc_mesh_edge_width"] = compact_number("Mesh edge width", GLOBAL_DEFAULTS["mesh_edge_width"], 0.1)
            with control_row():
                controls["gc_mesh_opacity"] = compact_number("Mesh opacity", GLOBAL_DEFAULTS["mesh_opacity"], 0.05)
                controls["gc_theme"] = compact_select("Theme", list(pio.templates.keys()), GLOBAL_DEFAULTS["theme"])
            with control_row():
                controls["gc_font_family"] = compact_select("Font family", FONT_FAMILIES, GLOBAL_DEFAULTS["font_family"])
                controls["gc_font_size"] = compact_number("Font size", GLOBAL_DEFAULTS["font_size"], 1)
            with control_row():
                controls["gc_title_font_size"] = compact_number("Title font size", GLOBAL_DEFAULTS["title_font_size"], 1)
            with control_row():
                colorscales = [""] + [x for x in px.colors.named_colorscales() if x]
                controls["gc_cmap"] = compact_select("CMAP", colorscales, GLOBAL_DEFAULTS["cmap"], clearable=True)
                controls["gc_cmap_model"] = compact_select(
                    "CMAP_MODEL", colorscales, GLOBAL_DEFAULTS["cmap_model"], clearable=True
                )

            subsection_title("Element Colors")
            for label, key in [
                ("Point", "color_point"),
                ("Frame", "color_frame"),
                ("Truss", "color_truss"),
                ("Link", "color_link"),
                ("Shell", "color_shell"),
                ("Plane", "color_plane"),
                ("Brick", "color_brick"),
                ("Tet", "color_tet"),
                ("Joint", "color_joint"),
                ("Contact", "color_contact"),
                ("PFEM", "color_pfem"),
                ("Constraint", "color_constraint"),
                ("BC", "color_bc"),
            ]:
                controls[f"gc_{key}"] = compact_color(label, GLOBAL_DEFAULTS[key])

        with content:
            plot = ui.plotly(current["fig"]).classes("w-full").style("height: 100vh;")
            plot.on(
                "plotly_relayout",
                lambda e: update_camera_state(controls["_camera_state"], e.args),
                throttle=0.2,
            )

        bind_auto_refresh(controls, update, exclude={"export_html_path"})

    run_gui("Model Viewer", host, port, debug, auto_open, root=root)
