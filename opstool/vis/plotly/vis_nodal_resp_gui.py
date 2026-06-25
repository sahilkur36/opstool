from __future__ import annotations

import ast

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from ...post import get_nodal_responses
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
from .vis_nodal_resp import PlotNodalResponse

try:
    from nicegui import ui
except ImportError as e:  # pragma: no cover
    msg = "NiceGUI is required for plot_nodal_resp_gui(). Install it with: pip install nicegui"
    raise ImportError(msg) from e


def _parse_step(step_dropdown: str | None, step_text: str | None):
    text = str(step_text).strip() if step_text is not None else ""
    if text:
        try:
            return int(text)
        except ValueError:
            return text
    value = str(step_dropdown).strip() if step_dropdown is not None else "0"
    try:
        return int(value)
    except ValueError:
        return value


def _parse_defo_scale(value):
    if value is None:
        return "auto"
    text = str(value).strip()
    if not text:
        return "auto"
    low = text.lower()
    if low == "auto":
        return "auto"
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        num = float(text)
        return int(num) if num.is_integer() else num
    except ValueError:
        return "auto"


def _parse_resp_dof(value: str | None):
    if value is None:
        return ("UX", "UY", "UZ")
    text = str(value).strip()
    if not text:
        return ("UX", "UY", "UZ")
    if text.startswith("[") or text.startswith("("):
        parsed = ast.literal_eval(text)
        if isinstance(parsed, str):
            return parsed.strip()
        if isinstance(parsed, (list, tuple)):
            values = [str(v).strip() for v in parsed if str(v).strip()]
            return values[0] if len(values) == 1 else tuple(values)
    values = [x.strip() for x in (text.split(",") if "," in text else text.split()) if x.strip()]
    return values[0] if len(values) == 1 else tuple(values or ["UX", "UY", "UZ"])


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title={"text": title},
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{"text": title, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False}],
    )
    return fig


def _apply_global(c):
    cmap = (
        None
        if c["gc_cmap"].value == ""
        else (PLOT_ARGS_DEFAULT["cmap"] if c["gc_cmap"].value == "__default__" else c["gc_cmap"].value)
    )
    set_plot_props(
        point_size=float(c["gc_point_size"].value or PLOT_ARGS_DEFAULT["point_size"]),
        line_width=float(c["gc_line_width"].value or PLOT_ARGS_DEFAULT["line_width"]),
        scale_factor=float(c["gc_scale_factor"].value or PLOT_ARGS_DEFAULT["scale_factor"]),
        show_mesh_edges=bool(c["gc_show_mesh_edges"].value),
        mesh_edge_color=c["gc_mesh_edge_color"].value or PLOT_ARGS_DEFAULT["mesh_edge_color"],
        mesh_edge_width=float(c["gc_mesh_edge_width"].value or PLOT_ARGS_DEFAULT["mesh_edge_width"]),
        mesh_opacity=float(c["gc_mesh_opacity"].value or PLOT_ARGS_DEFAULT["mesh_opacity"]),
        theme=c["gc_theme"].value or PLOT_ARGS_DEFAULT["theme"],
        font_family=c["gc_font_family"].value or PLOT_ARGS_DEFAULT["font_family"],
        font_size=int(c["gc_font_size"].value or PLOT_ARGS_DEFAULT["font_size"]),
        title_font_size=int(c["gc_title_font_size"].value or PLOT_ARGS_DEFAULT["title_font_size"]),
        cmap=cmap,
    )
    set_plot_colors(
        constraint=c["gc_color_constraint"].value or PLOT_ARGS_DEFAULT["color_constraint"],
        bc=c["gc_color_bc"].value or PLOT_ARGS_DEFAULT["color_bc"],
    )


def _reset_plotbase(plotbase: PlotNodalResponse) -> None:
    plotbase.FIGURE = go.Figure()
    plotbase.resps_norm = None
    plotbase.defo_scale_factor = None
    plotbase.set_interp_beam_on(False)


def _normalize_resp_type(resp_type: str) -> str:
    value = resp_type.lower()
    if value in ["disp", "dispacement"]:
        return "disp"
    if value in ["vel", "velocity"]:
        return "vel"
    if value in ["accel", "acceleration"]:
        return "accel"
    if value in ["reaction", "reactionforce"]:
        return "reaction"
    if value in ["reactionincinertia", "reactionincinertiaforce"]:
        return "reactionIncInertia"
    if value in ["rayleighforces", "rayleigh"]:
        return "rayleighForces"
    if value in ["pressure"]:
        return "pressure"
    msg = (
        f"Invalid response type: {resp_type}. "
        "Valid options are: disp, vel, accel, reaction, reactionIncInertia, rayleighForces, pressure."
    )
    raise ValueError(msg)


def _set_cached_comp_resp_type(plotbase: PlotNodalResponse, resp_type: str, component, response_cache) -> None:
    normalized = _normalize_resp_type(resp_type)
    plotbase.resp_type = normalized
    plotbase.component = component.upper() if isinstance(component, str) else list(component)
    if normalized not in response_cache:
        response_cache[normalized] = get_nodal_responses(
            plotbase.odb_tag,
            resp_type=normalized,
            lazy_load=plotbase.lazy_load,
            print_info=False,
        )
    plotbase.set_resp_step_data(response_cache[normalized])
    if normalized == "disp" and plotbase.nodal_disp_steps is None:
        plotbase.nodal_disp_steps = response_cache[normalized]


def _configure_plotbase(plotbase: PlotNodalResponse, c, response_cache) -> None:
    _reset_plotbase(plotbase)
    plotbase.set_unit(
        symbol=clean_optional_text(c["unit_symbol"].value),
        factor=float(c["unit_factor"].value or 1.0),
    )
    _set_cached_comp_resp_type(
        plotbase, c["resp_type"].value, _parse_resp_dof(c["resp_dof_text"].value), response_cache
    )


def _plot_cached_static(plotbase: PlotNodalResponse, c, response_cache) -> go.Figure:
    _apply_global(c)
    _configure_plotbase(plotbase, c, response_cache)
    if c["view_mode"].value == "slide":
        plotbase.plot_slide(
            alpha=_parse_defo_scale(c["defo_scale"].value),
            show_defo=bool(c["show_defo"].value),
            show_bc=bool(c["show_bc"].value),
            bc_scale=float(c["bc_scale"].value or 1.0),
            show_mp_constraint=bool(c["show_mp_constraint"].value),
            style=c["style"].value,
            show_origin=bool(c["show_undeformed"].value),
            show_max_min=bool(c["show_max_min"].value),
        )
    else:
        plotbase.plot_peak_step(
            step=_parse_step(c["step_dropdown"].value, c["step_text"].value),
            alpha=_parse_defo_scale(c["defo_scale"].value),
            show_defo=bool(c["show_defo"].value),
            show_bc=bool(c["show_bc"].value),
            bc_scale=float(c["bc_scale"].value or 1.0),
            show_mp_constraint=bool(c["show_mp_constraint"].value),
            show_origin=bool(c["show_undeformed"].value),
            style=c["style"].value,
            show_max_min=bool(c["show_max_min"].value),
        )
    return plotbase.update_fig(show_outline=bool(c["show_outline"].value))


def _plot_cached_animation(plotbase: PlotNodalResponse, c, response_cache) -> go.Figure:
    _apply_global(c)
    _configure_plotbase(plotbase, c, response_cache)
    framerate = int(c["animation_framerate"].value) if c["animation_framerate"].value is not None else None
    plotbase.plot_anim(
        framerate=framerate,
        alpha=_parse_defo_scale(c["animation_defo_scale"].value),
        show_defo=bool(c["animation_show_defo"].value),
        show_bc=bool(c["show_bc"].value),
        bc_scale=float(c["bc_scale"].value or 1.0),
        show_mp_constraint=bool(c["show_mp_constraint"].value),
        show_origin=bool(c["animation_show_undeformed"].value),
        style=c["style"].value,
        show_max_min=bool(c["animation_show_max_min"].value),
    )
    return plotbase.update_fig(show_outline=bool(c["show_outline"].value))


def plot_nodal_resp_gui(  # noqa: C901
    odb_tag: int | str,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    auto_open: bool = True,
) -> None:
    """Launch a NiceGUI viewer for nodal responses."""
    if odb_tag is None:
        raise ValueError("plot_nodal_resp_gui requires a non-None odb_tag.")  # noqa: TRY003

    def root() -> None:
        reset_plot_props()
        controls = {}
        plotbases = {}
        response_caches = {}
        camera_state = {"static": {}, "animation": {}}
        current = {
            "static": _empty_figure("Loading..."),
            "animation": _empty_figure("Click 'Generate Animation'"),
            "active": "static",
            "animation_generated": False,
        }
        preserve_plotly_view(current["static"], camera_state["static"])
        preserve_plotly_view(current["animation"], camera_state["animation"])
        sidebar, content = page_shell("Nodal Response Viewer")

        def get_plotbase() -> PlotNodalResponse:
            lazy_load = bool(controls["lazy_load"].value)
            if lazy_load not in plotbases:
                plotbases[lazy_load] = PlotNodalResponse(odb_tag, lazy_load=lazy_load)
                response_caches[lazy_load] = {}
            return plotbases[lazy_load]

        def get_response_cache():
            return response_caches[bool(controls["lazy_load"].value)]

        def update_static() -> None:
            plotbase = get_plotbase()
            current["static"] = _plot_cached_static(plotbase, controls, get_response_cache())
            preserve_plotly_view(current["static"], camera_state["static"])
            update_plot(static_plot, current["static"])
            set_message(message, "Static figure updated.")

        def update_animation() -> None:
            plotbase = get_plotbase()
            current["animation"] = _plot_cached_animation(plotbase, controls, get_response_cache())
            current["animation_generated"] = True
            preserve_plotly_view(current["animation"], camera_state["animation"])
            update_plot(animation_plot, current["animation"])
            set_message(message, "Animation generated.")

        def export_current() -> None:
            fig = current["static"] if current["active"] == "static" else current["animation"]
            set_message(message, export_html(fig, controls["export_html_path"].value))

        with sidebar:
            with ui.row().classes("w-full no-wrap").style("gap: 6px;"):
                compact_button("Refresh", update_static)
                compact_button("Exit", shutdown_gui)
            message = ui.label("").style("font-size: 11px; color: #555; min-height: 14px;")

            section_title("Response")
            with control_row():
                controls["resp_type"] = compact_select(
                    "Response Type",
                    ["disp", "vel", "accel", "reaction", "reactionIncInertia", "rayleighForces", "pressure"],
                    "disp",
                )
                controls["resp_dof_text"] = compact_input("Response DOF", "UX, UY, UZ", 'e.g. UX or ["UX","UY"]')
            with control_row():
                controls["view_mode"] = compact_select("View Mode", ["single", "slide"], "single")
                controls["step_dropdown"] = compact_select("Step", ["0", "absMax", "absMin", "Max", "Min"], "absMax")
            with control_row():
                controls["step_text"] = compact_input("Custom Step", "", "optional custom step")
            with control_row():
                controls["unit_symbol"] = compact_input("Unit Symbol", "", "e.g. mm")
                controls["unit_factor"] = compact_number("Unit Factor", 1.0, 0.001)
            with control_row():
                controls["style"] = compact_select(
                    "Style", ["surface", "wireframe", "points", "points_gaussian"], "surface"
                )
                controls["show_outline"] = compact_checkbox("Show Outline", False)
            with control_row():
                controls["show_defo"] = compact_checkbox("Show Defo", True)
                controls["defo_scale"] = compact_input("Defo Scale", "auto", "auto / true / false / 2.0")
            with control_row():
                controls["show_undeformed"] = compact_checkbox("Show Undeformed", False)
                controls["show_max_min"] = compact_checkbox("Show Max/Min", False)
            with control_row():
                controls["show_bc"] = compact_checkbox("Show BC", True)
                controls["bc_scale"] = compact_number("BC Scale", 1.0, 0.1)
            with control_row():
                controls["show_mp_constraint"] = compact_checkbox("Show MP Constraint", False)
                controls["lazy_load"] = compact_checkbox("Lazy Load", False)
            with control_row():
                compact_button("Update Static Figure", update_static)

            section_title("Animation")
            with control_row():
                controls["animation_framerate"] = compact_number("Frame Rate", 8, 1)
            with control_row():
                controls["animation_show_defo"] = compact_checkbox("Show Defo", True)
                controls["animation_defo_scale"] = compact_input("Defo Scale", "auto", "auto / true / false / 2.0")
            with control_row():
                controls["animation_show_undeformed"] = compact_checkbox("Show Undeformed", False)
                controls["animation_show_max_min"] = compact_checkbox("Show Max/Min", False)
            with control_row():
                compact_button("Generate Animation", update_animation)

            section_title("Export")
            with control_row():
                controls["export_html_path"] = compact_input("HTML Path", "nodal_response_viewer.html")
                compact_button("Export HTML", export_current)

            section_title("Global Control")
            subsection_title("General Props")
            with control_row():
                controls["gc_point_size"] = compact_number("Point size", PLOT_ARGS_DEFAULT["point_size"], 0.5)
                controls["gc_line_width"] = compact_number("Line width", PLOT_ARGS_DEFAULT["line_width"], 0.5)
            with control_row():
                controls["gc_scale_factor"] = compact_number("Scale factor", PLOT_ARGS_DEFAULT["scale_factor"], 0.01)
                controls["gc_show_mesh_edges"] = compact_checkbox(
                    "Show mesh edges", PLOT_ARGS_DEFAULT["show_mesh_edges"]
                )
            with control_row():
                controls["gc_mesh_edge_color"] = compact_color("Mesh edge color", PLOT_ARGS_DEFAULT["mesh_edge_color"])
                controls["gc_mesh_edge_width"] = compact_number(
                    "Mesh edge width", PLOT_ARGS_DEFAULT["mesh_edge_width"], 0.1
                )
            with control_row():
                controls["gc_mesh_opacity"] = compact_number("Mesh opacity", PLOT_ARGS_DEFAULT["mesh_opacity"], 0.05)
                controls["gc_theme"] = compact_select("Theme", list(pio.templates.keys()), PLOT_ARGS_DEFAULT["theme"])
            with control_row():
                controls["gc_font_family"] = compact_select(
                    "Font family", FONT_FAMILIES, PLOT_ARGS_DEFAULT["font_family"]
                )
                controls["gc_font_size"] = compact_number("Font size", PLOT_ARGS_DEFAULT["font_size"], 1)
            with control_row():
                controls["gc_title_font_size"] = compact_number(
                    "Title font size", PLOT_ARGS_DEFAULT["title_font_size"], 1
                )
                controls["gc_cmap"] = compact_select(
                    "CMAP", ["__default__", "", *px.colors.named_colorscales()], "__default__"
                )
            subsection_title("Constraint Colors")
            with control_row():
                controls["gc_color_constraint"] = compact_color(
                    "Constraint color", PLOT_ARGS_DEFAULT["color_constraint"]
                )
                controls["gc_color_bc"] = compact_color("BC color", PLOT_ARGS_DEFAULT["color_bc"])

        with content:
            with ui.tabs().classes("w-full") as tabs:
                static_tab = ui.tab("Static Figure")
                animation_tab = ui.tab("Animation")
            with ui.tab_panels(tabs, value=static_tab).classes("w-full").style("height: calc(100vh - 48px);") as panels:
                with ui.tab_panel(static_tab):
                    static_plot = ui.plotly(current["static"]).classes("w-full").style("height: calc(100vh - 64px);")
                    static_plot.on(
                        "plotly_relayout", lambda e: update_camera_state(camera_state["static"], e.args), throttle=0.2
                    )
                with ui.tab_panel(animation_tab):
                    animation_plot = (
                        ui.plotly(current["animation"]).classes("w-full").style("height: calc(100vh - 64px);")
                    )
                    animation_plot.on(
                        "plotly_relayout",
                        lambda e: update_camera_state(camera_state["animation"], e.args),
                        throttle=0.2,
                    )

            def on_panel_change() -> None:
                current["active"] = "animation" if panels.value == animation_tab else "static"
                if current["active"] == "animation" and not current["animation_generated"]:
                    update_animation()

            panels.on_value_change(lambda _: on_panel_change())
            update_static()

        def refresh_active() -> None:
            update_animation() if current["active"] == "animation" else update_static()

        common_controls = {
            "resp_type",
            "resp_dof_text",
            "style",
            "show_outline",
            "show_bc",
            "bc_scale",
            "show_mp_constraint",
            "lazy_load",
            "unit_symbol",
            "unit_factor",
        }
        static_controls = {
            "view_mode",
            "step_dropdown",
            "step_text",
            "show_defo",
            "defo_scale",
            "show_undeformed",
            "show_max_min",
        }
        animation_controls = {name for name in controls if name.startswith("animation_")}
        global_controls = {name for name in controls if name.startswith("gc_")}
        bind_auto_refresh({name: controls[name] for name in common_controls}, refresh_active)
        bind_auto_refresh({name: controls[name] for name in static_controls}, update_static)
        bind_auto_refresh({name: controls[name] for name in animation_controls}, update_animation)
        bind_auto_refresh({name: controls[name] for name in global_controls}, refresh_active)

    run_gui("Nodal Response Viewer", host, port, debug, auto_open, root=root)
