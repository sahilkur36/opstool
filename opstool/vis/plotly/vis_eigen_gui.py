from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from ...post import load_eigen_data, load_linear_buckling_data
from ._nicegui_utils import (
    FONT_FAMILIES,
    bind_auto_refresh,
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
from .vis_eigen import PlotBucklingBase, PlotEigenBase

try:
    from nicegui import ui
except ImportError as e:  # pragma: no cover
    msg = "NiceGUI is required for plot_eigen_gui(). Install it with: pip install nicegui"
    raise ImportError(msg) from e


GLOBAL_DEFAULTS = PLOT_ARGS_DEFAULT.copy()
LOCAL_DEFAULTS = {
    "shape_view_type": "shape",
    "mode_tag_start": 1,
    "mode_tag_end": 3,
    "subplots": False,
    "scale": 1.0,
    "show_outline": False,
    "show_origin": False,
    "style": "surface",
    "show_bc": True,
    "bc_scale": 1.0,
    "show_mp_constraint": False,
    "interpolate_beam": True,
    "animation_mode_tag": 1,
    "animation_n_cycle": 5,
    "animation_framerate": 3,
    "animation_scale": 1.0,
    "animation_show_outline": False,
    "animation_show_origin": False,
    "animation_style": "surface",
    "animation_show_bc": True,
    "animation_bc_scale": 1.0,
    "animation_show_mp_constraint": False,
    "animation_interpolate_beam": True,
    "export_html_path": "eigen_viewer.html",
}


def _normalize_cmap(value):
    if value in (None, "", "__default__"):
        return PLOT_ARGS_DEFAULT["cmap"] if value == "__default__" else None
    return value


def _get_saved_mode_count(odb_tag: int | str, mode: str, interpolate_beam: bool) -> int:
    if mode == "eigen":
        modal_props, _, _, _ = load_eigen_data(
            odb_tag=odb_tag, mode_tag=1, resave=False, interpolate_beam=interpolate_beam
        )
    elif mode == "buckling":
        modal_props, _, _ = load_linear_buckling_data(odb_tag=odb_tag)
    else:
        raise ValueError(f"Unsupported mode: {mode}")  # noqa: TRY003

    if hasattr(modal_props, "coords"):
        if "modeTags" in modal_props.coords:
            return int(modal_props.coords["modeTags"].values.max())
        if "mode" in modal_props.coords:
            return int(modal_props.coords["mode"].values.max())
    if hasattr(modal_props, "dims"):
        for dim in modal_props.dims:
            if "mode" in dim.lower():
                return int(modal_props.sizes[dim])
    raise ValueError("Cannot determine the saved mode count from the existing file.")  # noqa: TRY003


def _get_mode_count_from_modal_props(modal_props) -> int:
    if hasattr(modal_props, "coords"):
        if "modeTags" in modal_props.coords:
            return int(modal_props.coords["modeTags"].values.max())
        if "mode" in modal_props.coords:
            return int(modal_props.coords["mode"].values.max())
    if hasattr(modal_props, "dims"):
        for dim in modal_props.dims:
            if "mode" in dim.lower():
                return int(modal_props.sizes[dim])
    raise ValueError("Cannot determine the saved mode count from the existing file.")  # noqa: TRY003


def _load_cached_eigen_data(odb_tag: int | str, mode: str):
    if mode == "eigen":
        modal_props_i, eigenvectors_i, interp_eigenvectors, model_info_i = load_eigen_data(
            odb_tag=odb_tag,
            mode_tag=1,
            solver="-genBandArpack",
            resave=False,
            interpolate_beam=True,
        )
        modal_props, eigenvectors, _, model_info = load_eigen_data(
            odb_tag=odb_tag,
            mode_tag=1,
            solver="-genBandArpack",
            resave=False,
            interpolate_beam=False,
        )
        return {
            True: (modal_props_i, eigenvectors_i, interp_eigenvectors, model_info_i),
            False: (modal_props, eigenvectors, None, model_info),
        }
    modal_props, eigenvectors, model_info = load_linear_buckling_data(odb_tag=odb_tag)
    return {True: (modal_props, eigenvectors, None, model_info), False: (modal_props, eigenvectors, None, model_info)}


def _make_plotbase(cached_data, mode: str, interpolate_beam: bool):
    modal_props, eigenvectors, interp_eigenvectors, model_info = cached_data[interpolate_beam]
    if mode == "buckling":
        return PlotBucklingBase(model_info, modal_props, eigenvectors)
    return PlotEigenBase(model_info, modal_props, eigenvectors, interp_eigenvectors=interp_eigenvectors)


def _plot_cached_shape(cached_data, mode: str, c, start: int, end: int) -> go.Figure:
    plotbase = _make_plotbase(cached_data, mode, bool(c["interpolate_beam"].value))
    kwargs = {
        "alpha": float(c["scale"].value or LOCAL_DEFAULTS["scale"]),
        "style": c["style"].value or LOCAL_DEFAULTS["style"],
        "show_origin": bool(c["show_origin"].value),
        "show_bc": bool(c["show_bc"].value),
        "bc_scale": float(c["bc_scale"].value or LOCAL_DEFAULTS["bc_scale"]),
        "show_mp_constraint": bool(c["show_mp_constraint"].value),
    }
    if bool(c["subplots"].value):
        plotbase.subplots(start, end, show_outline=bool(c["show_outline"].value), **kwargs)
    else:
        plotbase.plot_slides(start, end, **kwargs)
    return plotbase.update_fig(show_outline=bool(c["show_outline"].value))


def _plot_cached_animation(cached_data, mode: str, c, mode_tag: int) -> go.Figure:
    plotbase = _make_plotbase(cached_data, mode, bool(c["animation_interpolate_beam"].value))
    plotbase.plot_anim(
        mode_tag,
        n_cycle=int(c["animation_n_cycle"].value or LOCAL_DEFAULTS["animation_n_cycle"]),
        framerate=int(c["animation_framerate"].value or LOCAL_DEFAULTS["animation_framerate"]),
        alpha=float(c["animation_scale"].value or LOCAL_DEFAULTS["animation_scale"]),
        style=c["animation_style"].value or LOCAL_DEFAULTS["animation_style"],
        show_origin=bool(c["animation_show_origin"].value),
        show_bc=bool(c["animation_show_bc"].value),
        bc_scale=float(c["animation_bc_scale"].value or LOCAL_DEFAULTS["animation_bc_scale"]),
        show_mp_constraint=bool(c["animation_show_mp_constraint"].value),
    )
    return plotbase.update_fig(show_outline=bool(c["animation_show_outline"].value))


def _plot_cached_table(cached_data, c, start: int, end: int) -> go.Figure:
    plotbase = _make_plotbase(cached_data, "eigen", bool(c["interpolate_beam"].value))
    return plotbase.plot_props_table(start, end)


def _apply_global(c) -> None:
    cmap = _normalize_cmap(c["gc_cmap"].value)
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
        cmap=cmap,
    )
    set_plot_colors(
        constraint=c["gc_color_constraint"].value or GLOBAL_DEFAULTS["color_constraint"],
        bc=c["gc_color_bc"].value or GLOBAL_DEFAULTS["color_bc"],
        cmap=cmap,
    )


def plot_eigen_gui(  # noqa: C901
    odb_tag: int | str,
    mode: str = "eigen",
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    auto_open: bool = True,
) -> None:
    """Launch a NiceGUI app for modal-shape, modal-table, and animation visualization."""
    if odb_tag is None:
        raise ValueError("plot_eigen_gui requires a non-None odb_tag.")  # noqa: TRY003
    mode = (mode or "eigen").lower()
    if mode not in {"eigen", "buckling"}:
        raise ValueError("mode must be either 'eigen' or 'buckling'")  # noqa: TRY003

    cached_data = _load_cached_eigen_data(odb_tag, mode)
    saved_mode_counts = {
        interpolate: _get_mode_count_from_modal_props(data[0]) for interpolate, data in cached_data.items()
    }

    def root() -> None:  # noqa: C901
        reset_plot_props()
        controls = {}
        camera_state = {"shape": {}, "animation": {}}
        current = {"shape": go.Figure(), "animation": go.Figure(), "active": "shape", "animation_generated": False}
        sidebar, content = page_shell(f"{mode.capitalize()} Viewer")

        def update_shape() -> None:
            _apply_global(controls)
            start = int(controls["mode_tag_start"].value or LOCAL_DEFAULTS["mode_tag_start"])
            end = int(controls["mode_tag_end"].value or LOCAL_DEFAULTS["mode_tag_end"])
            if start > end:
                start, end = end, start
            max_mode = saved_mode_counts[bool(controls["interpolate_beam"].value)]
            if end > max_mode:
                current["shape"] = go.Figure()
                msg = f"Requested end mode = {end}, but saved file only contains modes up to {max_mode}."
            elif controls["shape_view_type"].value == "table":
                if mode != "eigen":
                    current["shape"] = go.Figure()
                    msg = "Modal table is only supported for eigen mode."
                else:
                    current["shape"] = _plot_cached_table(cached_data, controls, start, end)
                    msg = f"Showing eigen modal table for modes {start} to {end}."
            else:
                current["shape"] = _plot_cached_shape(cached_data, mode, controls, start, end)
                msg = f"Showing {mode} mode shapes for modes {start} to {end}."
            preserve_plotly_view(current["shape"], camera_state["shape"])
            update_plot(shape_plot, current["shape"])
            set_message(message, msg)

        def update_animation() -> None:
            _apply_global(controls)
            mode_tag = int(controls["animation_mode_tag"].value or LOCAL_DEFAULTS["animation_mode_tag"])
            max_mode = saved_mode_counts[bool(controls["animation_interpolate_beam"].value)]
            if mode_tag > max_mode:
                current["animation"] = go.Figure()
                msg = f"Requested mode = {mode_tag}, but saved file only contains modes up to {max_mode}."
            else:
                current["animation"] = _plot_cached_animation(cached_data, mode, controls, mode_tag)
                msg = f"Showing {mode} animation for mode {mode_tag}."
            current["animation_generated"] = True
            preserve_plotly_view(current["animation"], camera_state["animation"])
            update_plot(animation_plot, current["animation"])
            set_message(message, msg)

        def export_current() -> None:
            fig = current["shape"] if current["active"] == "shape" else current["animation"]
            set_message(message, export_html(fig, controls["export_html_path"].value))

        with sidebar:
            with ui.row().classes("w-full no-wrap").style("gap: 6px;"):
                compact_button("Refresh", update_shape)
                compact_button("Exit", shutdown_gui)
            message = ui.label("").style("font-size: 11px; color: #555; min-height: 14px;")

            section_title("Export")
            with control_row():
                controls["export_html_path"] = compact_input("HTML Path", LOCAL_DEFAULTS["export_html_path"])
                compact_button("Export HTML", export_current)

            section_title("Mode Shapes / Table")
            display_options = ["shape", "table"] if mode == "eigen" else ["shape"]
            with control_row():
                controls["shape_view_type"] = compact_select("Display", display_options, display_options[0])
            with control_row():
                controls["mode_tag_start"] = compact_number("Start mode", 1, 1)
                controls["mode_tag_end"] = compact_number("End mode", 3, 1)
            with control_row():
                controls["scale"] = compact_number("Scale", 1.0, 0.1)
                controls["style"] = compact_select("Style", ["surface", "wireframe"], "surface")
            with control_row():
                controls["subplots"] = compact_checkbox("Subplots", False)
                controls["interpolate_beam"] = compact_checkbox("Interpolate beam", True)
            with control_row():
                controls["show_outline"] = compact_checkbox("Show outline", False)
                controls["show_origin"] = compact_checkbox("Show origin", False)
            with control_row():
                controls["show_bc"] = compact_checkbox("Show BC", True)
                controls["bc_scale"] = compact_number("BC scale", 1.0, 0.1)
            with control_row():
                controls["show_mp_constraint"] = compact_checkbox("Show MP constraint", False)
            with control_row():
                compact_button("Update Shape / Table", update_shape)

            section_title("Animation")
            with control_row():
                controls["animation_mode_tag"] = compact_number("Mode", 1, 1)
            with control_row():
                controls["animation_scale"] = compact_number("Scale", 1.0, 0.1)
                controls["animation_style"] = compact_select("Style", ["surface", "wireframe"], "surface")
            with control_row():
                controls["animation_n_cycle"] = compact_number("Cycles", 5, 1)
                controls["animation_framerate"] = compact_number("Frame rate", 3, 1)
            with control_row():
                controls["animation_interpolate_beam"] = compact_checkbox("Interpolate beam", True)
                controls["animation_show_outline"] = compact_checkbox("Show outline", False)
            with control_row():
                controls["animation_show_origin"] = compact_checkbox("Show origin", False)
                controls["animation_show_bc"] = compact_checkbox("Show BC", True)
            with control_row():
                controls["animation_bc_scale"] = compact_number("BC scale", 1.0, 0.1)
                controls["animation_show_mp_constraint"] = compact_checkbox("Show MP constraint", False)
            with control_row():
                compact_button("Generate Animation", update_animation)

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
                controls["gc_cmap"] = compact_select("CMAP", ["__default__", "", *px.colors.named_colorscales()], "__default__")
            subsection_title("Constraint Colors")
            with control_row():
                controls["gc_color_constraint"] = compact_color("Constraint color", GLOBAL_DEFAULTS["color_constraint"])
                controls["gc_color_bc"] = compact_color("BC color", GLOBAL_DEFAULTS["color_bc"])

        current["shape"] = _plot_cached_shape(
            cached_data,
            mode,
            controls,
            LOCAL_DEFAULTS["mode_tag_start"],
            LOCAL_DEFAULTS["mode_tag_end"],
        )
        preserve_plotly_view(current["shape"], camera_state["shape"])

        with content:
            with ui.tabs().classes("w-full") as tabs:
                shape_tab = ui.tab("Mode Shapes / Table")
                animation_tab = ui.tab("Animation")
            with ui.tab_panels(tabs, value=shape_tab).classes("w-full").style("height: calc(100vh - 48px);") as panels:
                with ui.tab_panel(shape_tab):
                    shape_plot = ui.plotly(current["shape"]).classes("w-full").style("height: calc(100vh - 64px);")
                    shape_plot.on("plotly_relayout", lambda e: update_camera_state(camera_state["shape"], e.args), throttle=0.2)
                with ui.tab_panel(animation_tab):
                    animation_plot = ui.plotly(current["animation"]).classes("w-full").style("height: calc(100vh - 64px);")
                    animation_plot.on("plotly_relayout", lambda e: update_camera_state(camera_state["animation"], e.args), throttle=0.2)

            def _active_changed() -> None:
                current["active"] = "animation" if panels.value == animation_tab else "shape"
                if current["active"] == "animation" and not current["animation_generated"]:
                    update_animation()

            panels.on_value_change(lambda _: _active_changed())

        def refresh_active() -> None:
            update_animation() if current["active"] == "animation" else update_shape()

        shape_controls = {
            "shape_view_type", "mode_tag_start", "mode_tag_end", "scale", "style", "subplots", "interpolate_beam",
            "show_outline", "show_origin", "show_bc", "bc_scale", "show_mp_constraint",
        }
        animation_controls = {name for name in controls if name.startswith("animation_")}
        global_controls = {name for name in controls if name.startswith("gc_")}
        bind_auto_refresh({name: controls[name] for name in shape_controls}, update_shape)
        bind_auto_refresh({name: controls[name] for name in animation_controls}, update_animation)
        bind_auto_refresh({name: controls[name] for name in global_controls}, refresh_active)

    run_gui(f"{mode.capitalize()} Viewer", host, port, debug, auto_open, root=root)
