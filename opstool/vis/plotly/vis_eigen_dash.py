from __future__ import annotations

import os
import threading
import webbrowser

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from flask import request

try:
    from dash import Dash, Input, Output, State, dcc, html
except ImportError as e:
    raise ImportError("Dash is required for plot_eigen_dash(). Install it with: pip install dash") from e  # noqa: TRY003

from ...post import load_eigen_data, load_linear_buckling_data
from .plot_utils import PLOT_ARGS_DEFAULT, reset_plot_props, set_plot_colors, set_plot_props
from .vis_eigen import plot_eigen, plot_eigen_animation, plot_eigen_table

# =============================================================================
# Default values
# =============================================================================
GLOBAL_DEFAULTS = PLOT_ARGS_DEFAULT.copy()

LOCAL_DEFAULTS = {
    "shape_view_type": "shape",  # shape | table (table only for eigen mode)
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

PLOTLY_THEMES = list(pio.templates.keys())
PLOTLY_COLORSCALES = px.colors.named_colorscales()

FONT_FAMILIES = [
    "Arial",
    "Arial Black",
    "Calibri",
    "Cambria",
    "Candara",
    "Comic Sans MS",
    "Courier New",
    "Georgia",
    "Helvetica",
    "Impact",
    "Segoe UI",
    "Tahoma",
    "Times New Roman",
    "Trebuchet MS",
    "Verdana",
]


# =============================================================================
# Helper functions
# =============================================================================
def _open_browser(url: str) -> None:
    webbrowser.open_new(url)


def _is_on(value: list[str] | None) -> bool:
    return bool(value and "on" in value)


def _on_value(flag: bool) -> list[str]:
    return ["on"] if flag else []


def _clean_optional_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_cmap_value(value, default_value=None):
    """
    Normalize Dash cmap input.

    Returns
    -------
    list | str | None
    """
    if value is None:
        return None
    if value == "":
        return None
    if value == "__default__":
        return default_value
    if isinstance(value, (list, tuple)):
        return value
    return str(value).strip()


def _section_title(text: str) -> html.Div:
    return html.Div(
        text,
        style={
            "fontWeight": "700",
            "fontSize": "14px",
            "marginTop": "10px",
            "marginBottom": "5px",
            "paddingBottom": "3px",
            "borderBottom": "1px solid #d9d9d9",
        },
    )


def _subsection_title(text: str) -> html.Div:
    return html.Div(
        text,
        style={
            "fontWeight": "600",
            "fontSize": "12px",
            "marginTop": "6px",
            "marginBottom": "4px",
            "color": "#444",
        },
    )


def _bool_switch(label: str, comp_id: str, value: bool = False) -> html.Div:
    return html.Div(
        dcc.Checklist(
            id=comp_id,
            options=[{"label": label, "value": "on"}],
            value=_on_value(value),
            style={"margin": "0"},
            inputStyle={"marginRight": "5px"},
            labelStyle={"fontSize": "12px", "margin": "0"},
        ),
        style={"marginBottom": "2px"},
    )


def _number_input(label: str, comp_id: str, value: float | int, step: float = 0.1) -> html.Div:
    return html.Div(
        [
            html.Label(
                label,
                style={"display": "block", "fontSize": "11px", "marginBottom": "1px"},
            ),
            dcc.Input(
                id=comp_id,
                type="number",
                value=value,
                step=step,
                style={
                    "width": "100%",
                    "padding": "3px 6px",
                    "boxSizing": "border-box",
                    "fontSize": "12px",
                    "height": "28px",
                },
            ),
        ],
        style={"marginBottom": "4px"},
    )


def _text_input(label: str, comp_id: str, value: str = "", placeholder: str = "") -> html.Div:
    return html.Div(
        [
            html.Label(
                label,
                style={"display": "block", "fontSize": "11px", "marginBottom": "1px"},
            ),
            dcc.Input(
                id=comp_id,
                type="text",
                value=value,
                placeholder=placeholder,
                style={
                    "width": "100%",
                    "padding": "3px 6px",
                    "boxSizing": "border-box",
                    "fontSize": "12px",
                    "height": "28px",
                },
            ),
        ],
        style={"marginBottom": "4px"},
    )


def _dropdown(label: str, comp_id: str, options: list[dict[str, str]], value: str, clearable: bool = False) -> html.Div:
    return html.Div(
        [
            html.Label(
                label,
                style={"display": "block", "fontSize": "11px", "marginBottom": "1px"},
            ),
            dcc.Dropdown(
                id=comp_id,
                options=options,
                value=value,
                clearable=clearable,
                style={"fontSize": "12px"},
            ),
        ],
        style={"marginBottom": "4px"},
    )


def _row(*children, gap: str = "6px", margin_bottom: str = "2px") -> html.Div:
    row_children = []
    for child in children:
        row_children.append(html.Div(child, style={"flex": "1", "minWidth": "0"}))
    return html.Div(
        row_children,
        style={
            "display": "flex",
            "gap": gap,
            "marginBottom": margin_bottom,
            "alignItems": "end",
        },
    )


def _button(label: str, comp_id: str, width: str = "100%") -> html.Button:
    return html.Button(
        label,
        id=comp_id,
        n_clicks=0,
        style={
            "width": width,
            "padding": "5px 8px",
            "cursor": "pointer",
            "fontSize": "12px",
        },
    )


def _get_saved_mode_count(odb_tag: int | str, mode: str, interpolate_beam: bool) -> int:
    """
    Read the maximum saved mode index from an existing file only.
    """
    mode_name = (mode or "eigen").lower()

    if mode_name == "eigen":
        modal_props, _, _, _ = load_eigen_data(
            odb_tag=odb_tag,
            mode_tag=1,
            resave=False,
            interpolate_beam=interpolate_beam,
        )
    elif mode_name == "buckling":
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


def _apply_global_plot_settings(
    gc_point_size,
    gc_line_width,
    gc_scale_factor,
    gc_show_mesh_edges,
    gc_mesh_edge_color,
    gc_mesh_edge_width,
    gc_mesh_opacity,
    gc_theme,
    gc_font_family,
    gc_font_size,
    gc_title_font_size,
    gc_cmap,
    gc_color_constraint,
    gc_color_bc,
) -> None:
    cmap_value = _normalize_cmap_value(gc_cmap, default_value=PLOT_ARGS_DEFAULT["cmap"])

    set_plot_props(
        point_size=float(gc_point_size) if gc_point_size is not None else GLOBAL_DEFAULTS["point_size"],
        line_width=float(gc_line_width) if gc_line_width is not None else GLOBAL_DEFAULTS["line_width"],
        theme=gc_theme or GLOBAL_DEFAULTS["theme"],
        scale_factor=float(gc_scale_factor) if gc_scale_factor is not None else GLOBAL_DEFAULTS["scale_factor"],
        show_mesh_edges=_is_on(gc_show_mesh_edges),
        mesh_edge_color=gc_mesh_edge_color or GLOBAL_DEFAULTS["mesh_edge_color"],
        mesh_edge_width=float(gc_mesh_edge_width)
        if gc_mesh_edge_width is not None
        else GLOBAL_DEFAULTS["mesh_edge_width"],
        mesh_opacity=float(gc_mesh_opacity) if gc_mesh_opacity is not None else GLOBAL_DEFAULTS["mesh_opacity"],
        font_family=gc_font_family or GLOBAL_DEFAULTS["font_family"],
        font_size=int(gc_font_size) if gc_font_size is not None else GLOBAL_DEFAULTS["font_size"],
        title_font_size=int(gc_title_font_size)
        if gc_title_font_size is not None
        else GLOBAL_DEFAULTS["title_font_size"],
        cmap=cmap_value,
    )

    set_plot_colors(
        constraint=gc_color_constraint or GLOBAL_DEFAULTS["color_constraint"],
        bc=gc_color_bc or GLOBAL_DEFAULTS["color_bc"],
        cmap=cmap_value,
    )


def _make_initial_shape_figure(odb_tag: int | str, mode: str) -> go.Figure:
    return plot_eigen(
        mode_tags=[LOCAL_DEFAULTS["mode_tag_start"], LOCAL_DEFAULTS["mode_tag_end"]],
        odb_tag=odb_tag,
        subplots=LOCAL_DEFAULTS["subplots"],
        scale=LOCAL_DEFAULTS["scale"],
        show_outline=LOCAL_DEFAULTS["show_outline"],
        show_origin=LOCAL_DEFAULTS["show_origin"],
        style=LOCAL_DEFAULTS["style"],
        show_bc=LOCAL_DEFAULTS["show_bc"],
        bc_scale=LOCAL_DEFAULTS["bc_scale"],
        show_mp_constraint=LOCAL_DEFAULTS["show_mp_constraint"],
        solver="-genBandArpack",
        mode=mode,
        interpolate_beam=LOCAL_DEFAULTS["interpolate_beam"],
    )


# =============================================================================
# Layout
# =============================================================================
def _make_layout(init_shape_fig: go.Figure, mode: str) -> html.Div:
    cmap_options = [{"label": "Default", "value": "__default__"}, {"label": "(None)", "value": ""}] + [
        {"label": x, "value": x} for x in PLOTLY_COLORSCALES
    ]

    if mode == "eigen":
        display_options = [
            {"label": "Mode Shapes", "value": "shape"},
            {"label": "Modal Table", "value": "table"},
        ]
        default_display = LOCAL_DEFAULTS["shape_view_type"]
    else:
        display_options = [{"label": "Mode Shapes", "value": "shape"}]
        default_display = "shape"

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(
                                f"{mode.capitalize()} Viewer",
                                style={"margin": "0", "fontSize": "17px", "flex": "1"},
                            ),
                            _button("Refresh", "toolbar_refresh_btn", width="78px"),
                            _button("Exit", "toolbar_exit_btn", width="78px"),
                        ],
                        style={
                            "display": "flex",
                            "gap": "6px",
                            "alignItems": "center",
                            "marginBottom": "6px",
                            "paddingBottom": "6px",
                            "borderBottom": "1px solid #d9d9d9",
                        },
                    ),
                    html.Div(
                        id="toolbar_message",
                        style={
                            "fontSize": "11px",
                            "color": "#555",
                            "marginBottom": "4px",
                            "minHeight": "14px",
                        },
                    ),
                    _section_title("Export"),
                    _row(
                        _text_input(
                            "HTML Path",
                            "toolbar_export_html_path",
                            value=LOCAL_DEFAULTS["export_html_path"],
                            placeholder="e.g. results/eigen_viewer.html",
                        ),
                        html.Div(_button("Export HTML", "toolbar_export_html_btn"), style={"paddingBottom": "4px"}),
                        gap="6px",
                    ),
                    _section_title("Mode Shapes / Table"),
                    _row(
                        _dropdown("Display", "shape_view_type", display_options, default_display),
                        html.Div(),
                    ),
                    _row(
                        _number_input("Start mode", "mode_tag_start", LOCAL_DEFAULTS["mode_tag_start"], 1),
                        _number_input("End mode", "mode_tag_end", LOCAL_DEFAULTS["mode_tag_end"], 1),
                    ),
                    _row(
                        _number_input("Scale", "scale", LOCAL_DEFAULTS["scale"], 0.1),
                        _dropdown(
                            "Style",
                            "style",
                            [
                                {"label": "Surface", "value": "surface"},
                                {"label": "Wireframe", "value": "wireframe"},
                            ],
                            LOCAL_DEFAULTS["style"],
                        ),
                    ),
                    _row(
                        _bool_switch("Subplots", "subplots", LOCAL_DEFAULTS["subplots"]),
                        _bool_switch("Interpolate beam", "interpolate_beam", LOCAL_DEFAULTS["interpolate_beam"]),
                        margin_bottom="0px",
                    ),
                    _row(
                        _bool_switch("Show outline", "show_outline", LOCAL_DEFAULTS["show_outline"]),
                        _bool_switch("Show origin", "show_origin", LOCAL_DEFAULTS["show_origin"]),
                        margin_bottom="0px",
                    ),
                    _row(
                        _bool_switch("Show BC", "show_bc", LOCAL_DEFAULTS["show_bc"]),
                        _number_input("BC scale", "bc_scale", LOCAL_DEFAULTS["bc_scale"], 0.1),
                    ),
                    _row(
                        _bool_switch("Show MP constraint", "show_mp_constraint", LOCAL_DEFAULTS["show_mp_constraint"]),
                        html.Div(),
                        margin_bottom="0px",
                    ),
                    _row(
                        _button("Update Shape / Table", "update_shape_btn"),
                        html.Div(),
                    ),
                    _section_title("Animation"),
                    _row(
                        _number_input("Mode", "animation_mode_tag", LOCAL_DEFAULTS["animation_mode_tag"], 1),
                        html.Div(),
                    ),
                    _row(
                        _number_input("Scale", "animation_scale", LOCAL_DEFAULTS["animation_scale"], 0.1),
                        _dropdown(
                            "Style",
                            "animation_style",
                            [
                                {"label": "Surface", "value": "surface"},
                                {"label": "Wireframe", "value": "wireframe"},
                            ],
                            LOCAL_DEFAULTS["animation_style"],
                        ),
                    ),
                    _row(
                        _number_input("Cycles", "animation_n_cycle", LOCAL_DEFAULTS["animation_n_cycle"], 1),
                        _number_input("Frame rate", "animation_framerate", LOCAL_DEFAULTS["animation_framerate"], 1),
                    ),
                    _row(
                        _bool_switch(
                            "Interpolate beam",
                            "animation_interpolate_beam",
                            LOCAL_DEFAULTS["animation_interpolate_beam"],
                        ),
                        _bool_switch(
                            "Show outline", "animation_show_outline", LOCAL_DEFAULTS["animation_show_outline"]
                        ),
                        margin_bottom="0px",
                    ),
                    _row(
                        _bool_switch("Show origin", "animation_show_origin", LOCAL_DEFAULTS["animation_show_origin"]),
                        _bool_switch("Show BC", "animation_show_bc", LOCAL_DEFAULTS["animation_show_bc"]),
                        margin_bottom="0px",
                    ),
                    _row(
                        _number_input("BC scale", "animation_bc_scale", LOCAL_DEFAULTS["animation_bc_scale"], 0.1),
                        _bool_switch(
                            "Show MP constraint",
                            "animation_show_mp_constraint",
                            LOCAL_DEFAULTS["animation_show_mp_constraint"],
                        ),
                    ),
                    _row(
                        _button("Generate Animation", "generate_animation_btn"),
                        html.Div(),
                    ),
                    _section_title("Reset"),
                    _row(
                        _button("Reset All Controls", "reset_all_btn"),
                        _button("Reset Global Control", "gc_reset_btn"),
                    ),
                    _section_title("Global Control"),
                    _subsection_title("General Props"),
                    _row(
                        _number_input("Point size", "gc_point_size", GLOBAL_DEFAULTS["point_size"], 0.5),
                        _number_input("Line width", "gc_line_width", GLOBAL_DEFAULTS["line_width"], 0.5),
                    ),
                    _row(
                        _number_input("Scale factor", "gc_scale_factor", GLOBAL_DEFAULTS["scale_factor"], 0.01),
                        _bool_switch("Show mesh edges", "gc_show_mesh_edges", GLOBAL_DEFAULTS["show_mesh_edges"]),
                    ),
                    _row(
                        _text_input("Mesh edge color", "gc_mesh_edge_color", GLOBAL_DEFAULTS["mesh_edge_color"]),
                        _number_input("Mesh edge width", "gc_mesh_edge_width", GLOBAL_DEFAULTS["mesh_edge_width"], 0.1),
                    ),
                    _row(
                        _number_input("Mesh opacity", "gc_mesh_opacity", GLOBAL_DEFAULTS["mesh_opacity"], 0.05),
                        _dropdown(
                            "Theme",
                            "gc_theme",
                            [{"label": x, "value": x} for x in PLOTLY_THEMES],
                            GLOBAL_DEFAULTS["theme"],
                        ),
                    ),
                    _row(
                        _dropdown(
                            "Font family",
                            "gc_font_family",
                            [{"label": x, "value": x} for x in FONT_FAMILIES],
                            GLOBAL_DEFAULTS["font_family"],
                        ),
                        _number_input("Font size", "gc_font_size", GLOBAL_DEFAULTS["font_size"], 1),
                    ),
                    _row(
                        _number_input("Title font size", "gc_title_font_size", GLOBAL_DEFAULTS["title_font_size"], 1),
                        _dropdown("CMAP", "gc_cmap", cmap_options, "__default__"),
                    ),
                    _subsection_title("Constraint Colors"),
                    _row(
                        _text_input("Constraint color", "gc_color_constraint", GLOBAL_DEFAULTS["color_constraint"]),
                        _text_input("BC color", "gc_color_bc", GLOBAL_DEFAULTS["color_bc"]),
                    ),
                ],
                style={
                    "width": "355px",
                    "minWidth": "355px",
                    "maxWidth": "355px",
                    "height": "100vh",
                    "overflowY": "auto",
                    "padding": "10px",
                    "borderRight": "1px solid #ddd",
                    "boxSizing": "border-box",
                    "backgroundColor": "#fafafa",
                },
            ),
            html.Div(
                [
                    dcc.Tabs(
                        id="main_view_tab",
                        value="shape_tab",
                        children=[
                            dcc.Tab(
                                label="Mode Shapes / Table",
                                value="shape_tab",
                                children=[
                                    dcc.Graph(
                                        id="shape_graph",
                                        figure=init_shape_fig,
                                        style={"height": "92vh", "width": "100%"},
                                        config={"displaylogo": False},
                                    )
                                ],
                            ),
                            dcc.Tab(
                                label="Animation",
                                value="animation_tab",
                                children=[
                                    dcc.Graph(
                                        id="animation_graph",
                                        figure=go.Figure(),
                                        style={"height": "92vh", "width": "100%"},
                                        config={"displaylogo": False},
                                    )
                                ],
                            ),
                        ],
                    ),
                    dcc.Store(id="exit_app_store"),
                    dcc.Store(id="refresh_store"),
                ],
                style={
                    "flex": "1",
                    "height": "100vh",
                    "overflow": "hidden",
                },
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "row",
            "width": "100%",
            "height": "100vh",
            "margin": "0",
            "padding": "0",
        },
    )


# =============================================================================
# Main app
# =============================================================================
def plot_eigen_dash(
    odb_tag: int | str,
    mode: str = "eigen",
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    auto_open: bool = True,
) -> None:
    """
    High-performance Dash app for modal-shape, modal-table, and animation
    visualization from an already saved eigen/buckling file.

    Parameters
    ----------
    odb_tag : int | str
        Existing saved data tag. Must not be None.
    mode : str, default: "eigen"
        Analysis type. Must be either "eigen" or "buckling".
        Only the selected analysis type will appear in the UI.

    Notes
    -----
    - The app never saves or regenerates data.
    - Modal table is only available for eigen mode because the underlying
      plot_eigen_table(...) function uses load_eigen_data(...).
    """
    if odb_tag is None:
        raise ValueError(  # noqa: TRY003
            "plot_eigen_dash requires a non-None odb_tag. Please save the data first and then pass its tag."
        )

    mode = (mode or "eigen").lower()
    if mode not in {"eigen", "buckling"}:
        raise ValueError("mode must be either 'eigen' or 'buckling'")  # noqa: TRY003

    # Cache saved mode counts once to avoid repeated file reads.
    # Only the selected analysis type is loaded.
    saved_mode_counts = {
        True: _get_saved_mode_count(odb_tag, mode, True),
        False: _get_saved_mode_count(odb_tag, mode, False),
    }

    app = Dash(__name__)
    server = app.server

    @server.route("/shutdown", methods=["POST"])
    def _shutdown_server():
        func = request.environ.get("werkzeug.server.shutdown")
        if func is not None:
            func()
            return "Server shutting down..."
        return "Shutdown not available.", 500

    init_shape_fig = _make_initial_shape_figure(odb_tag, mode=mode)
    app.layout = _make_layout(init_shape_fig, mode=mode)

    @app.callback(
        Output("mode_tag_start", "value"),
        Output("mode_tag_end", "value"),
        Output("shape_view_type", "value"),
        Output("subplots", "value"),
        Output("scale", "value"),
        Output("show_outline", "value"),
        Output("show_origin", "value"),
        Output("style", "value"),
        Output("show_bc", "value"),
        Output("bc_scale", "value"),
        Output("show_mp_constraint", "value"),
        Output("interpolate_beam", "value"),
        Output("animation_mode_tag", "value"),
        Output("animation_n_cycle", "value"),
        Output("animation_framerate", "value"),
        Output("animation_scale", "value"),
        Output("animation_show_outline", "value"),
        Output("animation_show_origin", "value"),
        Output("animation_style", "value"),
        Output("animation_show_bc", "value"),
        Output("animation_bc_scale", "value"),
        Output("animation_show_mp_constraint", "value"),
        Output("animation_interpolate_beam", "value"),
        Output("toolbar_export_html_path", "value"),
        Output("gc_point_size", "value"),
        Output("gc_line_width", "value"),
        Output("gc_scale_factor", "value"),
        Output("gc_show_mesh_edges", "value"),
        Output("gc_mesh_edge_color", "value"),
        Output("gc_mesh_edge_width", "value"),
        Output("gc_mesh_opacity", "value"),
        Output("gc_theme", "value"),
        Output("gc_font_family", "value"),
        Output("gc_font_size", "value"),
        Output("gc_title_font_size", "value"),
        Output("gc_cmap", "value"),
        Output("gc_color_constraint", "value"),
        Output("gc_color_bc", "value"),
        Output("toolbar_message", "children", allow_duplicate=True),
        Input("reset_all_btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_all_controls(n_clicks: int):
        reset_plot_props()
        reset_display = LOCAL_DEFAULTS["shape_view_type"] if mode == "eigen" else "shape"
        return (
            LOCAL_DEFAULTS["mode_tag_start"],
            LOCAL_DEFAULTS["mode_tag_end"],
            reset_display,
            _on_value(LOCAL_DEFAULTS["subplots"]),
            LOCAL_DEFAULTS["scale"],
            _on_value(LOCAL_DEFAULTS["show_outline"]),
            _on_value(LOCAL_DEFAULTS["show_origin"]),
            LOCAL_DEFAULTS["style"],
            _on_value(LOCAL_DEFAULTS["show_bc"]),
            LOCAL_DEFAULTS["bc_scale"],
            _on_value(LOCAL_DEFAULTS["show_mp_constraint"]),
            _on_value(LOCAL_DEFAULTS["interpolate_beam"]),
            LOCAL_DEFAULTS["animation_mode_tag"],
            LOCAL_DEFAULTS["animation_n_cycle"],
            LOCAL_DEFAULTS["animation_framerate"],
            LOCAL_DEFAULTS["animation_scale"],
            _on_value(LOCAL_DEFAULTS["animation_show_outline"]),
            _on_value(LOCAL_DEFAULTS["animation_show_origin"]),
            LOCAL_DEFAULTS["animation_style"],
            _on_value(LOCAL_DEFAULTS["animation_show_bc"]),
            LOCAL_DEFAULTS["animation_bc_scale"],
            _on_value(LOCAL_DEFAULTS["animation_show_mp_constraint"]),
            _on_value(LOCAL_DEFAULTS["animation_interpolate_beam"]),
            LOCAL_DEFAULTS["export_html_path"],
            GLOBAL_DEFAULTS["point_size"],
            GLOBAL_DEFAULTS["line_width"],
            GLOBAL_DEFAULTS["scale_factor"],
            _on_value(GLOBAL_DEFAULTS["show_mesh_edges"]),
            GLOBAL_DEFAULTS["mesh_edge_color"],
            GLOBAL_DEFAULTS["mesh_edge_width"],
            GLOBAL_DEFAULTS["mesh_opacity"],
            GLOBAL_DEFAULTS["theme"],
            GLOBAL_DEFAULTS["font_family"],
            GLOBAL_DEFAULTS["font_size"],
            GLOBAL_DEFAULTS["title_font_size"],
            "__default__",
            GLOBAL_DEFAULTS["color_constraint"],
            GLOBAL_DEFAULTS["color_bc"],
            "All controls have been reset.",
        )

    @app.callback(
        Output("gc_point_size", "value", allow_duplicate=True),
        Output("gc_line_width", "value", allow_duplicate=True),
        Output("gc_scale_factor", "value", allow_duplicate=True),
        Output("gc_show_mesh_edges", "value", allow_duplicate=True),
        Output("gc_mesh_edge_color", "value", allow_duplicate=True),
        Output("gc_mesh_edge_width", "value", allow_duplicate=True),
        Output("gc_mesh_opacity", "value", allow_duplicate=True),
        Output("gc_theme", "value", allow_duplicate=True),
        Output("gc_font_family", "value", allow_duplicate=True),
        Output("gc_font_size", "value", allow_duplicate=True),
        Output("gc_title_font_size", "value", allow_duplicate=True),
        Output("gc_cmap", "value", allow_duplicate=True),
        Output("gc_color_constraint", "value", allow_duplicate=True),
        Output("gc_color_bc", "value", allow_duplicate=True),
        Output("toolbar_message", "children", allow_duplicate=True),
        Input("gc_reset_btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_global_controls(n_clicks: int):
        reset_plot_props()
        return (
            GLOBAL_DEFAULTS["point_size"],
            GLOBAL_DEFAULTS["line_width"],
            GLOBAL_DEFAULTS["scale_factor"],
            _on_value(GLOBAL_DEFAULTS["show_mesh_edges"]),
            GLOBAL_DEFAULTS["mesh_edge_color"],
            GLOBAL_DEFAULTS["mesh_edge_width"],
            GLOBAL_DEFAULTS["mesh_opacity"],
            GLOBAL_DEFAULTS["theme"],
            GLOBAL_DEFAULTS["font_family"],
            GLOBAL_DEFAULTS["font_size"],
            GLOBAL_DEFAULTS["title_font_size"],
            "__default__",
            GLOBAL_DEFAULTS["color_constraint"],
            GLOBAL_DEFAULTS["color_bc"],
            "Global controls have been reset.",
        )

    @app.callback(
        Output("refresh_store", "data"),
        Output("toolbar_message", "children", allow_duplicate=True),
        Input("toolbar_refresh_btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _refresh_app(n_clicks: int):
        return {"refresh": n_clicks}, "Refresh requested."

    @app.callback(
        Output("toolbar_message", "children", allow_duplicate=True),
        Input("toolbar_export_html_btn", "n_clicks"),
        State("toolbar_export_html_path", "value"),
        State("main_view_tab", "value"),
        State("shape_graph", "figure"),
        State("animation_graph", "figure"),
        prevent_initial_call=True,
    )
    def _export_html(n_clicks: int, export_path: str, active_tab: str, shape_fig: dict, animation_fig: dict) -> str:
        path = (export_path or "").strip()
        if not path:
            raise ValueError("Please provide a valid HTML output path.")  # noqa: TRY003

        if not path.lower().endswith(".html"):
            path += ".html"

        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        fig_dict = shape_fig if active_tab == "shape_tab" else animation_fig
        fig = go.Figure(fig_dict)
        fig.write_html(path, include_plotlyjs=True, full_html=True)
        return f"Exported HTML to: {path}"

    @app.callback(
        Output("exit_app_store", "data"),
        Output("toolbar_message", "children", allow_duplicate=True),
        Input("toolbar_exit_btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _exit_app(n_clicks: int):
        return {"exit": True}, "Exiting application..."

    app.clientside_callback(
        """
        function(data) {
            if (!data || !data.exit) {
                return window.dash_clientside.no_update;
            }
            fetch('/shutdown', {method: 'POST'}).catch(() => {});
            setTimeout(function() {
                window.open('', '_self');
                window.close();
            }, 300);
            return window.dash_clientside.no_update;
        }
        """,
        Output("toolbar_exit_btn", "title"),
        Input("exit_app_store", "data"),
    )

    # -------------------------------------------------------------------------
    # Shape / Table callback
    # -------------------------------------------------------------------------
    @app.callback(
        Output("shape_graph", "figure"),
        Output("toolbar_message", "children"),
        Input("update_shape_btn", "n_clicks"),
        Input("refresh_store", "data"),
        State("shape_view_type", "value"),
        State("mode_tag_start", "value"),
        State("mode_tag_end", "value"),
        State("subplots", "value"),
        State("scale", "value"),
        State("show_outline", "value"),
        State("show_origin", "value"),
        State("style", "value"),
        State("show_bc", "value"),
        State("bc_scale", "value"),
        State("show_mp_constraint", "value"),
        State("interpolate_beam", "value"),
        State("gc_point_size", "value"),
        State("gc_line_width", "value"),
        State("gc_scale_factor", "value"),
        State("gc_show_mesh_edges", "value"),
        State("gc_mesh_edge_color", "value"),
        State("gc_mesh_edge_width", "value"),
        State("gc_mesh_opacity", "value"),
        State("gc_theme", "value"),
        State("gc_font_family", "value"),
        State("gc_font_size", "value"),
        State("gc_title_font_size", "value"),
        State("gc_cmap", "value"),
        State("gc_color_constraint", "value"),
        State("gc_color_bc", "value"),
        prevent_initial_call=True,
    )
    def _update_shape_figure(
        update_shape_btn,
        refresh_data,
        shape_view_type,
        mode_tag_start,
        mode_tag_end,
        subplots,
        scale,
        show_outline,
        show_origin,
        style,
        show_bc,
        bc_scale,
        show_mp_constraint,
        interpolate_beam,
        gc_point_size,
        gc_line_width,
        gc_scale_factor,
        gc_show_mesh_edges,
        gc_mesh_edge_color,
        gc_mesh_edge_width,
        gc_mesh_opacity,
        gc_theme,
        gc_font_family,
        gc_font_size,
        gc_title_font_size,
        gc_cmap,
        gc_color_constraint,
        gc_color_bc,
    ):
        _apply_global_plot_settings(
            gc_point_size,
            gc_line_width,
            gc_scale_factor,
            gc_show_mesh_edges,
            gc_mesh_edge_color,
            gc_mesh_edge_width,
            gc_mesh_opacity,
            gc_theme,
            gc_font_family,
            gc_font_size,
            gc_title_font_size,
            gc_cmap,
            gc_color_constraint,
            gc_color_bc,
        )

        start_mode = int(mode_tag_start) if mode_tag_start is not None else LOCAL_DEFAULTS["mode_tag_start"]
        end_mode = int(mode_tag_end) if mode_tag_end is not None else LOCAL_DEFAULTS["mode_tag_end"]
        if start_mode > end_mode:
            start_mode, end_mode = end_mode, start_mode

        max_saved_mode = saved_mode_counts[_is_on(interpolate_beam)]

        if end_mode > max_saved_mode:
            fig = go.Figure()
            msg = (
                f"Shape view: requested end mode = {end_mode}, but saved file only contains modes up to "
                f"{max_saved_mode}."
            )
            return fig, msg

        if shape_view_type == "table":
            if mode != "eigen":
                fig = go.Figure()
                msg = "Modal table is only supported for eigen mode."
                return fig, msg

            fig = plot_eigen_table(
                mode_tags=[start_mode, end_mode],
                odb_tag=odb_tag,
                solver="-genBandArpack",
            )
            msg = f"Showing eigen modal table for modes {start_mode} to {end_mode}."
            return fig, msg

        fig = plot_eigen(
            mode_tags=[start_mode, end_mode],
            odb_tag=odb_tag,
            subplots=_is_on(subplots),
            scale=float(scale) if scale is not None else LOCAL_DEFAULTS["scale"],
            show_outline=_is_on(show_outline),
            show_origin=_is_on(show_origin),
            style=style or LOCAL_DEFAULTS["style"],
            show_bc=_is_on(show_bc),
            bc_scale=float(bc_scale) if bc_scale is not None else LOCAL_DEFAULTS["bc_scale"],
            show_mp_constraint=_is_on(show_mp_constraint),
            solver="-genBandArpack",
            mode=mode,
            interpolate_beam=_is_on(interpolate_beam),
        )
        msg = f"Showing {mode} mode shapes for modes {start_mode} to {end_mode}."
        return fig, msg

    # -------------------------------------------------------------------------
    # Animation callback
    # -------------------------------------------------------------------------
    @app.callback(
        Output("animation_graph", "figure"),
        Output("toolbar_message", "children", allow_duplicate=True),
        Input("generate_animation_btn", "n_clicks"),
        Input("refresh_store", "data"),
        State("animation_mode_tag", "value"),
        State("animation_n_cycle", "value"),
        State("animation_framerate", "value"),
        State("animation_scale", "value"),
        State("animation_show_outline", "value"),
        State("animation_show_origin", "value"),
        State("animation_style", "value"),
        State("animation_show_bc", "value"),
        State("animation_bc_scale", "value"),
        State("animation_show_mp_constraint", "value"),
        State("animation_interpolate_beam", "value"),
        State("gc_point_size", "value"),
        State("gc_line_width", "value"),
        State("gc_scale_factor", "value"),
        State("gc_show_mesh_edges", "value"),
        State("gc_mesh_edge_color", "value"),
        State("gc_mesh_edge_width", "value"),
        State("gc_mesh_opacity", "value"),
        State("gc_theme", "value"),
        State("gc_font_family", "value"),
        State("gc_font_size", "value"),
        State("gc_title_font_size", "value"),
        State("gc_cmap", "value"),
        State("gc_color_constraint", "value"),
        State("gc_color_bc", "value"),
        prevent_initial_call=True,
    )
    def _update_animation_figure(
        generate_animation_btn,
        refresh_data,
        animation_mode_tag,
        animation_n_cycle,
        animation_framerate,
        animation_scale,
        animation_show_outline,
        animation_show_origin,
        animation_style,
        animation_show_bc,
        animation_bc_scale,
        animation_show_mp_constraint,
        animation_interpolate_beam,
        gc_point_size,
        gc_line_width,
        gc_scale_factor,
        gc_show_mesh_edges,
        gc_mesh_edge_color,
        gc_mesh_edge_width,
        gc_mesh_opacity,
        gc_theme,
        gc_font_family,
        gc_font_size,
        gc_title_font_size,
        gc_cmap,
        gc_color_constraint,
        gc_color_bc,
    ):
        _apply_global_plot_settings(
            gc_point_size,
            gc_line_width,
            gc_scale_factor,
            gc_show_mesh_edges,
            gc_mesh_edge_color,
            gc_mesh_edge_width,
            gc_mesh_opacity,
            gc_theme,
            gc_font_family,
            gc_font_size,
            gc_title_font_size,
            gc_cmap,
            gc_color_constraint,
            gc_color_bc,
        )

        anim_mode_tag = (
            int(animation_mode_tag) if animation_mode_tag is not None else LOCAL_DEFAULTS["animation_mode_tag"]
        )
        max_saved_mode = saved_mode_counts[_is_on(animation_interpolate_beam)]

        if anim_mode_tag > max_saved_mode:
            fig = go.Figure()
            msg = (
                f"Animation view: requested mode = {anim_mode_tag}, but saved file only contains modes up to "
                f"{max_saved_mode}."
            )
            return fig, msg

        fig = plot_eigen_animation(
            mode_tag=anim_mode_tag,
            odb_tag=odb_tag,
            n_cycle=int(animation_n_cycle) if animation_n_cycle is not None else LOCAL_DEFAULTS["animation_n_cycle"],
            framerate=int(animation_framerate)
            if animation_framerate is not None
            else LOCAL_DEFAULTS["animation_framerate"],
            scale=float(animation_scale) if animation_scale is not None else LOCAL_DEFAULTS["animation_scale"],
            solver="-genBandArpack",
            show_outline=_is_on(animation_show_outline),
            show_origin=_is_on(animation_show_origin),
            style=animation_style or LOCAL_DEFAULTS["animation_style"],
            show_bc=_is_on(animation_show_bc),
            bc_scale=float(animation_bc_scale)
            if animation_bc_scale is not None
            else LOCAL_DEFAULTS["animation_bc_scale"],
            show_mp_constraint=_is_on(animation_show_mp_constraint),
            mode=mode,
            interpolate_beam=_is_on(animation_interpolate_beam),
        )
        msg = f"Showing {mode} animation for mode {anim_mode_tag}."
        return fig, msg

    if auto_open:
        url = f"http://{host}:{port}"
        threading.Timer(1.0, _open_browser, args=(url,)).start()

    app.run(host=host, port=port, debug=debug)
