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
    raise ImportError("Dash is required for plot_model_dash(). Install it with: pip install dash") from e  # noqa: TRY003


from .plot_utils import PLOT_ARGS_DEFAULT, reset_plot_props, set_plot_colors, set_plot_props
from .vis_model import plot_model

# =============================================================================
# Default values
# =============================================================================
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

PLOTLY_THEMES = pio.templates.keys()

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
    """Open the given URL in the default browser."""
    webbrowser.open_new(url)


def _is_on(value: list[str] | None) -> bool:
    """Return True if a checklist-style control is enabled."""
    return bool(value and "on" in value)


def _on_value(flag: bool) -> list[str]:
    """Convert a bool to Dash checklist value."""
    return ["on"] if flag else []


def _clean_optional_text(value) -> str | None:
    """Convert empty text inputs to None."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _section_title(text: str) -> html.Div:
    """Create a section title block."""
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
    """Create a subsection title block."""
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
    """Create a compact boolean control."""
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
    """Create a compact labeled numeric input."""
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
    """Create a compact labeled text input."""
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


def _dropdown(
    label: str,
    comp_id: str,
    options: list[dict[str, str]],
    value: str,
    clearable: bool = False,
) -> html.Div:
    """Create a compact labeled dropdown."""
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
    """Place controls in a compact horizontal row."""
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
    """Create a compact button."""
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


def _color_grid() -> html.Div:
    """Create the compact color configuration grid."""
    items = [
        ("Point", "gc_color_point", GLOBAL_DEFAULTS["color_point"]),
        ("Frame", "gc_color_frame", GLOBAL_DEFAULTS["color_frame"]),
        ("Truss", "gc_color_truss", GLOBAL_DEFAULTS["color_truss"]),
        ("Link", "gc_color_link", GLOBAL_DEFAULTS["color_link"]),
        ("Shell", "gc_color_shell", GLOBAL_DEFAULTS["color_shell"]),
        ("Plane", "gc_color_plane", GLOBAL_DEFAULTS["color_plane"]),
        ("Brick", "gc_color_brick", GLOBAL_DEFAULTS["color_brick"]),
        ("Tet", "gc_color_tet", GLOBAL_DEFAULTS["color_tet"]),
        ("Joint", "gc_color_joint", GLOBAL_DEFAULTS["color_joint"]),
        ("Contact", "gc_color_contact", GLOBAL_DEFAULTS["color_contact"]),
        ("PFEM", "gc_color_pfem", GLOBAL_DEFAULTS["color_pfem"]),
        ("Constraint", "gc_color_constraint", GLOBAL_DEFAULTS["color_constraint"]),
        ("BC", "gc_color_bc", GLOBAL_DEFAULTS["color_bc"]),
    ]

    children = []
    for label, comp_id, value in items:
        children.append(
            html.Div([
                html.Label(label, style={"fontSize": "11px", "marginBottom": "1px"}),
                dcc.Input(
                    id=comp_id,
                    type="text",
                    value=value,
                    style={
                        "width": "100%",
                        "padding": "3px 6px",
                        "boxSizing": "border-box",
                        "fontSize": "12px",
                        "height": "28px",
                    },
                ),
            ])
        )

    return html.Div(
        children=children,
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "5px",
            "marginBottom": "4px",
        },
    )


# =============================================================================
# Layout
# =============================================================================
def _make_layout(init_fig: go.Figure) -> html.Div:
    """Create the full Dash layout."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(
                                "Model Viewer",
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
                            placeholder="e.g. results/model_viewer.html",
                        ),
                        html.Div(
                            _button("Export HTML", "toolbar_export_html_btn"),
                            style={"paddingBottom": "4px"},
                        ),
                        gap="6px",
                    ),
                    _section_title("Appearance"),
                    _row(
                        _dropdown(
                            "Style",
                            "style",
                            [
                                {"label": "Surface", "value": "surface"},
                                {"label": "Wireframe", "value": "wireframe"},
                            ],
                            LOCAL_DEFAULTS["style"],
                        ),
                        _text_input(
                            "Single Color",
                            "color",
                            value=LOCAL_DEFAULTS["color"],
                            placeholder="e.g. black / #1f77b4",
                        ),
                    ),
                    _row(
                        _bool_switch("Show element hover", "show_ele_hover", LOCAL_DEFAULTS["show_ele_hover"]),
                        _bool_switch("Show outline", "show_outline", LOCAL_DEFAULTS["show_outline"]),
                        margin_bottom="0px",
                    ),
                    _section_title("Labels"),
                    _row(
                        _bool_switch(
                            "Show node numbering", "show_node_numbering", LOCAL_DEFAULTS["show_node_numbering"]
                        ),
                        _bool_switch(
                            "Show element numbering", "show_ele_numbering", LOCAL_DEFAULTS["show_ele_numbering"]
                        ),
                        margin_bottom="0px",
                    ),
                    _section_title("Constraints"),
                    _row(
                        _bool_switch("Show BC", "show_bc", LOCAL_DEFAULTS["show_bc"]),
                        _number_input("BC scale", "bc_scale", LOCAL_DEFAULTS["bc_scale"], 0.1),
                    ),
                    _row(
                        _bool_switch("Show link", "show_link", LOCAL_DEFAULTS["show_link"]),
                        _bool_switch("Show MP constraint", "show_mp_constraint", LOCAL_DEFAULTS["show_mp_constraint"]),
                        margin_bottom="0px",
                    ),
                    _row(
                        _bool_switch(
                            "Show constraint DOFs", "show_constraint_dofs", LOCAL_DEFAULTS["show_constraint_dofs"]
                        ),
                        html.Div(),
                        margin_bottom="0px",
                    ),
                    _section_title("Loads"),
                    _row(
                        _bool_switch("Show nodal loads", "show_nodal_loads", LOCAL_DEFAULTS["show_nodal_loads"]),
                        _bool_switch("Show element loads", "show_ele_loads", LOCAL_DEFAULTS["show_ele_loads"]),
                        margin_bottom="0px",
                    ),
                    _row(
                        _number_input("Load scale", "load_scale", LOCAL_DEFAULTS["load_scale"], 0.1),
                        html.Div(),
                    ),
                    _section_title("Local Axes"),
                    _row(
                        _bool_switch("Show local axes", "show_local_axes", LOCAL_DEFAULTS["show_local_axes"]),
                        _number_input("Axes scale", "local_axes_scale", LOCAL_DEFAULTS["local_axes_scale"], 0.1),
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
                        html.Div(),
                    ),
                    _subsection_title("Colormaps"),
                    _row(
                        _dropdown(
                            "CMAP",
                            "gc_cmap",
                            [{"label": "(None)", "value": ""}]
                            + [{"label": x, "value": x} for x in PLOTLY_COLORSCALES if x],
                            GLOBAL_DEFAULTS["cmap"],
                        ),
                        _dropdown(
                            "CMAP_MODEL",
                            "gc_cmap_model",
                            [{"label": "(None)", "value": ""}]
                            + [{"label": x, "value": x} for x in PLOTLY_COLORSCALES if x],
                            GLOBAL_DEFAULTS["cmap_model"],
                        ),
                    ),
                    _subsection_title("Element Colors"),
                    _color_grid(),
                ],
                style={
                    "width": "340px",
                    "minWidth": "340px",
                    "maxWidth": "340px",
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
                    dcc.Graph(
                        id="model_graph",
                        figure=init_fig,
                        style={"height": "100vh", "width": "100%"},
                        config={"displaylogo": False},
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
def plot_model_dash(
    odb_tag: int | str | None = None,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    auto_open: bool = True,
) -> None:
    """
    Launch a Dash application for interactive geometric model visualization.

    Features
    --------
    - Compact left control panel
    - Right-side Plotly figure
    - Top toolbar with Refresh and Exit
    - Export section placed before Appearance
    - Reset All Controls and Reset Global Control
    - Dropdown selection for CMAP, CMAP_MODEL, and font family
    - Compact one-row grouping for related controls

    Parameters
    ----------
    odb_tag : int | str | None, default: None
        Tag of the output database (ODB) to be visualized.
        If None, model data will be extracted from the current running memory.

    host : str, default: "127.0.0.1"
        Host address for the Dash server.

    port : int, default: 8050
        Port number for the Dash server.

    debug : bool, default: False
        Whether to start Dash in debug mode.

    auto_open : bool, default: True
        Whether to automatically open the app in the default browser.

    Returns
    -------
    None
    """
    app = Dash(__name__)
    server = app.server

    @server.route("/shutdown", methods=["POST"])
    def _shutdown_server():
        """
        Stop the local Werkzeug server.

        This works when Dash is running with the Werkzeug development server.
        """
        func = request.environ.get("werkzeug.server.shutdown")
        if func is not None:
            func()
            return "Server shutting down..."
        return "Shutdown not available.", 500

    init_fig = plot_model(odb_tag=odb_tag)
    app.layout = _make_layout(init_fig)

    @app.callback(
        Output("style", "value"),
        Output("color", "value"),
        Output("show_ele_hover", "value"),
        Output("show_outline", "value"),
        Output("show_node_numbering", "value"),
        Output("show_ele_numbering", "value"),
        Output("show_bc", "value"),
        Output("bc_scale", "value"),
        Output("show_link", "value"),
        Output("show_mp_constraint", "value"),
        Output("show_constraint_dofs", "value"),
        Output("show_nodal_loads", "value"),
        Output("show_ele_loads", "value"),
        Output("load_scale", "value"),
        Output("show_local_axes", "value"),
        Output("local_axes_scale", "value"),
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
        Output("gc_cmap_model", "value"),
        Output("gc_color_point", "value"),
        Output("gc_color_frame", "value"),
        Output("gc_color_truss", "value"),
        Output("gc_color_link", "value"),
        Output("gc_color_shell", "value"),
        Output("gc_color_plane", "value"),
        Output("gc_color_brick", "value"),
        Output("gc_color_tet", "value"),
        Output("gc_color_joint", "value"),
        Output("gc_color_contact", "value"),
        Output("gc_color_pfem", "value"),
        Output("gc_color_constraint", "value"),
        Output("gc_color_bc", "value"),
        Output("toolbar_message", "children"),
        Input("reset_all_btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_all_controls(n_clicks: int):
        """Reset all local and global controls to their default values."""
        reset_plot_props()
        return (
            LOCAL_DEFAULTS["style"],
            LOCAL_DEFAULTS["color"],
            _on_value(LOCAL_DEFAULTS["show_ele_hover"]),
            _on_value(LOCAL_DEFAULTS["show_outline"]),
            _on_value(LOCAL_DEFAULTS["show_node_numbering"]),
            _on_value(LOCAL_DEFAULTS["show_ele_numbering"]),
            _on_value(LOCAL_DEFAULTS["show_bc"]),
            LOCAL_DEFAULTS["bc_scale"],
            _on_value(LOCAL_DEFAULTS["show_link"]),
            _on_value(LOCAL_DEFAULTS["show_mp_constraint"]),
            _on_value(LOCAL_DEFAULTS["show_constraint_dofs"]),
            _on_value(LOCAL_DEFAULTS["show_nodal_loads"]),
            _on_value(LOCAL_DEFAULTS["show_ele_loads"]),
            LOCAL_DEFAULTS["load_scale"],
            _on_value(LOCAL_DEFAULTS["show_local_axes"]),
            LOCAL_DEFAULTS["local_axes_scale"],
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
            GLOBAL_DEFAULTS["cmap"],
            GLOBAL_DEFAULTS["cmap_model"],
            GLOBAL_DEFAULTS["color_point"],
            GLOBAL_DEFAULTS["color_frame"],
            GLOBAL_DEFAULTS["color_truss"],
            GLOBAL_DEFAULTS["color_link"],
            GLOBAL_DEFAULTS["color_shell"],
            GLOBAL_DEFAULTS["color_plane"],
            GLOBAL_DEFAULTS["color_brick"],
            GLOBAL_DEFAULTS["color_tet"],
            GLOBAL_DEFAULTS["color_joint"],
            GLOBAL_DEFAULTS["color_contact"],
            GLOBAL_DEFAULTS["color_pfem"],
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
        Output("gc_cmap_model", "value", allow_duplicate=True),
        Output("gc_color_point", "value", allow_duplicate=True),
        Output("gc_color_frame", "value", allow_duplicate=True),
        Output("gc_color_truss", "value", allow_duplicate=True),
        Output("gc_color_link", "value", allow_duplicate=True),
        Output("gc_color_shell", "value", allow_duplicate=True),
        Output("gc_color_plane", "value", allow_duplicate=True),
        Output("gc_color_brick", "value", allow_duplicate=True),
        Output("gc_color_tet", "value", allow_duplicate=True),
        Output("gc_color_joint", "value", allow_duplicate=True),
        Output("gc_color_contact", "value", allow_duplicate=True),
        Output("gc_color_pfem", "value", allow_duplicate=True),
        Output("gc_color_constraint", "value", allow_duplicate=True),
        Output("gc_color_bc", "value", allow_duplicate=True),
        Output("toolbar_message", "children", allow_duplicate=True),
        Input("gc_reset_btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_global_controls(n_clicks: int):
        """Reset only global controls to their default values."""
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
            GLOBAL_DEFAULTS["cmap"],
            GLOBAL_DEFAULTS["cmap_model"],
            GLOBAL_DEFAULTS["color_point"],
            GLOBAL_DEFAULTS["color_frame"],
            GLOBAL_DEFAULTS["color_truss"],
            GLOBAL_DEFAULTS["color_link"],
            GLOBAL_DEFAULTS["color_shell"],
            GLOBAL_DEFAULTS["color_plane"],
            GLOBAL_DEFAULTS["color_brick"],
            GLOBAL_DEFAULTS["color_tet"],
            GLOBAL_DEFAULTS["color_joint"],
            GLOBAL_DEFAULTS["color_contact"],
            GLOBAL_DEFAULTS["color_pfem"],
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
        """Trigger a figure refresh without changing current controls."""
        return {"refresh": n_clicks}, "Figure refreshed."

    @app.callback(
        Output("toolbar_message", "children", allow_duplicate=True),
        Input("toolbar_export_html_btn", "n_clicks"),
        State("toolbar_export_html_path", "value"),
        State("model_graph", "figure"),
        prevent_initial_call=True,
    )
    def _export_html(n_clicks: int, export_path: str, fig_dict: dict) -> str:
        """Export the current figure to an HTML file."""
        path = (export_path or "").strip()
        if not path:
            return "Please provide a valid HTML output path."

        if not path.lower().endswith(".html"):
            path += ".html"

        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

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
        """Trigger front-end page close and attempt backend shutdown."""
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

    @app.callback(
        Output("model_graph", "figure"),
        Input("style", "value"),
        Input("color", "value"),
        Input("show_ele_hover", "value"),
        Input("show_outline", "value"),
        Input("show_node_numbering", "value"),
        Input("show_ele_numbering", "value"),
        Input("show_bc", "value"),
        Input("bc_scale", "value"),
        Input("show_link", "value"),
        Input("show_mp_constraint", "value"),
        Input("show_constraint_dofs", "value"),
        Input("show_nodal_loads", "value"),
        Input("show_ele_loads", "value"),
        Input("load_scale", "value"),
        Input("show_local_axes", "value"),
        Input("local_axes_scale", "value"),
        Input("gc_point_size", "value"),
        Input("gc_line_width", "value"),
        Input("gc_scale_factor", "value"),
        Input("gc_show_mesh_edges", "value"),
        Input("gc_mesh_edge_color", "value"),
        Input("gc_mesh_edge_width", "value"),
        Input("gc_mesh_opacity", "value"),
        Input("gc_theme", "value"),
        Input("gc_font_family", "value"),
        Input("gc_font_size", "value"),
        Input("gc_title_font_size", "value"),
        Input("gc_cmap", "value"),
        Input("gc_cmap_model", "value"),
        Input("gc_color_point", "value"),
        Input("gc_color_frame", "value"),
        Input("gc_color_truss", "value"),
        Input("gc_color_link", "value"),
        Input("gc_color_shell", "value"),
        Input("gc_color_plane", "value"),
        Input("gc_color_brick", "value"),
        Input("gc_color_tet", "value"),
        Input("gc_color_joint", "value"),
        Input("gc_color_contact", "value"),
        Input("gc_color_pfem", "value"),
        Input("gc_color_constraint", "value"),
        Input("gc_color_bc", "value"),
        Input("refresh_store", "data"),
    )
    def _update_figure(
        style,
        color,
        show_ele_hover,
        show_outline,
        show_node_numbering,
        show_ele_numbering,
        show_bc,
        bc_scale,
        show_link,
        show_mp_constraint,
        show_constraint_dofs,
        show_nodal_loads,
        show_ele_loads,
        load_scale,
        show_local_axes,
        local_axes_scale,
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
        gc_cmap_model,
        gc_color_point,
        gc_color_frame,
        gc_color_truss,
        gc_color_link,
        gc_color_shell,
        gc_color_plane,
        gc_color_brick,
        gc_color_tet,
        gc_color_joint,
        gc_color_contact,
        gc_color_pfem,
        gc_color_constraint,
        gc_color_bc,
        refresh_data,
    ) -> go.Figure:
        """Update the figure whenever local or global controls change."""
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
            cmap=_clean_optional_text(gc_cmap),
            cmap_model=_clean_optional_text(gc_cmap_model),
        )

        set_plot_colors(
            point=gc_color_point or GLOBAL_DEFAULTS["color_point"],
            frame=gc_color_frame or GLOBAL_DEFAULTS["color_frame"],
            truss=gc_color_truss or GLOBAL_DEFAULTS["color_truss"],
            link=gc_color_link or GLOBAL_DEFAULTS["color_link"],
            shell=gc_color_shell or GLOBAL_DEFAULTS["color_shell"],
            plane=gc_color_plane or GLOBAL_DEFAULTS["color_plane"],
            brick=gc_color_brick or GLOBAL_DEFAULTS["color_brick"],
            tet=gc_color_tet or GLOBAL_DEFAULTS["color_tet"],
            joint=gc_color_joint or GLOBAL_DEFAULTS["color_joint"],
            contact=gc_color_contact or GLOBAL_DEFAULTS["color_contact"],
            pfem=gc_color_pfem or GLOBAL_DEFAULTS["color_pfem"],
            constraint=gc_color_constraint or GLOBAL_DEFAULTS["color_constraint"],
            bc=gc_color_bc or GLOBAL_DEFAULTS["color_bc"],
            cmap=_clean_optional_text(gc_cmap),
            cmap_model=_clean_optional_text(gc_cmap_model),
        )

        fig = plot_model(
            odb_tag=odb_tag,
            show_node_numbering=_is_on(show_node_numbering),
            show_ele_numbering=_is_on(show_ele_numbering),
            show_ele_hover=_is_on(show_ele_hover),
            style=style or LOCAL_DEFAULTS["style"],
            color=_clean_optional_text(color),
            show_bc=_is_on(show_bc),
            bc_scale=float(bc_scale) if bc_scale is not None else LOCAL_DEFAULTS["bc_scale"],
            show_link=_is_on(show_link),
            show_mp_constraint=_is_on(show_mp_constraint),
            show_constraint_dofs=_is_on(show_constraint_dofs),
            show_nodal_loads=_is_on(show_nodal_loads),
            show_ele_loads=_is_on(show_ele_loads),
            load_scale=float(load_scale) if load_scale is not None else LOCAL_DEFAULTS["load_scale"],
            show_local_axes=_is_on(show_local_axes),
            local_axes_scale=float(local_axes_scale)
            if local_axes_scale is not None
            else LOCAL_DEFAULTS["local_axes_scale"],
            show_outline=_is_on(show_outline),
        )
        return fig

    if auto_open:
        url = f"http://{host}:{port}"
        threading.Timer(1.0, _open_browser, args=(url,)).start()

    app.run(host=host, port=port, debug=debug)
