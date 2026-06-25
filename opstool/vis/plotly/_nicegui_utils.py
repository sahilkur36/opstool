from __future__ import annotations

import os
import re
from typing import Any

import plotly.graph_objects as go

try:
    from nicegui import app, ui
except ImportError as e:  # pragma: no cover - exercised only without optional dependency
    msg = "NiceGUI is required for Plotly GUI viewers. Install it with: pip install nicegui"
    raise ImportError(msg) from e


SIDEBAR_WIDTH = "355px"
FONT_FAMILIES = [
    "Arial",
    "Arial, sans-serif",
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

CSS_COLOR_FALLBACKS = {
    "black": "#000000",
    "white": "#ffffff",
    "red": "#ff0000",
    "green": "#008000",
    "blue": "#0000ff",
    "gray": "#808080",
    "grey": "#808080",
}

INDEXED_RELAYOUT_KEY = re.compile(r"^(?P<name>.+)\[(?P<index>\d+)\]$")


def clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def export_html(fig: go.Figure, export_path: str | None) -> str:
    path = (export_path or "").strip()
    if not path:
        return "Please provide a valid HTML output path."
    if not path.lower().endswith(".html"):
        path += ".html"
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return f"Exported HTML to: {path}"


def run_gui(title: str, host: str, port: int, debug: bool, auto_open: bool, root=None) -> None:
    # Plotting may create/update .opstool.output files; hot reload would see
    # those writes and restart the viewer in a loop.
    # Browser auto-opening is also disabled by default to avoid spawning a new
    # tab if an external runner restarts the script.
    ui.run(root, host=host, port=port, title=title, reload=False, show=auto_open)


def shutdown_gui() -> None:
    ui.run_javascript("window.open('', '_self'); window.close();")
    app.shutdown()


def set_message(label, text: str) -> None:
    label.text = text
    label.update()


def update_plot(plot, fig: go.Figure) -> None:
    plot.figure = fig
    plot.update()


def preserve_plotly_view(fig: go.Figure, view_state: dict[str, dict] | None = None, uirevision: str = "opstool-gui") -> None:
    fig.update_layout(uirevision=uirevision)
    for layout_name, layout_value in (view_state or {}).items():
        if layout_value:
            fig.update_layout(**{layout_name: layout_value})


def _set_nested_relayout_value(target: dict, property_path: str, value: Any) -> None:
    parts = property_path.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    leaf = parts[-1]
    match = INDEXED_RELAYOUT_KEY.match(leaf)
    if not match:
        target[leaf] = value
        return
    name = match.group("name")
    index = int(match.group("index"))
    values = target.setdefault(name, [])
    while len(values) <= index:
        values.append(None)
    values[index] = value


def update_camera_state(view_state: dict[str, dict], payload) -> None:
    if isinstance(payload, (list, tuple)) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        if key in {"autosize", "width", "height"}:
            continue
        if "." not in key:
            if isinstance(value, dict):
                view_state[key] = value
            continue
        layout_name, property_path = key.split(".", 1)
        if not (
            layout_name.startswith("scene")
            or layout_name.startswith("xaxis")
            or layout_name.startswith("yaxis")
            or layout_name.startswith("map")
            or layout_name.startswith("polar")
        ):
            continue
        target = view_state.setdefault(layout_name, {})
        _set_nested_relayout_value(target, property_path, value)


def page_shell(title: str):
    ui.query("body").style("margin: 0; overflow: hidden;")
    with ui.row().classes("w-full h-screen no-wrap").style("gap: 0;"):
        sidebar = ui.column().style(
            f"width: {SIDEBAR_WIDTH}; min-width: {SIDEBAR_WIDTH}; max-width: {SIDEBAR_WIDTH}; "
            "height: 100vh; overflow-y: auto; padding: 10px; border-right: 1px solid #ddd; "
            "box-sizing: border-box; background: #fafafa; gap: 6px;"
        )
        content = ui.column().classes("h-screen").style("flex: 1; min-width: 0; overflow: hidden; gap: 0;")
    with sidebar:
        ui.label(title).style("font-size: 17px; font-weight: 700; margin: 0;")
    return sidebar, content


def section_title(text: str) -> None:
    ui.label(text).style(
        "font-weight: 700; font-size: 14px; margin-top: 10px; margin-bottom: 2px; "
        "padding-bottom: 3px; border-bottom: 1px solid #d9d9d9; width: 100%;"
    )


def subsection_title(text: str) -> None:
    ui.label(text).style("font-weight: 600; font-size: 12px; margin-top: 6px; color: #444;")


def control_row():
    return ui.row().classes("w-full no-wrap").style("gap: 8px; align-items: end;")


def bind_auto_refresh(controls: dict[str, Any], callback, exclude: set[str] | None = None) -> None:
    excluded = exclude or set()
    for name, control in controls.items():
        if name not in excluded and hasattr(control, "on_value_change"):
            control.on_value_change(lambda _: callback())


def compact_select(label: str, options, value=None, clearable: bool = False):
    option_values = list(options)
    if value not in (None, "") and value not in option_values:
        option_values.insert(0, value)
    return (
        ui.select(options=option_values, value=value, label=label, with_input=True, clearable=clearable)
        .props("dense outlined options-dense")
        .style("flex: 1; min-width: 0;")
    )


def compact_input(label: str, value: Any = "", placeholder: str = ""):
    return (
        ui.input(label=label, value=value, placeholder=placeholder)
        .props("dense outlined debounce=500")
        .style("flex: 1; min-width: 0;")
    )


def compact_number(label: str, value: Any = None, step: float = 1):
    return (
        ui.number(label=label, value=value, step=step)
        .props("dense outlined debounce=500")
        .style("flex: 1; min-width: 0;")
    )


def normalize_color_value(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("#") and len(text) in {4, 7, 9}:
        return text
    return CSS_COLOR_FALLBACKS.get(text.lower(), "#000000")


def compact_color(label: str, value: Any = "#000000"):
    color_value = normalize_color_value(value)
    if hasattr(ui, "color_input"):
        return (
            ui.color_input(label=label, value=color_value, preview=True)
            .props("dense outlined")
            .style("flex: 1; min-width: 0;")
        )
    return (
        ui.input(label=label, value=color_value)
        .props("dense outlined type=color debounce=500")
        .style("flex: 1; min-width: 0;")
    )


def compact_checkbox(label: str, value: bool = False):
    return ui.checkbox(label, value=value).props("dense").style("flex: 1; min-width: 0;")


def compact_button(text: str, on_click=None):
    return ui.button(text, on_click=on_click).props("dense outline no-caps").style("flex: 1; min-width: 0;")
