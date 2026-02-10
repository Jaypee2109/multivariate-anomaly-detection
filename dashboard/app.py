"""Anomaly Detection Dashboard — main Dash application."""

from __future__ import annotations

import argparse

import dash
import dash_bootstrap_components as dbc
from dash import html


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    use_pages=True,
    suppress_callback_exceptions=True,
)
app.title = "Anomaly Detection Dashboard"


def _navbar() -> dbc.Navbar:
    """Build the top navbar dynamically from registered pages."""
    pages = list(dash.page_registry.values())
    nav_links = [
        dbc.NavLink(
            page["name"],
            href=page["path"],
            active="exact",
            className="nav-page-link",
        )
        for page in pages
    ]
    return dbc.Navbar(
        html.Div(
            [
                dbc.NavbarBrand(
                    [html.I(className="bi bi-activity me-2"), "Anomaly Detection"],
                    class_name="ms-left px-3",
                ),
                dbc.Nav(nav_links, pills=True, className="ms-auto d-flex px-3"),
            ],
            className="d-flex w-75 align-items-center mx-auto",
        ),
        color="#000000",
        dark=True,
        sticky="top",
        className="navbar-with-border",
    )


app.layout = dbc.Container(
    [_navbar(), html.Div(dash.page_container)],
    className="m-0 p-0",
    fluid=True,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detection Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
