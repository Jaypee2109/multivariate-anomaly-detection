"""Home / landing page for the Anomaly Detection Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html

# Ensure the src package is importable when running from dashboard/
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from time_series_transformer.config import RAW_DATA_DIR  # noqa: E402

dash.register_page(__name__, path="/", name="Home", order=0)

# ---------------------------------------------------------------------------
# Available NAB datasets for the overview tab selector
# ---------------------------------------------------------------------------

NAB_DATASETS: dict[str, tuple[str, Path]] = {
    "nyc_taxi": (
        "NYC Taxi",
        RAW_DATA_DIR / "nab" / "realKnownCause" / "realKnownCause" / "nyc_taxi.csv",
    ),
    "machine_temp": (
        "Machine Temperature",
        RAW_DATA_DIR
        / "nab"
        / "realKnownCause"
        / "realKnownCause"
        / "machine_temperature_system_failure.csv",
    ),
    "ambient_temp": (
        "Ambient Temperature",
        RAW_DATA_DIR
        / "nab"
        / "realKnownCause"
        / "realKnownCause"
        / "ambient_temperature_system_failure.csv",
    ),
    "ec2_cpu": (
        "EC2 CPU Utilization",
        RAW_DATA_DIR
        / "nab"
        / "realAWSCloudwatch"
        / "realAWSCloudwatch"
        / "ec2_cpu_utilization_24ae8d.csv",
    ),
    "twitter_aapl": (
        "Twitter Volume (AAPL)",
        RAW_DATA_DIR / "nab" / "realTweets" / "realTweets" / "Twitter_volume_AAPL.csv",
    ),
}

# ---------------------------------------------------------------------------
# Helper: API status card (checks inference server live)
# ---------------------------------------------------------------------------


def _api_status_card() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(
                    [html.I(className="bi bi-hdd-network me-2"), "Inference API"],
                    className="card-title",
                ),
                dcc.Loading(
                    html.Div(id="api-status-content"),
                    type="dot",
                    color="#636efa",
                ),
            ]
        ),
        className="card-dark shadow rounded-3 h-100",
    )


def _models_card() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(
                    [html.I(className="bi bi-cpu me-2"), "Loaded Models"],
                    className="card-title",
                ),
                dcc.Loading(
                    html.Div(id="models-status-content"),
                    type="dot",
                    color="#636efa",
                ),
            ]
        ),
        className="card-dark shadow rounded-3 h-100",
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = html.Div(
    [
        # --- Section 1: Hero / About ---
        dbc.Row(
            dbc.Col(
                dbc.Container(
                    [
                        html.H1(
                            "Time Series Anomaly Detection",
                            className="display-5 mb-3",
                        ),
                        html.P(
                            "A comparative study of transformer-based and classical "
                            "anomaly detectors on univariate time series benchmarks.",
                            className="lead text-muted-light",
                        ),
                        html.Hr(style={"borderColor": "#444"}),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Container(
                                        [
                                            html.H6(
                                                [
                                                    html.I(className="bi bi-bullseye me-2"),
                                                    "Goal",
                                                ]
                                            ),
                                            html.P(
                                                "Detect anomalous points in time series "
                                                "using ARIMA residuals, Isolation Forest, "
                                                "LSTM forecast errors, and a compact "
                                                "Transformer encoder.",
                                                className="text-muted-light small",
                                            ),
                                        ],
                                        fluid=True,
                                        className="card-2 shadow rounded-3 p-3",
                                    ),
                                    md=4,
                                ),
                                dbc.Col(
                                    dbc.Container(
                                        [
                                            html.H6(
                                                [
                                                    html.I(className="bi bi-database me-2"),
                                                    "Datasets",
                                                ]
                                            ),
                                            html.P(
                                                "NAB (Numenta Anomaly Benchmark) — "
                                                "58 real-world and synthetic time series "
                                                "with labeled anomaly windows.",
                                                className="text-muted-light small",
                                            ),
                                        ],
                                        fluid=True,
                                        className="card-2 shadow rounded-3 p-3",
                                    ),
                                    md=4,
                                ),
                                dbc.Col(
                                    dbc.Container(
                                        [
                                            html.H6(
                                                [
                                                    html.I(className="bi bi-bar-chart-line me-2"),
                                                    "Metrics",
                                                ]
                                            ),
                                            html.P(
                                                "F1@point, F1@range, AUROC, AUPR — "
                                                "evaluated on a 70/30 time-ordered split.",
                                                className="text-muted-light small",
                                            ),
                                        ],
                                        fluid=True,
                                        className="card-2 shadow rounded-3 p-3",
                                    ),
                                    md=4,
                                ),
                            ],
                            className="g-3",
                        ),
                    ],
                    fluid=True,
                    className="card-1 py-3 shadow rounded-3",
                ),
                className="page-div pt-3",
            )
        ),
        # --- Section 2: API Status ---
        dbc.Row(
            dbc.Col(
                dbc.Container(
                    [
                        html.H3("System Status", className="mb-3"),
                        dbc.Row(
                            [
                                dbc.Col(_api_status_card(), md=6),
                                dbc.Col(_models_card(), md=6),
                            ],
                            className="g-3",
                        ),
                        # Hidden interval to poll API status on page load
                        dcc.Interval(
                            id="status-interval",
                            interval=10_000,
                            n_intervals=0,
                        ),
                    ],
                    fluid=True,
                    className="card-1 py-3 mt-4 shadow rounded-3",
                ),
            )
        ),
        # --- Section 3: Dataset Overview ---
        dbc.Row(
            dbc.Col(
                dbc.Container(
                    [
                        html.H3("Dataset Overview", className="mb-3"),
                        dcc.Tabs(
                            id="dataset-tabs",
                            value="nyc_taxi",
                            children=[
                                dcc.Tab(
                                    label=display_name,
                                    value=key,
                                    className="diagramm-tab-left",
                                    selected_className="selected-diagramm-tab",
                                )
                                for key, (display_name, _path) in NAB_DATASETS.items()
                            ],
                        ),
                        dcc.Loading(
                            dcc.Graph(
                                id="dataset-graph",
                                config={"displayModeBar": True},
                                style={"height": "400px"},
                            ),
                            type="default",
                            color="#636efa",
                        ),
                    ],
                    fluid=True,
                    className="card-1 py-3 mt-4 mb-4 shadow rounded-3",
                ),
            )
        ),
    ]
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("api-status-content", "children"),
    Output("models-status-content", "children"),
    Input("status-interval", "n_intervals"),
)
def update_status(_n: int) -> tuple:
    from api_client import get_client

    client = get_client()
    health = client.get_health()

    if health is None:
        api_content = html.Span(
            [html.I(className="bi bi-x-circle me-2"), "Offline"],
            className="status-offline",
        )
        models_content = html.Span(
            [html.I(className="bi bi-question-circle me-1"), "API offline"],
            title="Start with: python -m time_series_transformer serve",
            className="text-muted-light small",
            style={"cursor": "help"},
        )
        return api_content, models_content

    status = health.get("status", "unknown")
    models_loaded = health.get("models_loaded", [])

    if status == "healthy":
        api_content = html.Div(
            [
                html.Span(
                    [html.I(className="bi bi-check-circle me-2"), "Online"],
                    className="status-online",
                ),
                html.Br(),
                html.Small(
                    f"{len(models_loaded)} model(s) ready",
                    className="text-muted-light",
                ),
            ]
        )
    else:
        api_content = html.Div(
            [
                html.Span(
                    [html.I(className="bi bi-exclamation-triangle me-2"), status],
                    className="status-offline",
                ),
            ]
        )

    if models_loaded:
        model_badges = [
            dbc.Badge(name, color="primary", className="me-1 mb-1") for name in models_loaded
        ]
        models_content = html.Div(model_badges)
    else:
        models_content = html.Span(
            [html.I(className="bi bi-question-circle me-1"), "No models loaded"],
            title="Train with: python -m time_series_transformer train --save-checkpoints",
            className="text-muted-light small",
            style={"cursor": "help"},
        )

    return api_content, models_content


@callback(
    Output("dataset-graph", "figure"),
    Input("dataset-tabs", "value"),
)
def update_dataset_graph(selected: str) -> dict:
    _, csv_path = NAB_DATASETS.get(selected, list(NAB_DATASETS.values())[0])

    if not csv_path.exists():
        fig = px.line(title="Dataset not found")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111111", plot_bgcolor="#111111")
        return fig

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    fig = px.line(
        df,
        x="timestamp",
        y="value",
        template="plotly_dark",
        labels={"timestamp": "", "value": "Value"},
    )
    fig.update_layout(
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
        xaxis={"gridcolor": "#333"},
        yaxis={"gridcolor": "#333"},
    )
    return fig
