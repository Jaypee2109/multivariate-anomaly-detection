"""Home / landing page for the Anomaly Detection Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

# Ensure the src package is importable when running from dashboard/
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from datasets import (
    add_anomaly_zones,
    discover_smd_models,
    discover_smd_results,
    is_feature_column,
    list_smd_machines,
    load_smd_results,
    load_smd_train_test,
)  # noqa: E402

dash.register_page(__name__, path="/", name="Home", order=0)

# ---------------------------------------------------------------------------
# Chart theme
# ---------------------------------------------------------------------------

_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#111111",
    plot_bgcolor="#111111",
    margin={"l": 40, "r": 20, "t": 30, "b": 40},
    xaxis={"gridcolor": "#333"},
    yaxis={"gridcolor": "#333"},
)

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
                    [html.I(className="bi bi-cpu me-2"), "Trained Models"],
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
# Dataset Overview: show first few available SMD machines as tabs
# ---------------------------------------------------------------------------

_SMD_MACHINES = list_smd_machines()
_SMD_TABS = _SMD_MACHINES[:5] if _SMD_MACHINES else []
_DEFAULT_TAB = _SMD_TABS[0]["value"] if _SMD_TABS else ""


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
                            "A comparative study of classical and deep learning "
                            "anomaly detectors on multivariate time series from "
                            "the Server Machine Dataset (SMD).",
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
                                                    html.I(
                                                        className="bi bi-bullseye me-2"
                                                    ),
                                                    "Goal",
                                                ]
                                            ),
                                            html.P(
                                                "Detect anomalous intervals in multivariate "
                                                "server metrics using VAR residuals, "
                                                "Isolation Forest, LSTM Autoencoder, and "
                                                "LSTM Forecaster.",
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
                                                    html.I(
                                                        className="bi bi-database me-2"
                                                    ),
                                                    "Datasets",
                                                ]
                                            ),
                                            html.P(
                                                "SMD (Server Machine Dataset) — "
                                                "28 machines with 38 sensor features each.",
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
                                                    html.I(
                                                        className="bi bi-bar-chart-line me-2"
                                                    ),
                                                    "Metrics",
                                                ]
                                            ),
                                            html.P(
                                                "F1, Precision, Recall, AUROC — "
                                                "evaluated point-wise on the test split.",
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
                            value=_DEFAULT_TAB,
                            children=(
                                [
                                    dcc.Tab(
                                        label=t["label"],
                                        value=t["value"],
                                        className="diagramm-tab-left",
                                        selected_className="selected-diagramm-tab",
                                    )
                                    for t in _SMD_TABS
                                ]
                                if _SMD_TABS
                                else [
                                    dcc.Tab(
                                        label="No data",
                                        value="",
                                        className="diagramm-tab-left",
                                        selected_className="selected-diagramm-tab",
                                    )
                                ]
                            ),
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
    # --- API status (inference server) ---
    from api_client import get_client

    client = get_client()
    health = client.get_health()

    if health is None:
        api_content = html.Span(
            [html.I(className="bi bi-x-circle me-2"), "Offline"],
            className="status-offline",
        )
    elif health.get("status") == "healthy":
        api_content = html.Span(
            [html.I(className="bi bi-check-circle me-2"), "Online"],
            className="status-online",
        )
    else:
        api_content = html.Span(
            [html.I(className="bi bi-exclamation-triangle me-2"), health.get("status", "unknown")],
            className="status-offline",
        )

    # --- Trained models (from artifact CSVs) ---
    results = discover_smd_results()
    if not results:
        models_content = html.Span(
            [html.I(className="bi bi-question-circle me-1"), "No results"],
            title="Train with: python -m time_series_transformer train-mv --machine machine-1-1",
            className="text-muted-light small",
            style={"cursor": "help"},
        )
        return api_content, models_content

    # Discover models from first available artifact
    df = load_smd_results(results[0]["value"])
    model_slugs = discover_smd_models(df) if df is not None else []

    if model_slugs:
        model_badges = [
            dbc.Badge(slug, color="primary", className="me-1 mb-1")
            for slug in model_slugs
        ]
        models_content = html.Div([
            html.Div(model_badges),
            html.Small(
                f"{len(results)} machine(s) trained",
                className="text-muted-light",
            ),
        ])
    else:
        models_content = html.Span(
            [html.I(className="bi bi-question-circle me-1"), "No models found"],
            title="Train with: python -m time_series_transformer train-mv --machine machine-1-1",
            className="text-muted-light small",
            style={"cursor": "help"},
        )

    return api_content, models_content


@callback(
    Output("dataset-graph", "figure"),
    Input("dataset-tabs", "value"),
)
def update_dataset_graph(machine_id: str) -> go.Figure:
    """Show a few features from the selected SMD machine."""
    if not machine_id:
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT, title="No data available.")
        return fig

    result = load_smd_train_test(machine_id)
    if result is None:
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT, title=f"No data for {machine_id}")
        return fig

    train_df, test_df, test_labels = result
    df = pd.concat([train_df, test_df], ignore_index=True)
    features = [c for c in df.columns if is_feature_column(c)]
    show_features = features[:4]

    fig = go.Figure()
    colors = ["#636efa", "#00e676", "#ff5252", "#ffa726"]
    for i, feat in enumerate(show_features):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=df[feat],
                mode="lines",
                name=feat,
                line={"color": colors[i % len(colors)], "width": 1},
            )
        )

    # Show ground-truth anomaly zones (test split only)
    if test_labels is not None:
        # Build full-length mask: False for train, actual labels for test
        full_mask = pd.Series(False, index=range(len(df)))
        full_mask.iloc[len(train_df) :] = test_labels.values.astype(bool)
        add_anomaly_zones(fig, full_mask)

    fig.update_layout(
        **_DARK_LAYOUT,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#ffffff"),
        xaxis_title="Timestep",
    )
    return fig
