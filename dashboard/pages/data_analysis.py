"""Data Analysis page — interactive EDA for NAB time series."""

from __future__ import annotations

import io

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from datasets import (
    discover_categories,
    discover_datasets,
    get_default_category,
    get_default_dataset,
    load_dataset,
    load_labels,
)

dash.register_page(__name__, path="/analysis", name="Data Analysis", order=1)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CAT = get_default_category()
_DEFAULT_DS = get_default_dataset(_DEFAULT_CAT)

# ---------------------------------------------------------------------------
# Chart theme helpers
# ---------------------------------------------------------------------------

_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#111111",
    plot_bgcolor="#111111",
    margin={"l": 50, "r": 20, "t": 40, "b": 40},
    xaxis={"gridcolor": "#333"},
    yaxis={"gridcolor": "#333"},
)


def _empty_fig(msg: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_DARK_LAYOUT,
        title=msg,
    )
    return fig


# ---------------------------------------------------------------------------
# Reusable layout components
# ---------------------------------------------------------------------------


def _dataset_selector() -> dbc.Container:
    """Category + dataset dropdowns."""
    categories = discover_categories()
    default_datasets = discover_datasets(_DEFAULT_CAT) if _DEFAULT_CAT else []

    return dbc.Container(
        [
            html.H3("Dataset Selection", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Category", className="small text-muted-light"),
                            dcc.Dropdown(
                                id="ds-category",
                                options=categories,
                                value=_DEFAULT_CAT,
                                clearable=False,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("Dataset", className="small text-muted-light"),
                            dcc.Dropdown(
                                id="ds-dataset",
                                options=default_datasets,
                                value=_DEFAULT_DS,
                                clearable=False,
                            ),
                        ],
                        md=5,
                    ),
                    dbc.Col(
                        [
                            html.Label("Show Labels", className="small text-muted-light"),
                            dbc.Switch(
                                id="ds-show-labels",
                                value=True,
                                label="Anomaly ground truth",
                                className="mt-1",
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="g-3",
            ),
        ],
        fluid=True,
        className="card-1 py-3 shadow rounded-3",
    )


_STAT_TOOLTIPS: dict[str, str] = {
    "stat-overview": ("Dataset size, time span, number of labeled anomalies, and missing values."),
    "stat-central": (
        "Mean: arithmetic average. "
        "Median: middle value (robust to outliers). "
        "Mode: most frequent value."
    ),
    "stat-spread": (
        "Std: standard deviation (average distance from the mean). "
        "Min/Max: value range. "
        "IQR: interquartile range (Q3\u2013Q1), measures spread of the middle 50%."
    ),
    "stat-shape": (
        "Skewness: asymmetry (\u22480 = symmetric, >0 = right tail, <0 = left tail). "
        "Kurtosis: tail heaviness (\u22480 = normal, >0 = heavy tails). "
        "Q1/Q3: 25th and 75th percentiles."
    ),
}


def _stat_card(card_id: str, title: str, icon: str) -> dbc.Col:
    tooltip_text = _STAT_TOOLTIPS.get(card_id, "")
    card = dbc.Card(
        dbc.CardBody(
            [
                html.H6(
                    [html.I(className=f"bi bi-{icon} me-2"), title],
                    className="card-title mb-2",
                    id=f"{card_id}-title",
                ),
                html.Div(id=card_id, className="small"),
            ]
        ),
        className="card-dark shadow rounded-3 h-100",
    )
    tooltip = dbc.Tooltip(
        tooltip_text,
        target=f"{card_id}-title",
        placement="top",
        className="stat-tooltip",
    )
    return dbc.Col([card, tooltip], md=3)


def _statistics_section() -> dbc.Container:
    return dbc.Container(
        [
            html.H3("Descriptive Statistics", className="mb-3"),
            dbc.Row(
                [
                    _stat_card("stat-overview", "Overview", "info-circle"),
                    _stat_card("stat-central", "Central Tendency", "bullseye"),
                    _stat_card("stat-spread", "Spread", "arrows-expand"),
                    _stat_card("stat-shape", "Shape", "distribute-vertical"),
                ],
                className="g-3",
            ),
        ],
        fluid=True,
        className="card-1 py-3 mt-4 shadow rounded-3",
    )


def _timeseries_section() -> dbc.Container:
    return dbc.Container(
        [
            html.H3("Time Series", className="mb-3"),
            dcc.Loading(
                dcc.Graph(
                    id="ts-graph",
                    config={"displayModeBar": "hover", "scrollZoom": True},
                    style={"height": "420px"},
                ),
                type="default",
                color="#636efa",
            ),
        ],
        fluid=True,
        className="card-1 py-3 mt-4 shadow rounded-3",
    )


def _distribution_section() -> dbc.Container:
    return dbc.Container(
        [
            html.H3("Distribution", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            dcc.Graph(
                                id="hist-graph",
                                config={"displayModeBar": "hover"},
                                style={"height": "350px"},
                            ),
                            type="default",
                            color="#636efa",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dcc.Loading(
                            dcc.Graph(
                                id="box-graph",
                                config={"displayModeBar": "hover"},
                                style={"height": "350px"},
                            ),
                            type="default",
                            color="#636efa",
                        ),
                        md=6,
                    ),
                ],
                className="g-3",
            ),
        ],
        fluid=True,
        className="card-1 py-3 mt-4 shadow rounded-3",
    )


def _rolling_section() -> dbc.Container:
    return dbc.Container(
        [
            html.H3("Rolling Statistics", className="mb-3"),
            dcc.Tabs(
                id="rolling-tabs",
                value="24",
                children=[
                    dcc.Tab(
                        label=label,
                        value=val,
                        className="diagramm-tab-left",
                        selected_className="selected-diagramm-tab",
                    )
                    for val, label in [
                        ("6", "6h"),
                        ("12", "12h"),
                        ("24", "24h"),
                        ("48", "48h"),
                        ("168", "7d"),
                    ]
                ],
            ),
            dcc.Loading(
                dcc.Graph(
                    id="rolling-graph",
                    config={"displayModeBar": "hover", "scrollZoom": True},
                    style={"height": "380px"},
                ),
                type="default",
                color="#636efa",
            ),
        ],
        fluid=True,
        className="card-1 py-3 mt-4 shadow rounded-3",
    )


def _acf_section() -> dbc.Container:
    return dbc.Container(
        [
            html.H3("Autocorrelation", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            dcc.Graph(
                                id="acf-graph",
                                config={"displayModeBar": "hover"},
                                style={"height": "320px"},
                            ),
                            type="default",
                            color="#636efa",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dcc.Loading(
                            dcc.Graph(
                                id="pacf-graph",
                                config={"displayModeBar": "hover"},
                                style={"height": "320px"},
                            ),
                            type="default",
                            color="#636efa",
                        ),
                        md=6,
                    ),
                ],
                className="g-3",
            ),
        ],
        fluid=True,
        className="card-1 py-3 mt-4 mb-4 shadow rounded-3",
    )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

layout = html.Div(
    [
        # Invisible store: holds the loaded dataframe as JSON
        dcc.Store(id="ds-store", storage_type="memory"),
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        _dataset_selector(),
                        _statistics_section(),
                        _timeseries_section(),
                        _distribution_section(),
                        _rolling_section(),
                        _acf_section(),
                    ],
                ),
                className="page-div pt-3",
            )
        ),
    ]
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("ds-dataset", "options"),
    Output("ds-dataset", "value"),
    Input("ds-category", "value"),
)
def update_dataset_dropdown(category: str) -> tuple[list[dict], str | None]:
    """Populate the dataset dropdown when category changes."""
    datasets = discover_datasets(category) if category else []
    first = datasets[0]["value"] if datasets else None
    return datasets, first


@callback(
    Output("ds-store", "data"),
    Input("ds-dataset", "value"),
    State("ds-category", "value"),
)
def load_data(rel_path: str | None, category: str | None) -> dict | None:
    """Load the selected dataset into the store."""
    if not rel_path:
        return None
    df = load_dataset(rel_path)
    if df is None:
        return None
    # Extract filename from rel_path for label lookup
    filename = rel_path.split("/")[-1]
    labels = load_labels(category, filename) if category else []
    return {
        "df": df.to_json(date_format="iso", orient="split"),
        "name": filename,
        "category": category,
        "labels": labels,
    }


@callback(
    Output("stat-overview", "children"),
    Output("stat-central", "children"),
    Output("stat-spread", "children"),
    Output("stat-shape", "children"),
    Input("ds-store", "data"),
)
def update_statistics(store: dict | None) -> tuple:
    """Compute and display descriptive statistics."""
    if not store:
        empty = html.Span("—", className="text-muted-light")
        return empty, empty, empty, empty

    df = pd.read_json(io.StringIO(store["df"]), orient="split")
    v = df["value"]

    def _row(label: str, val: str) -> html.Div:
        return html.Div(
            [
                html.Span(f"{label}: ", className="text-muted-light"),
                html.Span(val),
            ],
            className="mb-1",
        )

    n_labels = len(store.get("labels", []))
    ts = pd.to_datetime(df["timestamp"])
    duration = ts.max() - ts.min()

    overview = html.Div(
        [
            _row("Points", f"{len(df):,}"),
            _row("Time span", str(duration)),
            _row("Labels", f"{n_labels} anomaly points" if n_labels else "None"),
            _row("Missing", f"{v.isna().sum()}"),
        ]
    )

    central = html.Div(
        [
            _row("Mean", f"{v.mean():.4f}"),
            _row("Median", f"{v.median():.4f}"),
            _row("Mode", f"{v.mode().iloc[0]:.4f}" if len(v.mode()) > 0 else "N/A"),
        ]
    )

    spread = html.Div(
        [
            _row("Std", f"{v.std():.4f}"),
            _row("Min", f"{v.min():.4f}"),
            _row("Max", f"{v.max():.4f}"),
            _row("IQR", f"{v.quantile(0.75) - v.quantile(0.25):.4f}"),
        ]
    )

    skew_val = v.skew()
    kurt_val = v.kurtosis()
    shape = html.Div(
        [
            _row("Skewness", f"{skew_val:.4f}"),
            _row("Kurtosis", f"{kurt_val:.4f}"),
            _row("Q1", f"{v.quantile(0.25):.4f}"),
            _row("Q3", f"{v.quantile(0.75):.4f}"),
        ]
    )

    return overview, central, spread, shape


@callback(
    Output("ts-graph", "figure"),
    Input("ds-store", "data"),
    Input("ds-show-labels", "value"),
)
def update_timeseries(store: dict | None, show_labels: bool) -> go.Figure:
    """Render the time series with optional anomaly labels."""
    if not store:
        return _empty_fig("Select a dataset")

    df = pd.read_json(io.StringIO(store["df"]), orient="split")
    fig = px.line(
        df,
        x="timestamp",
        y="value",
        labels={"timestamp": "", "value": "Value"},
    )
    fig.update_traces(line_color="#636efa")

    # Overlay anomaly labels
    if show_labels and store.get("labels"):
        label_times = pd.to_datetime(store["labels"])
        ts = pd.to_datetime(df["timestamp"])
        mask = ts.isin(label_times)
        if mask.any():
            anom_df = df[mask.values]
            fig.add_trace(
                go.Scatter(
                    x=anom_df["timestamp"],
                    y=anom_df["value"],
                    mode="markers",
                    marker={"color": "#ff5252", "size": 6, "symbol": "x"},
                    name="Anomaly (ground truth)",
                )
            )

    fig.update_layout(**_DARK_LAYOUT, showlegend=True)
    fig.update_layout(
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font_color="#ffffff",
        )
    )
    return fig


@callback(
    Output("hist-graph", "figure"),
    Output("box-graph", "figure"),
    Input("ds-store", "data"),
)
def update_distribution(store: dict | None) -> tuple[go.Figure, go.Figure]:
    """Render histogram and box plot."""
    if not store:
        return _empty_fig("No data"), _empty_fig("No data")

    df = pd.read_json(io.StringIO(store["df"]), orient="split")

    # Histogram
    hist_fig = px.histogram(
        df,
        x="value",
        nbins=80,
        labels={"value": "Value", "count": "Count"},
        title="Value Distribution",
    )
    hist_fig.update_traces(marker_color="#636efa")
    hist_fig.update_layout(**_DARK_LAYOUT)

    # Box plot
    box_fig = go.Figure(
        go.Box(
            y=df["value"],
            marker_color="#636efa",
            boxmean="sd",
            name="Value",
        )
    )
    box_fig.update_layout(**_DARK_LAYOUT, title="Box Plot")

    return hist_fig, box_fig


@callback(
    Output("rolling-graph", "figure"),
    Input("ds-store", "data"),
    Input("rolling-tabs", "value"),
)
def update_rolling(store: dict | None, window_str: str) -> go.Figure:
    """Render rolling mean and std."""
    if not store:
        return _empty_fig("No data")

    df = pd.read_json(io.StringIO(store["df"]), orient="split")
    window = int(window_str)

    rolling_mean = df["value"].rolling(window=window, min_periods=1).mean()
    rolling_std = df["value"].rolling(window=window, min_periods=1).std()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines",
            name="Original",
            line={"color": "rgba(99,110,250,0.3)", "width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=rolling_mean,
            mode="lines",
            name=f"Rolling Mean ({window})",
            line={"color": "#00e676", "width": 2},
        )
    )
    # Confidence band: mean ± 2*std
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    fig.add_trace(
        go.Scatter(
            x=pd.concat([df["timestamp"], df["timestamp"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(0,230,118,0.1)",
            line={"color": "rgba(0,0,0,0)"},
            name="\u00b12\u03c3 Band",
            showlegend=True,
        )
    )

    fig.update_layout(
        **_DARK_LAYOUT,
        title=f"Rolling Statistics (window={window})",
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#ffffff"),
    )
    return fig


@callback(
    Output("acf-graph", "figure"),
    Output("pacf-graph", "figure"),
    Input("ds-store", "data"),
)
def update_acf(store: dict | None) -> tuple[go.Figure, go.Figure]:
    """Render ACF and PACF bar charts."""
    if not store:
        return _empty_fig("No data"), _empty_fig("No data")

    df = pd.read_json(io.StringIO(store["df"]), orient="split")
    values = df["value"].dropna().values

    n_lags = min(60, len(values) // 4)
    if n_lags < 2:
        return _empty_fig("Not enough data"), _empty_fig("Not enough data")

    # ACF
    from statsmodels.tsa.stattools import acf, pacf

    acf_values = acf(values, nlags=n_lags, fft=True)
    ci_bound = 1.96 / np.sqrt(len(values))
    lags = list(range(len(acf_values)))

    acf_fig = go.Figure()
    acf_fig.add_trace(go.Bar(x=lags, y=acf_values, marker_color="#636efa", name="ACF"))
    acf_fig.add_hline(y=ci_bound, line_dash="dash", line_color="#ff5252", opacity=0.6)
    acf_fig.add_hline(y=-ci_bound, line_dash="dash", line_color="#ff5252", opacity=0.6)
    acf_fig.update_layout(
        **_DARK_LAYOUT,
        title="Autocorrelation (ACF)",
        xaxis_title="Lag",
        yaxis_title="Correlation",
    )

    # PACF
    try:
        pacf_values = pacf(values, nlags=n_lags)
    except Exception:
        return acf_fig, _empty_fig("PACF computation failed")

    pacf_fig = go.Figure()
    pacf_fig.add_trace(go.Bar(x=lags, y=pacf_values, marker_color="#636efa", name="PACF"))
    pacf_fig.add_hline(y=ci_bound, line_dash="dash", line_color="#ff5252", opacity=0.6)
    pacf_fig.add_hline(y=-ci_bound, line_dash="dash", line_color="#ff5252", opacity=0.6)
    pacf_fig.update_layout(
        **_DARK_LAYOUT,
        title="Partial Autocorrelation (PACF)",
        xaxis_title="Lag",
        yaxis_title="Correlation",
    )

    return acf_fig, pacf_fig
