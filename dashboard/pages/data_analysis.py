"""Data Analysis page — interactive EDA for SMD multivariate time series."""

from __future__ import annotations

import io

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from datasets import add_anomaly_zones, is_feature_column, list_smd_machines, load_smd_train_test

dash.register_page(__name__, path="/analysis", name="Data Analysis", order=1)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_MACHINES = list_smd_machines()
_DEFAULT_MACHINE = _MACHINES[0]["value"] if _MACHINES else None

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
    fig.update_layout(**_DARK_LAYOUT, title=msg)
    return fig


_BLANK = go.Figure()
_BLANK.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_visible=False,
    yaxis_visible=False,
    margin=dict(l=0, r=0, t=0, b=0),
)

# ---------------------------------------------------------------------------
# Reusable layout components
# ---------------------------------------------------------------------------


def _dataset_selector() -> dbc.Container:
    """Machine + feature dropdowns."""
    return dbc.Container(
        [
            html.H3("Dataset Selection", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Machine", className="small text-muted-light"),
                            dcc.Dropdown(
                                id="ds-machine",
                                options=_MACHINES,
                                value=_DEFAULT_MACHINE,
                                clearable=False,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("Feature", className="small text-muted-light"),
                            dcc.Dropdown(
                                id="ds-feature",
                                options=[],
                                value=None,
                                clearable=False,
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            html.Label("Split", className="small text-muted-light"),
                            dcc.Dropdown(
                                id="ds-split",
                                options=[
                                    {"label": "Both", "value": "both"},
                                    {"label": "Train", "value": "train"},
                                    {"label": "Test", "value": "test"},
                                ],
                                value="both",
                                clearable=False,
                            ),
                        ],
                        md=2,
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
                        md=4,
                    ),
                ],
                className="g-3",
            ),
        ],
        fluid=True,
        className="card-1 py-3 shadow rounded-3",
    )


_STAT_TOOLTIPS: dict[str, str] = {
    "stat-overview": "Dataset size, number of features, number of labeled anomalies, and missing values.",
    "stat-central": (
        "Mean: arithmetic average. "
        "Median: middle value (robust to outliers). "
        "Mode: most frequent value."
    ),
    "stat-spread": (
        "Std: standard deviation (average distance from the mean). "
        "Min/Max: value range. "
        "IQR: interquartile range (Q3-Q1), measures spread of the middle 50%."
    ),
    "stat-shape": (
        "Skewness: asymmetry (~0 = symmetric, >0 = right tail, <0 = left tail). "
        "Kurtosis: tail heaviness (~0 = normal, >0 = heavy tails). "
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
                    figure=_BLANK,
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
                                figure=_BLANK,
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
                                figure=_BLANK,
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
                        ("6", "6"),
                        ("12", "12"),
                        ("24", "24"),
                        ("48", "48"),
                        ("100", "100"),
                    ]
                ],
            ),
            dcc.Loading(
                dcc.Graph(
                    id="rolling-graph",
                    figure=_BLANK,
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
                                figure=_BLANK,
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
                                figure=_BLANK,
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
    Output("ds-store", "data"),
    Output("ds-feature", "options"),
    Output("ds-feature", "value"),
    Input("ds-machine", "value"),
    Input("ds-split", "value"),
)
def load_data(
    machine_id: str | None, split: str,
) -> tuple[dict | None, list, str | None]:
    """Load raw SMD data into the store, filtered by split."""
    if not machine_id:
        return None, [], None

    result = load_smd_train_test(machine_id)
    if result is None:
        return None, [], None

    train_df, test_df, test_labels = result

    if split == "train":
        df = train_df.copy()
        df["is_anomaly"] = False
    elif split == "test":
        df = test_df.copy()
        df["is_anomaly"] = test_labels.values.astype(bool)
    else:  # both
        df = pd.concat([train_df, test_df], ignore_index=True)
        labels = pd.Series(False, index=df.index)
        labels.iloc[len(train_df):] = test_labels.values.astype(bool)
        df["is_anomaly"] = labels

    features = [c for c in df.columns if is_feature_column(c)]
    feat_options = [{"label": f, "value": f} for f in features]
    feat_default = features[0] if features else None

    return (
        {
            "df": df.to_json(orient="split"),
            "features": features,
            "machine_id": machine_id,
            "has_labels": True,
            "split": split,
            "n_train": len(train_df),
        },
        feat_options,
        feat_default,
    )


@callback(
    Output("stat-overview", "children"),
    Output("stat-central", "children"),
    Output("stat-spread", "children"),
    Output("stat-shape", "children"),
    Input("ds-store", "data"),
    Input("ds-feature", "value"),
)
def update_statistics(
    store: dict | None, feature: str | None,
) -> tuple:
    """Compute and display descriptive statistics for selected feature."""
    empty = html.Span("-", className="text-muted-light")
    if not store or not feature:
        return empty, empty, empty, empty

    df = pd.read_json(io.StringIO(store["df"]), orient="split")
    if feature not in df.columns:
        return empty, empty, empty, empty
    v = df[feature]

    def _row(label: str, val: str) -> html.Div:
        return html.Div(
            [
                html.Span(f"{label}: ", className="text-muted-light"),
                html.Span(val),
            ],
            className="mb-1",
        )

    n_features = len(store.get("features", []))
    n_labels = 0
    if store.get("has_labels") and "is_anomaly" in df.columns:
        n_labels = int(df["is_anomaly"].astype(bool).sum())

    overview = html.Div(
        [
            _row("Points", f"{len(df):,}"),
            _row("Features", f"{n_features}"),
            _row("Anomaly labels", f"{n_labels:,}"),
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
    Input("ds-feature", "value"),
    Input("ds-show-labels", "value"),
)
def update_timeseries(
    store: dict | None, feature: str | None, show_labels: bool,
) -> go.Figure:
    """Render the time series with optional anomaly labels."""
    if not store or not feature:
        return _empty_fig("Select a machine")

    df = pd.read_json(io.StringIO(store["df"]), orient="split")
    if feature not in df.columns:
        return _empty_fig(f"Feature {feature} not found")

    split = store.get("split", "both")
    n_train = store.get("n_train", 0)

    fig = go.Figure()

    if split == "both" and 0 < n_train < len(df):
        # Train portion
        fig.add_trace(
            go.Scatter(
                x=list(range(n_train)),
                y=df[feature].iloc[:n_train],
                mode="lines",
                name="Train",
                line={"color": "#636efa", "width": 1},
            )
        )
        # Test portion
        fig.add_trace(
            go.Scatter(
                x=list(range(n_train, len(df))),
                y=df[feature].iloc[n_train:],
                mode="lines",
                name="Test",
                line={"color": "#00e676", "width": 1},
            )
        )
        # Boundary line
        fig.add_vline(
            x=n_train, line_dash="dash", line_color="#888888", line_width=1,
            annotation_text="Train / Test",
            annotation_font_color="#888888",
            annotation_font_size=10,
        )
    else:
        color = "#636efa" if split != "test" else "#00e676"
        label = "Train" if split == "train" else ("Test" if split == "test" else feature)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=df[feature],
                mode="lines",
                name=label,
                line={"color": color, "width": 1},
            )
        )

    if show_labels and "is_anomaly" in df.columns:
        add_anomaly_zones(fig, df["is_anomaly"].astype(bool))

    fig.update_layout(
        **_DARK_LAYOUT,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#ffffff"),
        xaxis_title="Timestep",
        yaxis_title=feature,
    )
    return fig


@callback(
    Output("hist-graph", "figure"),
    Output("box-graph", "figure"),
    Input("ds-store", "data"),
    Input("ds-feature", "value"),
)
def update_distribution(
    store: dict | None, feature: str | None,
) -> tuple[go.Figure, go.Figure]:
    """Render histogram and box plot for selected feature."""
    if not store or not feature:
        return _empty_fig("No data"), _empty_fig("No data")

    df = pd.read_json(io.StringIO(store["df"]), orient="split")
    if feature not in df.columns:
        return _empty_fig("No data"), _empty_fig("No data")
    v = df[feature]

    hist_fig = px.histogram(
        x=v,
        nbins=80,
        labels={"x": feature, "count": "Count"},
        title=f"{feature} Distribution",
    )
    hist_fig.update_traces(marker_color="#636efa")
    hist_fig.update_layout(**_DARK_LAYOUT)

    box_fig = go.Figure(
        go.Violin(
            y=v,
            box_visible=True,
            meanline_visible=True,
            line_color="#636efa",
            fillcolor="rgba(99, 110, 250, 0.3)",
            name=feature,
            points=False,
        )
    )
    box_fig.update_layout(**_DARK_LAYOUT, title="Violin Plot")

    return hist_fig, box_fig


@callback(
    Output("rolling-graph", "figure"),
    Input("ds-store", "data"),
    Input("ds-feature", "value"),
    Input("rolling-tabs", "value"),
)
def update_rolling(
    store: dict | None, feature: str | None, window_str: str,
) -> go.Figure:
    """Render rolling mean and std."""
    if not store or not feature:
        return _empty_fig("No data")

    df = pd.read_json(io.StringIO(store["df"]), orient="split")
    if feature not in df.columns:
        return _empty_fig("No data")
    v = df[feature]
    window = int(window_str)
    x_axis = list(range(len(v)))

    rolling_mean = v.rolling(window=window, min_periods=1).mean()
    rolling_std = v.rolling(window=window, min_periods=1).std()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=v,
            mode="lines",
            name="Original",
            line={"color": "rgba(99,110,250,0.3)", "width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=rolling_mean,
            mode="lines",
            name=f"Rolling Mean ({window})",
            line={"color": "#00e676", "width": 2},
        )
    )
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    fig.add_trace(
        go.Scatter(
            x=x_axis + x_axis[::-1],
            y=pd.concat([upper, lower[::-1]]).tolist(),
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
        xaxis_title="Timestep",
        yaxis_title=feature,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#ffffff"),
    )
    return fig


@callback(
    Output("acf-graph", "figure"),
    Output("pacf-graph", "figure"),
    Input("ds-store", "data"),
    Input("ds-feature", "value"),
)
def update_acf(
    store: dict | None, feature: str | None,
) -> tuple[go.Figure, go.Figure]:
    """Render ACF and PACF bar charts."""
    if not store or not feature:
        return _empty_fig("No data"), _empty_fig("No data")

    df = pd.read_json(io.StringIO(store["df"]), orient="split")
    if feature not in df.columns:
        return _empty_fig("No data"), _empty_fig("No data")
    values = df[feature].dropna().values

    n_lags = min(60, len(values) // 4)
    if n_lags < 2:
        return _empty_fig("Not enough data"), _empty_fig("Not enough data")

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
