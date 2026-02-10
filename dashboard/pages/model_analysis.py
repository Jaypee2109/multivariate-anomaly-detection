"""Model Analysis page — compare anomaly detection models."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from mlflow_loader import (
    build_color_map,
    discover_models_from_artifacts,
    enforce_min_one,
    load_artifacts_csv,
    load_data_run_params,
    load_mlflow_runs,
)
from plotly.subplots import make_subplots

dash.register_page(__name__, path="/models", name="Model Analysis")

# ---------------------------------------------------------------------------
# Chart theme
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


# ---------------------------------------------------------------------------
# Metric definitions (point-level and range-level)
# ---------------------------------------------------------------------------

POINT_METRICS = [
    ("metrics.point/f1", "F1"),
    ("metrics.point/precision", "Precision"),
    ("metrics.point/recall", "Recall"),
    ("metrics.point/auc_roc", "AUC-ROC"),
]

RANGE_METRICS = [
    ("metrics.range/f1", "F1"),
    ("metrics.range/precision", "Precision"),
    ("metrics.range/recall", "Recall"),
    ("metrics.anomaly_rate", "Anomaly Rate"),
]


# ---------------------------------------------------------------------------
# Layout components
# ---------------------------------------------------------------------------


def _model_selector() -> dbc.Container:
    """Model selection card — first box on the page, includes page title."""
    return dbc.Container(
        [
            html.H3("Model Selection", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        html.Label("Models", className="small text-muted-light"),
                        width="auto",
                    ),
                    dbc.Col(
                        dcc.Checklist(
                            id="ma-models",
                            options=[],
                            value=[],
                            inline=True,
                            className="model-checklist",
                        ),
                    ),
                ],
                className="align-items-center",
            ),
        ],
        fluid=True,
        className="card-1 py-3 shadow rounded-3",
    )


def _metric_bars_section() -> dbc.Container:
    return dbc.Container(
        [
            html.H3("Metric Comparison", className="mb-3"),
            dcc.Tabs(
                id="ma-metric-tabs",
                value="point",
                children=[
                    dcc.Tab(
                        label="Point",
                        value="point",
                        className="diagramm-tab-left",
                        selected_className="selected-diagramm-tab",
                    ),
                    dcc.Tab(
                        label="Range",
                        value="range",
                        className="diagramm-tab-right",
                        selected_className="selected-diagramm-tab",
                    ),
                ],
            ),
            html.Hr(style={"borderColor": "#333", "margin": "0 0 4px 0"}),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            dcc.Graph(
                                id=f"ma-bar-{i}",
                                config={"displayModeBar": "hover"},
                                style={"height": "320px"},
                            ),
                            type="default",
                            color="#636efa",
                        ),
                        md=3,
                    )
                    for i in range(4)
                ],
                className="g-0",
            ),
        ],
        fluid=True,
        className="card-1 py-3 mt-4 shadow rounded-3",
    )


def _timeseries_section() -> dbc.Container:
    return dbc.Container(
        [
            html.H3("Detection Results", className="mb-3"),
            dcc.Loading(
                dcc.Graph(
                    id="ma-timeseries",
                    config={"displayModeBar": "hover", "scrollZoom": True},
                    style={"height": "600px"},
                ),
                type="default",
                color="#636efa",
            ),
        ],
        fluid=True,
        className="card-1 py-3 mt-4 shadow rounded-3",
    )


def _distributions_section() -> dbc.Container:
    return dbc.Container(
        [
            html.H3("Score Distributions", className="mb-3"),
            dcc.Loading(
                dcc.Graph(
                    id="ma-distributions",
                    config={"displayModeBar": "hover"},
                    style={"height": "350px"},
                ),
                type="default",
                color="#636efa",
            ),
        ],
        fluid=True,
        className="card-1 py-3 mt-4 shadow rounded-3",
    )


def _config_section() -> dbc.Container:
    return dbc.Container(
        [
            html.H3("Model Configuration", className="mb-3"),
            html.Div(
                id="ma-config-table",
                className="dark-scroll",
                style={"maxHeight": "450px", "overflowY": "auto"},
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
        dcc.Store(id="ma-store", storage_type="memory"),
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        _model_selector(),
                        _metric_bars_section(),
                        _timeseries_section(),
                        _distributions_section(),
                        _config_section(),
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
    Output("ma-store", "data"),
    Input("ma-timeseries", "id"),  # fires once on page load
)
def load_store(_: str) -> dict | None:
    """Load MLflow runs and artifact data into the store."""
    runs_df = load_mlflow_runs()
    artifacts_df = load_artifacts_csv()

    if runs_df is None and artifacts_df is None:
        return {"error": "No MLflow runs or artifact data found."}

    # Discover models from artifacts
    artifact_models = (
        discover_models_from_artifacts(artifacts_df) if artifacts_df is not None else []
    )

    # Discover models from MLflow
    mlflow_models = sorted(runs_df["model_name"].unique().tolist()) if runs_df is not None else []

    # Union of both sources
    all_models = sorted(set(artifact_models + mlflow_models))
    if not all_models:
        return {"error": "No models found in MLflow or artifacts."}

    color_map = build_color_map(all_models)

    store: dict = {
        "models": all_models,
        "color_map": color_map,
        "error": "",
    }

    if runs_df is not None:
        store["runs"] = runs_df.to_dict("records")
        store["runs_columns"] = list(runs_df.columns)

    if artifacts_df is not None:
        store["artifacts"] = artifacts_df.to_json(date_format="iso", orient="split")
        store["artifact_models"] = artifact_models

    store["data_params"] = load_data_run_params()

    return store


@callback(
    Output("ma-models", "options"),
    Output("ma-models", "value"),
    Input("ma-store", "data"),
)
def update_model_checklist(store: dict | None) -> tuple[list, list]:
    """Build the model checklist with colored dots."""
    if not store or store.get("error"):
        return [], []

    models = store.get("models", [])
    color_map = store.get("color_map", {})

    options = [
        {
            "label": html.Span(
                [
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "10px",
                            "height": "10px",
                            "borderRadius": "50%",
                            "backgroundColor": color_map.get(m, "#636efa"),
                            "marginRight": "6px",
                        }
                    ),
                    html.Span(m),
                ]
            ),
            "value": m,
        }
        for m in models
    ]

    return options, models  # all selected by default


@callback(
    Output("ma-bar-0", "figure"),
    Output("ma-bar-1", "figure"),
    Output("ma-bar-2", "figure"),
    Output("ma-bar-3", "figure"),
    Input("ma-store", "data"),
    Input("ma-metric-tabs", "value"),
    Input("ma-models", "value"),
)
def update_metric_bars(
    store: dict | None, metric_level: str, selected: list[str]
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Render 4 metric bar charts."""
    empty = _empty_fig("No data")
    if not store or store.get("error") or "runs" not in store:
        msg = store.get("error", "No MLflow data") if store else "Loading..."
        return (
            _empty_fig(msg),
            _empty_fig(msg),
            _empty_fig(msg),
            _empty_fig(msg),
        )

    import pandas as pd

    runs_df = pd.DataFrame(store["runs"])
    color_map = store.get("color_map", {})
    models = store.get("models", [])
    selected = enforce_min_one(selected, models)

    metrics = POINT_METRICS if metric_level == "point" else RANGE_METRICS

    figs = []
    for col, display_name in metrics:
        if col not in runs_df.columns:
            figs.append(_empty_fig(f"{display_name}: N/A"))
            continue

        subset = runs_df[runs_df["model_name"].isin(selected)].copy()
        subset = subset[["model_name", col]]

        if subset.empty:
            figs.append(_empty_fig(f"{display_name}: No data"))
            continue

        # Sort by value but keep models with NaN at the start
        subset = subset.sort_values(col, ascending=True, na_position="first")

        fig = px.bar(
            subset,
            x="model_name",
            y=col,
            color="model_name",
            color_discrete_map=color_map,
            category_orders={"model_name": subset["model_name"].tolist()},
            labels={"model_name": "", col: display_name},
            title=display_name,
        )
        fig.update_layout(**_DARK_LAYOUT, showlegend=False)
        fig.update_layout(
            bargap=0.3,
            yaxis_title=display_name,
        )
        figs.append(fig)

    while len(figs) < 4:
        figs.append(empty)

    return tuple(figs)  # type: ignore[return-value]


@callback(
    Output("ma-timeseries", "figure"),
    Input("ma-store", "data"),
    Input("ma-models", "value"),
)
def update_timeseries(store: dict | None, selected: list[str]) -> go.Figure:
    """Render detection results: value + anomaly markers (top) and scores (bottom)."""
    if not store or store.get("error") or "artifacts" not in store:
        msg = store.get("error", "No artifact data") if store else "Loading..."
        return _empty_fig(msg)

    import pandas as pd

    df = pd.read_json(store["artifacts"], orient="split")
    artifact_models = store.get("artifact_models", [])
    color_map = store.get("color_map", {})
    models = store.get("models", [])
    selected = enforce_min_one(selected, models)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.6, 0.4],
        subplot_titles=["Time Series & Anomalies", "Anomaly Scores"],
    )

    # --- Top subplot: original time series + anomaly markers ---
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines",
            name="Value",
            line={"color": "#ffffff", "width": 1},
        ),
        row=1,
        col=1,
    )

    for model in selected:
        if model not in artifact_models:
            continue
        score_col = f"{model}_score"
        anom_col = f"{model}_is_anomaly"
        if anom_col not in df.columns:
            continue

        color = color_map.get(model, "#636efa")
        anom_mask = df[anom_col].astype(bool)
        if not anom_mask.any():
            continue

        anom_pts = df[anom_mask].copy()

        hover = None
        if score_col in df.columns:
            scores = pd.to_numeric(anom_pts[score_col], errors="coerce")
            hover = [f"Score: {s:.4g}" if pd.notna(s) else "" for s in scores]

        fig.add_trace(
            go.Scatter(
                x=anom_pts["timestamp"],
                y=anom_pts["value"],
                mode="markers",
                name=model,
                marker={"color": color, "size": 6, "symbol": "x"},
                text=hover,
                hovertemplate="%{x}<br>Value: %{y:.4g}<br>%{text}<extra>%{fullData.name}</extra>"
                if hover
                else None,
                legendgroup=model,
            ),
            row=1,
            col=1,
        )

    # --- Bottom subplot: anomaly score lines ---
    for model in selected:
        if model not in artifact_models:
            continue
        score_col = f"{model}_score"
        if score_col not in df.columns:
            continue

        color = color_map.get(model, "#636efa")
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=pd.to_numeric(df[score_col], errors="coerce"),
                mode="lines",
                name=f"{model} (score)",
                line={"color": color, "width": 1.5},
                legendgroup=model,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#ffffff"),
    )
    fig.update_xaxes(gridcolor="#333")
    fig.update_yaxes(gridcolor="#333")
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Anomaly Score", row=2, col=1)

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.update(font_color="#aaaaaa", font_size=12)

    return fig


@callback(
    Output("ma-distributions", "figure"),
    Input("ma-store", "data"),
    Input("ma-models", "value"),
)
def update_distributions(store: dict | None, selected: list[str]) -> go.Figure:
    """Render overlaid histograms of anomaly scores per model."""
    if not store or store.get("error") or "artifacts" not in store:
        return _empty_fig("No artifact data")

    import pandas as pd

    df = pd.read_json(store["artifacts"], orient="split")
    artifact_models = store.get("artifact_models", [])
    color_map = store.get("color_map", {})
    models = store.get("models", [])
    selected = enforce_min_one(selected, models)

    fig = go.Figure()
    for model in selected:
        if model not in artifact_models:
            continue
        score_col = f"{model}_score"
        if score_col not in df.columns:
            continue

        scores = pd.to_numeric(df[score_col], errors="coerce").dropna()
        if scores.empty:
            continue

        fig.add_trace(
            go.Histogram(
                x=scores,
                name=model,
                marker_color=color_map.get(model, "#636efa"),
                opacity=0.6,
                nbinsx=60,
            )
        )

    fig.update_layout(
        **_DARK_LAYOUT,
        barmode="overlay",
        xaxis_title="Anomaly Score",
        yaxis_title="Count",
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#ffffff"),
    )

    return fig


@callback(
    Output("ma-config-table", "children"),
    Input("ma-store", "data"),
    Input("ma-models", "value"),
)
def update_config_table(store: dict | None, selected: list[str]) -> html.Div:
    """Render a parameter comparison table from MLflow run params."""
    if not store or store.get("error") or "runs" not in store:
        msg = store.get("error", "No MLflow data") if store else "Loading..."
        return html.Span(msg, className="text-muted-light")

    import pandas as pd

    runs_df = pd.DataFrame(store["runs"])
    all_models = store.get("models", [])
    selected = enforce_min_one(selected, all_models)

    # Filter to selected models
    runs_df = runs_df[runs_df["model_name"].isin(selected)]

    # Extract param columns (params.*)
    param_cols = [c for c in runs_df.columns if c.startswith("params.")]

    # Also include key metrics
    extra_cols = ["metrics.fit_time_seconds", "metrics.anomaly_rate", "metrics.test_size"]
    display_cols = param_cols + [c for c in extra_cols if c in runs_df.columns]

    if not display_cols:
        return html.Span("No parameters available.", className="text-muted-light")

    # Build a transposed table: rows = parameters, columns = models
    models = sorted(runs_df["model_name"].unique().tolist())

    # Collect data-level params to avoid duplicating them in the model section
    data_params = store.get("data_params", {})
    data_param_keys = set(data_params.keys())

    rows = []
    for col in display_cols:
        # Skip columns covered by data_params (appended at the end)
        if col in data_param_keys:
            continue
        # Clean up display name
        display = col.replace("params.", "").replace("metrics.", "")
        values = []
        for model in models:
            model_row = runs_df[runs_df["model_name"] == model]
            if model_row.empty or col not in model_row.columns:
                values.append("—")
            else:
                val = model_row.iloc[0][col]
                if pd.isna(val):
                    values.append("—")
                elif isinstance(val, float):
                    values.append(f"{val:.4g}")
                else:
                    values.append(str(val))

        # Skip rows where all values are "—"
        if all(v == "—" for v in values):
            continue
        rows.append((display, values))

    # Append dataset-level params at the end (shared across all models)
    for key, val in data_params.items():
        display = key.replace("params.", "")
        rows.append((display, [val] * len(models)))

    if not rows:
        return html.Span("No parameters available.", className="text-muted-light")

    header = html.Thead(
        html.Tr(
            [html.Th("Parameter", className="config-table-th")]
            + [html.Th(m, className="config-table-th") for m in models]
        )
    )
    body = html.Tbody(
        [
            html.Tr(
                [html.Td(display, className="config-table-td")]
                + [html.Td(v, className="config-table-td") for v in values]
            )
            for display, values in rows
        ]
    )

    return html.Table(
        [header, body],
        className="config-table",
    )
