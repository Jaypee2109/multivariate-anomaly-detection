"""Model Analysis page — compare multivariate anomaly detection models on SMD."""

from __future__ import annotations

import io

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from datasets import (
    add_anomaly_zones,
    build_color_map,
    discover_smd_features,
    discover_smd_models,
    enforce_min_one,
    list_smd_machines,
    load_smd_results,
    load_smd_train_test,
)
from plotly.subplots import make_subplots
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
)

dash.register_page(__name__, path="/models", name="Model Analysis", order=2)

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
# Layout components
# ---------------------------------------------------------------------------

_MACHINES = list_smd_machines()
_DEFAULT_MACHINE = _MACHINES[0]["value"] if _MACHINES else None


def _model_selector() -> dbc.Container:
    """Machine selector + model checklist + feature dropdown."""
    machines = _MACHINES
    return dbc.Container(
        [
            html.H3("Model Selection", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Machine", className="small text-muted-light"),
                            dcc.Dropdown(
                                id="ma-machine",
                                options=machines,
                                value=_DEFAULT_MACHINE,
                                clearable=False,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Feature", className="small text-muted-light"),
                            dcc.Dropdown(
                                id="ma-feature",
                                options=[],
                                value=None,
                                clearable=False,
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            html.Label("Models", className="small text-muted-light"),
                            html.Div(
                                [
                                    dcc.Checklist(
                                        id="ma-models",
                                        options=[],
                                        value=[],
                                        inline=True,
                                        className="model-checklist",
                                    ),
                                    html.Div(id="ma-models-status"),
                                ],
                                className="model-checklist-wrapper",
                            ),
                        ],
                    ),
                ],
                className="align-items-start g-3",
            ),
        ],
        fluid=True,
        className="card-1 py-3 shadow rounded-3",
    )


def _metric_bars_section() -> dbc.Container:
    return dbc.Container(
        [
            html.H3("Metric Comparison", className="mb-3"),
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
    Input("ma-machine", "value"),
)
def load_store(machine_id: str | None) -> dict | None:
    """Load SMD artifact CSV into the store (falls back to raw data for features)."""
    if not machine_id:
        return {"error": "No machine selected."}

    # Always derive features from raw SMD data so the dropdown is populated
    raw = load_smd_train_test(machine_id)
    raw_features = list(raw[0].columns) if raw else []

    df = load_smd_results(machine_id)
    if df is None:
        # No artifacts yet — return features only, no models
        return {
            "features": raw_features,
            "models": [],
            "color_map": {},
            "machine_id": machine_id,
            "error": "No model results yet. Run the pipeline first.",
        }

    models = discover_smd_models(df)
    features = discover_smd_features(df) or raw_features

    color_map = build_color_map(models) if models else {}

    return {
        "artifacts": df.to_json(orient="split"),
        "models": models,
        "features": features,
        "color_map": color_map,
        "machine_id": machine_id,
        "error": "" if models else "No model results yet. Run the pipeline first.",
    }


@callback(
    Output("ma-feature", "options"),
    Output("ma-feature", "value"),
    Output("ma-models", "options"),
    Output("ma-models", "value"),
    Output("ma-models-status", "children"),
    Input("ma-store", "data"),
)
def update_selectors(
    store: dict | None,
) -> tuple[list, str | None, list, list, html.Span | None]:
    """Populate feature dropdown and model checklist from store."""
    if not store:
        hint = html.Span(
            [html.I(className="bi bi-question-circle me-1"), "Loading..."],
            className="text-muted-light small",
        )
        return [], None, [], [], hint

    features = store.get("features", [])
    feat_options = [{"label": f, "value": f} for f in features]
    feat_default = features[0] if features else None

    models = store.get("models", [])
    error = store.get("error", "")

    if not models:
        hint = html.Span(
            [html.I(className="bi bi-info-circle me-1"), "No results yet."],
            className="text-muted-light small",
            title=error or "Run the multivariate pipeline to generate model artifacts.",
        )
        return feat_options, feat_default, [], [], hint

    color_map = store.get("color_map", {})
    model_options = [
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

    return feat_options, feat_default, model_options, models, None


@callback(
    Output("ma-bar-0", "figure"),
    Output("ma-bar-1", "figure"),
    Output("ma-bar-2", "figure"),
    Output("ma-bar-3", "figure"),
    Input("ma-store", "data"),
    Input("ma-models", "value"),
)
def update_metric_bars(
    store: dict | None, selected: list[str]
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Compute and render metric bar charts from the artifact CSV."""
    if not store or store.get("error") or "artifacts" not in store:
        return tuple(_empty_fig("No results yet.") for _ in range(4))

    df = pd.read_json(io.StringIO(store["artifacts"]), orient="split")
    models = store.get("models", [])
    color_map = store.get("color_map", {})
    selected = enforce_min_one(selected, models)

    if "is_anomaly" not in df.columns:
        return tuple(_empty_fig("No ground truth") for _ in range(4))

    y_true = df["is_anomaly"].astype(int).values

    # Compute metrics per selected model
    metric_names = ["F1", "Precision", "Recall", "AUC-ROC"]
    rows: list[dict] = []
    for model in selected:
        score_col = f"{model}_score"
        anom_col = f"{model}_is_anomaly"
        if anom_col not in df.columns:
            continue

        y_pred = df[anom_col].astype(int).values
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0,
        )

        auc = None
        if score_col in df.columns:
            scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0).values
            if len(np.unique(y_true)) > 1:
                try:
                    auc = roc_auc_score(y_true, scores)
                except ValueError:
                    pass

        rows.append({
            "model": model,
            "F1": f1,
            "Precision": prec,
            "Recall": rec,
            "AUC-ROC": auc if auc is not None else 0.0,
        })

    if not rows:
        return tuple(_empty_fig("No model data") for _ in range(4))

    metrics_df = pd.DataFrame(rows)

    figs = []
    for metric in metric_names:
        subset = metrics_df[["model", metric]].sort_values(metric, ascending=True)
        fig = px.bar(
            subset,
            x="model",
            y=metric,
            color="model",
            color_discrete_map=color_map,
            category_orders={"model": subset["model"].tolist()},
            labels={"model": "", metric: metric},
            title=metric,
        )
        fig.update_layout(**_DARK_LAYOUT, showlegend=False, bargap=0.3)
        figs.append(fig)

    return tuple(figs)


@callback(
    Output("ma-timeseries", "figure"),
    Input("ma-store", "data"),
    Input("ma-models", "value"),
    Input("ma-feature", "value"),
)
def update_timeseries(
    store: dict | None, selected: list[str], feature: str | None,
) -> go.Figure:
    """Render detection results: feature + anomaly markers (top) and scores (bottom)."""
    if not store or store.get("error") or "artifacts" not in store:
        return _empty_fig("No results yet.")

    df = pd.read_json(io.StringIO(store["artifacts"]), orient="split")
    models = store.get("models", [])
    color_map = store.get("color_map", {})
    selected = enforce_min_one(selected, models)

    if not feature or feature not in df.columns:
        features = store.get("features", [])
        feature = features[0] if features else None
    if not feature:
        return _empty_fig("No features available")

    x_axis = list(range(len(df)))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.6, 0.4],
        subplot_titles=[f"{feature} & Anomalies", "Anomaly Scores"],
    )

    # Top subplot: selected feature line
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=df[feature],
            mode="lines",
            name=feature,
            line={"color": "#ffffff", "width": 1},
        ),
        row=1,
        col=1,
    )

    # Ground truth anomaly zones
    if "is_anomaly" in df.columns:
        add_anomaly_zones(
            fig, df["is_anomaly"].astype(bool),
            label="Ground Truth", row=1, col=1,
        )

    # Per-model anomaly markers
    for model in selected:
        anom_col = f"{model}_is_anomaly"
        score_col = f"{model}_score"
        if anom_col not in df.columns:
            continue

        color = color_map.get(model, "#636efa")
        anom_mask = df[anom_col].astype(bool)
        if not anom_mask.any():
            continue

        anom_idx = [i for i, v in enumerate(anom_mask) if v]
        hover = None
        if score_col in df.columns:
            scores = pd.to_numeric(df[score_col], errors="coerce")
            hover = [f"Score: {scores.iloc[i]:.4g}" for i in anom_idx]

        fig.add_trace(
            go.Scatter(
                x=anom_idx,
                y=df[feature].iloc[anom_idx],
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

    # Bottom subplot: anomaly score lines
    for model in selected:
        score_col = f"{model}_score"
        if score_col not in df.columns:
            continue

        color = color_map.get(model, "#636efa")
        fig.add_trace(
            go.Scatter(
                x=x_axis,
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
    fig.update_xaxes(title_text="Timestep", row=2, col=1)
    fig.update_yaxes(title_text=feature, row=1, col=1)
    fig.update_yaxes(title_text="Anomaly Score", row=2, col=1)

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
        return _empty_fig("No data")

    df = pd.read_json(io.StringIO(store["artifacts"]), orient="split")
    models = store.get("models", [])
    color_map = store.get("color_map", {})
    selected = enforce_min_one(selected, models)

    fig = go.Figure()
    for model in selected:
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
