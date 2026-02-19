"""Model Analysis page — compare multivariate anomaly detection models on SMD."""

from __future__ import annotations

import contextlib
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
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from time_series_transformer.evaluation import (
    compute_best_f1,
    compute_detection_latency,
    compute_point_adjust_metrics,
)

dash.register_page(__name__, path="/models", name="Model Analysis", order=2)

# ---------------------------------------------------------------------------
# Chart theme
# ---------------------------------------------------------------------------

_DISPLAY_NAMES = {
    "custom_transformer_t2v": "Custom Transformer",
    "isolation_forest_mv": "Isolation Forest",
    "lstm_autoencoder": "LSTM Autoencoder",
    "tranad": "TranAD",
}


def _display_name(model: str) -> str:
    return _DISPLAY_NAMES.get(model, model)


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


# Transparent placeholder — invisible until callback replaces it,
# so only the dcc.Loading spinner is visible during initial load.
_BLANK = go.Figure()
_BLANK.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_visible=False,
    yaxis_visible=False,
    margin=dict(l=0, r=0, t=0, b=0),
)


# ---------------------------------------------------------------------------
# Layout components
# ---------------------------------------------------------------------------

_MACHINES = list_smd_machines()
_DEFAULT_MACHINE = _MACHINES[0]["value"] if _MACHINES else None


def _model_selector() -> dbc.Container:
    """Machine selector + model checklist."""
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
                                figure=_BLANK,
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
            dbc.Row(
                [
                    dbc.Col(
                        html.H3("Detection Results", className="mb-0"),
                        width="auto",
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="ma-feature",
                            options=[],
                            value=None,
                            clearable=False,
                            style={"minWidth": "100px"},
                        ),
                        width="auto",
                    ),
                ],
                className="align-items-center g-3 mb-3",
            ),
            dcc.Loading(
                dcc.Graph(
                    id="ma-timeseries",
                    figure=_BLANK,
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
                    figure=_BLANK,
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
    Output("ma-models", "options"),
    Output("ma-models", "value"),
    Output("ma-models-status", "children"),
    Output("ma-feature", "options"),
    Output("ma-feature", "value"),
    Input("ma-store", "data"),
)
def update_selectors(
    store: dict | None,
) -> tuple[list, list, html.Span | None, list, str | None]:
    """Populate model checklist and feature dropdown from store."""
    if not store:
        hint = html.Span(
            [html.I(className="bi bi-question-circle me-1"), "Loading..."],
            className="text-muted-light small",
        )
        return [], [], hint, [], None

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
        return [], [], hint, feat_options, feat_default

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
                    html.Span(_display_name(m)),
                ]
            ),
            "value": m,
        }
        for m in models
    ]

    return model_options, models, None, feat_options, feat_default


# Metrics displayed as bar charts (subset) vs all metrics computed internally
_DISPLAY_METRICS = ["Precision", "Recall", "AUC-ROC", "PA-F1"]


@callback(
    *[Output(f"ma-bar-{i}", "figure") for i in range(4)],
    Input("ma-store", "data"),
    Input("ma-models", "value"),
)
def update_metric_bars(store: dict | None, selected: list[str]) -> tuple:
    """Compute all metrics from artifact CSV, display top 4 as bar charts."""
    empty = tuple(_empty_fig("No results yet.") for _ in range(4))
    if not store or store.get("error") or "artifacts" not in store:
        return empty

    df = pd.read_json(io.StringIO(store["artifacts"]), orient="split")
    models = store.get("models", [])
    color_map = store.get("color_map", {})
    display_color_map = {_display_name(k): v for k, v in color_map.items()}
    selected = enforce_min_one(selected, models)

    if "is_anomaly" not in df.columns:
        return tuple(_empty_fig("No ground truth") for _ in range(4))

    y_true_s = pd.Series(df["is_anomaly"].astype(int).values)
    y_true_arr = y_true_s.values

    rows: list[dict] = []
    for model in selected:
        score_col = f"{model}_score"
        anom_col = f"{model}_is_anomaly"
        if anom_col not in df.columns:
            continue

        y_pred_s = pd.Series(df[anom_col].astype(int).values)
        y_pred_arr = y_pred_s.values
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_arr,
            y_pred_arr,
            average="binary",
            zero_division=0,
        )

        scores_s = None
        auc_roc = 0.0
        auc_pr = 0.0
        best_f1 = 0.0
        if score_col in df.columns:
            scores_s = pd.Series(pd.to_numeric(df[score_col], errors="coerce").fillna(0).values)
            if len(np.unique(y_true_arr)) > 1:
                with contextlib.suppress(ValueError):
                    auc_roc = roc_auc_score(y_true_arr, scores_s.values)
                with contextlib.suppress(ValueError):
                    auc_pr = average_precision_score(y_true_arr, scores_s.values)
                bf = compute_best_f1(y_true_s, scores_s, n_thresholds=50)
                best_f1 = bf.f1

        # Point-adjust F1
        pa = compute_point_adjust_metrics(y_true_s, y_pred_s)

        # Detection latency (computed but not displayed as bar chart)
        dl = compute_detection_latency(y_true_s, y_pred_s)

        rows.append(
            {
                "model": _display_name(model),
                "F1": f1,
                "Precision": prec,
                "Recall": rec,
                "AUC-ROC": auc_roc,
                "PA-F1": pa.f1,
                "Best-F1": best_f1,
                "AUC-PR": auc_pr,
                "Latency": dl.mean_latency,
            }
        )

    if not rows:
        return tuple(_empty_fig("No model data") for _ in range(4))

    metrics_df = pd.DataFrame(rows)
    model_order = metrics_df.sort_values("F1", ascending=False)["model"].tolist()

    figs = []
    for metric in _DISPLAY_METRICS:
        fig = px.bar(
            metrics_df,
            x="model",
            y=metric,
            color="model",
            color_discrete_map=display_color_map,
            category_orders={"model": model_order},
            labels={"model": "", metric: metric},
            title=metric,
        )
        fig.update_traces(
            hovertemplate="%{y:.3f}<extra>%{x}</extra>",
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
    store: dict | None,
    selected: list[str],
    feature: str | None,
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
            fig,
            df["is_anomaly"].astype(bool),
            label="Ground Truth",
            row=1,
            col=1,
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
                name=_display_name(model),
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
                name=f"{_display_name(model)} (score)",
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
                name=_display_name(model),
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
