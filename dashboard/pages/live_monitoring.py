"""Live Monitoring page — simulate real-time anomaly detection."""

from __future__ import annotations

import logging

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html
from datasets import discover_categories, discover_datasets, load_dataset
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Track last model-poll status to avoid spamming the CLI every 5s
_last_model_status: str | None = None

dash.register_page(__name__, path="/live", name="Live Monitoring", order=3)

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

# Stable color palette for models
_PALETTE = [
    "#636efa",
    "#EF553B",
    "#00cc96",
    "#ab63fa",
    "#FFA15A",
    "#19d3f3",
    "#FF6692",
    "#B6E880",
]

SPEED_OPTIONS = [
    {"label": "1 pt/s", "value": 1},
    {"label": "5 pt/s", "value": 5},
    {"label": "10 pt/s", "value": 10},
    {"label": "25 pt/s", "value": 25},
    {"label": "50 pt/s", "value": 50},
]

# Rolling window size for the chart
CHART_WINDOW = 500


def _empty_fig(msg: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**_DARK_LAYOUT, title=msg)
    return fig


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _controls_section() -> dbc.Container:
    categories = discover_categories()
    default_cat = categories[0]["value"] if categories else None
    default_datasets = discover_datasets(default_cat) if default_cat else []

    return dbc.Container(
        [
            html.H3("Live Monitoring", className="mb-3"),
            dbc.Row(
                [
                    # Dataset selectors
                    dbc.Col(
                        [
                            html.Label("Category", className="small text-muted-light"),
                            dcc.Dropdown(
                                id="lm-category",
                                options=categories,
                                value=default_cat,
                                clearable=False,
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            html.Label("Dataset", className="small text-muted-light"),
                            dcc.Dropdown(
                                id="lm-dataset",
                                options=default_datasets,
                                value=default_datasets[0]["value"] if default_datasets else None,
                                clearable=False,
                            ),
                        ],
                        md=3,
                    ),
                    # Model selector
                    dbc.Col(
                        [
                            html.Label("Models", className="small text-muted-light"),
                            html.Div(
                                [
                                    dcc.Checklist(
                                        id="lm-models",
                                        options=[],
                                        value=[],
                                        inline=True,
                                        className="model-checklist",
                                    ),
                                    html.Div(id="lm-models-status"),
                                ],
                                className="model-checklist-wrapper",
                            ),
                        ],
                        md=3,
                    ),
                    # Speed selector
                    dbc.Col(
                        [
                            html.Label("Speed", className="small text-muted-light"),
                            dcc.Dropdown(
                                id="lm-speed",
                                options=SPEED_OPTIONS,
                                value=5,
                                clearable=False,
                            ),
                        ],
                        md=1,
                    ),
                    # Playback controls
                    dbc.Col(
                        html.Div(
                            [
                                dbc.Button(
                                    html.I(className="bi bi-play-fill"),
                                    id="lm-play-btn",
                                    color="success",
                                    size="sm",
                                    className="me-1",
                                ),
                                dbc.Button(
                                    html.I(className="bi bi-pause-fill"),
                                    id="lm-pause-btn",
                                    color="warning",
                                    size="sm",
                                    className="me-1",
                                ),
                                dbc.Button(
                                    html.I(className="bi bi-skip-start-fill"),
                                    id="lm-reset-btn",
                                    color="danger",
                                    size="sm",
                                ),
                            ],
                            className="d-flex align-items-end h-100 pb-1",
                        ),
                        width="auto",
                    ),
                    # Status indicators
                    dbc.Col(
                        html.Div(
                            [
                                html.Span(id="lm-api-status", className="small"),
                                html.Span(
                                    id="lm-progress",
                                    className="small text-muted-light ms-3",
                                ),
                                html.Span(
                                    id="lm-latency",
                                    className="small text-muted-light ms-3",
                                ),
                            ],
                            className="d-flex align-items-end h-100 pb-1",
                        ),
                        width="auto",
                    ),
                ],
                className="g-3 align-items-end",
            ),
        ],
        fluid=True,
        className="card-1 py-3 shadow rounded-3",
    )


def _chart_section() -> dbc.Container:
    return dbc.Container(
        [
            dcc.Graph(
                id="lm-chart",
                config={"displayModeBar": "hover", "scrollZoom": True},
                style={"height": "600px"},
                figure=_empty_fig("Select a dataset and press Play"),
            ),
        ],
        fluid=True,
        className="card-1 py-3 mt-4 shadow rounded-3",
    )


def _summary_section() -> dbc.Container:
    return dbc.Container(
        [
            html.H3("Detection Summary", className="mb-3"),
            html.Div(id="lm-summary", children=[]),
        ],
        fluid=True,
        className="card-1 py-3 mt-4 mb-4 shadow rounded-3",
    )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

layout = html.Div(
    [
        # Stores
        dcc.Store(id="lm-dataset-store", storage_type="memory"),
        dcc.Store(id="lm-results-store", storage_type="memory"),
        dcc.Store(id="lm-state-store", storage_type="memory", data={"index": 0}),
        # Interval timer (disabled by default)
        dcc.Interval(id="lm-interval", interval=1000, disabled=True, n_intervals=0),
        # Slow interval to refresh model list / API status
        dcc.Interval(id="lm-model-refresh", interval=5000, n_intervals=0),
        # Layout
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        _controls_section(),
                        _chart_section(),
                        _summary_section(),
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
    Output("lm-dataset", "options"),
    Output("lm-dataset", "value"),
    Input("lm-category", "value"),
)
def update_dataset_options(category: str | None) -> tuple[list, str | None]:
    """Populate dataset dropdown when category changes."""
    if not category:
        return [], None
    try:
        datasets = discover_datasets(category)
    except Exception:
        logger.warning("Failed to discover datasets for %s", category, exc_info=True)
        return [], None
    first = datasets[0]["value"] if datasets else None
    return datasets, first


@callback(
    Output("lm-dataset-store", "data"),
    Output("lm-results-store", "data", allow_duplicate=True),
    Output("lm-state-store", "data", allow_duplicate=True),
    Input("lm-dataset", "value"),
    prevent_initial_call=True,
)
def load_dataset_into_store(
    rel_path: str | None,
) -> tuple[dict | None, None, dict]:
    """Load the selected dataset and reset simulation state + cache."""
    _reset = (None, None, {"index": 0})
    if not rel_path:
        return _reset
    try:
        df = load_dataset(rel_path)
    except Exception:
        logger.warning("Failed to load dataset: %s", rel_path, exc_info=True)
        return _reset
    if df is None or df.empty:
        return _reset

    # Validate required columns
    if "timestamp" not in df.columns or "value" not in df.columns:
        logger.warning(
            "Dataset %s missing required columns (has: %s)", rel_path, list(df.columns)
        )
        return _reset

    return (
        {
            "timestamps": df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            "values": df["value"].tolist(),
            "name": rel_path.split("/")[-1],
            "total": len(df),
        },
        None,
        {"index": 0},
    )


@callback(
    Output("lm-models", "options"),
    Output("lm-models", "value"),
    Output("lm-models-status", "children"),
    Input("lm-dataset-store", "data"),
    Input("lm-model-refresh", "n_intervals"),
    State("lm-models", "value"),
)
def populate_models(
    _data: dict | None,
    _n: int,
    current_selection: list[str],
) -> tuple[list, list, html.Span | None]:
    """Query API for loaded models and build checklist."""
    from api_client import get_client

    def _status_hint(label: str, tooltip: str) -> tuple[list, list, html.Span]:
        """Return empty checklist + short label with detailed hover tooltip."""
        global _last_model_status  # noqa: PLW0603
        if _last_model_status != label:
            _last_model_status = label
            logger.warning("Live Monitoring: %s", tooltip)
        return (
            [],
            [],
            html.Span(
                [html.I(className="bi bi-question-circle me-1"), label],
                title=tooltip,
                className="text-muted-light small",
                style={"cursor": "help"},
            ),
        )

    try:
        client = get_client()
        health = client.get_health()
    except Exception:
        logger.warning("Failed to contact inference API", exc_info=True)
        return _status_hint("API error", "Check server logs for details")

    if health is None:
        return _status_hint(
            "API offline",
            "Start with: python -m time_series_transformer serve",
        )

    model_slugs = health.get("models_loaded", [])
    if not isinstance(model_slugs, list) or not model_slugs:
        return _status_hint(
            "No models loaded",
            "Train with: python -m time_series_transformer train --save-checkpoints",
        )

    global _last_model_status  # noqa: PLW0603
    _last_model_status = None

    options = []
    for i, slug in enumerate(sorted(model_slugs)):
        color = _PALETTE[i % len(_PALETTE)]
        options.append(
            {
                "label": html.Span(
                    [
                        html.Span(
                            style={
                                "display": "inline-block",
                                "width": "10px",
                                "height": "10px",
                                "borderRadius": "50%",
                                "backgroundColor": color,
                                "marginRight": "6px",
                            }
                        ),
                        html.Span(slug),
                    ]
                ),
                "value": slug,
            }
        )

    # Preserve current selection if still valid, otherwise select all
    valid = [m for m in (current_selection or []) if m in model_slugs]
    value = valid if valid else sorted(model_slugs)

    return options, value, None  # clear status hint


@callback(
    Output("lm-interval", "disabled"),
    Output("lm-state-store", "data", allow_duplicate=True),
    Output("lm-results-store", "data", allow_duplicate=True),
    Input("lm-play-btn", "n_clicks"),
    Input("lm-pause-btn", "n_clicks"),
    Input("lm-reset-btn", "n_clicks"),
    State("lm-state-store", "data"),
    prevent_initial_call=True,
)
def toggle_playback(
    _play: int | None,
    _pause: int | None,
    _reset: int | None,
    state: dict | None,
) -> tuple[bool, dict, dict | None]:
    """Handle play/pause/reset button clicks."""
    triggered = ctx.triggered_id
    if not isinstance(state, dict):
        state = {"index": 0}

    if triggered == "lm-play-btn":
        return False, state, dash.no_update
    if triggered == "lm-pause-btn":
        return True, state, dash.no_update
    if triggered == "lm-reset-btn":
        return True, {"index": 0}, None  # clear cache on reset
    return True, state, dash.no_update


@callback(
    Output("lm-chart", "figure"),
    Output("lm-chart", "extendData"),
    Output("lm-state-store", "data"),
    Output("lm-results-store", "data"),
    Output("lm-summary", "children"),
    Output("lm-progress", "children"),
    Output("lm-latency", "children"),
    Output("lm-api-status", "children"),
    Output("lm-interval", "disabled", allow_duplicate=True),
    Input("lm-interval", "n_intervals"),
    State("lm-dataset-store", "data"),
    State("lm-state-store", "data"),
    State("lm-results-store", "data"),
    State("lm-models", "value"),
    State("lm-speed", "value"),
    prevent_initial_call=True,
)
def tick(
    _n: int,
    dataset: dict | None,
    state: dict | None,
    cache: dict | None,
    selected_models: list[str],
    speed: int,
) -> tuple:
    """Core tick: advance index and stream from cached results.

    On the first tick (or when models/data change) a single API call
    pre-computes results for the full dataset and returns a complete
    ``figure``.  Subsequent ticks return only the **delta** via
    ``extendData`` — Plotly.js appends natively with zero re-render.

    Future live sources only need to push new points into
    ``lm-dataset-store``; the ``valid_to`` watermark will trigger a
    re-compute automatically for the new data.
    """
    from api_client import get_client

    NO = dash.no_update
    state = state or {"index": 0}
    cache = cache or {}
    idx = state.get("index", 0)
    speed = speed or 5  # fallback if dropdown has no value

    def _err(msg: str, *, api: bool | None = None, pause: bool = True) -> tuple:
        """Return a safe error tuple that satisfies all outputs."""
        return (
            _empty_fig(msg),
            NO,
            state,
            NO,
            [html.Span(msg, className="text-muted-light")],
            "",
            "",
            _api_badge(api),
            pause,
        )

    # No dataset loaded
    if not dataset:
        return _err("Select a dataset and press Play")

    # Validate dataset store keys
    if not isinstance(dataset, dict):
        logger.error("lm-dataset-store is not a dict: %s", type(dataset))
        return _err("Invalid dataset — please re-select")

    total = dataset.get("total", 0)
    timestamps = dataset.get("timestamps", [])
    values = dataset.get("values", [])

    if not timestamps or not values or total <= 0:
        return _err("Dataset is empty — please select another")

    if len(timestamps) != len(values) or len(timestamps) != total:
        logger.error(
            "Dataset length mismatch: ts=%d vals=%d total=%d",
            len(timestamps), len(values), total,
        )
        return _err("Dataset corrupted — please re-select")

    # Filter out placeholder "__none__" and invalid model slugs
    selected_models = [
        m for m in (selected_models or []) if m and m != "__none__"
    ]

    # Advance index
    new_idx = min(idx + speed, total)
    if new_idx <= 0:
        return _err("Press Play to start", pause=True)

    finished = new_idx >= total
    new_state = {"index": new_idx}

    # ------------------------------------------------------------------
    # No models selected → raw-data-only playback (API may be offline)
    # ------------------------------------------------------------------
    if not selected_models:
        fig = _build_chart_no_api(timestamps[:new_idx], values[:new_idx])
        return (
            fig,
            NO,
            new_state,
            NO,
            [html.Span("No models selected — showing raw data only.", className="text-muted-light")],
            f"{new_idx:,} / {total:,}",
            "",
            _api_badge(None),
            finished,
        )

    # Build color map
    color_map = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(sorted(selected_models))}

    # ------------------------------------------------------------------
    # Cache check — compute once, then stream
    # ------------------------------------------------------------------
    models_key = "|".join(sorted(selected_models))
    cache_hit = (
        isinstance(cache, dict)
        and cache.get("models_key") == models_key
        and cache.get("valid_to", 0) >= total
        and isinstance(cache.get("chart_data"), dict)
    )

    if not cache_hit:
        # One-time inference on the full dataset
        try:
            client = get_client()
            result = client.detect_dashboard(
                timestamps,
                values,
                selected_models or None,
            )
        except Exception:
            logger.warning("API call failed during tick", exc_info=True)
            result = None

        if result is None:
            fig = _build_chart_no_api(timestamps[:new_idx], values[:new_idx])
            return (
                fig,
                NO,
                new_state,
                NO,
                [html.Span("API offline — showing raw data only", className="text-muted-light")],
                f"{new_idx:,} / {total:,}",
                "",
                _api_badge(False),
                finished,
            )

        # Validate API response structure
        chart_data = result.get("chart_data")
        if not isinstance(chart_data, dict):
            logger.error("API returned invalid chart_data: %s", type(chart_data))
            fig = _build_chart_no_api(timestamps[:new_idx], values[:new_idx])
            return (
                fig,
                NO,
                new_state,
                NO,
                [html.Span("API returned invalid data", className="text-muted-light")],
                f"{new_idx:,} / {total:,}",
                "",
                _api_badge(True),
                finished,
            )

        cache = {
            "models_key": models_key,
            "valid_to": total,
            "chart_data": chart_data,
            "latency_ms": result.get("latency_ms", 0),
        }

        # Return full figure (init) — no extendData on first frame
        try:
            fig = _build_init_figure(cache, new_idx, selected_models, color_map)
        except Exception:
            logger.error("Failed to build init figure", exc_info=True)
            fig = _build_chart_no_api(timestamps[:new_idx], values[:new_idx])

        summary = _build_progressive_summary(cache, new_idx, color_map)
        return (
            fig,
            NO,
            new_state,
            cache,
            summary,
            f"{new_idx:,} / {total:,}",
            f"Computed in {cache.get('latency_ms', 0):.0f}ms",
            _api_badge(True),
            finished,
        )

    # ------------------------------------------------------------------
    # Cache hit → stream only the delta via extendData
    # ------------------------------------------------------------------
    try:
        extend = _compute_extend_data(cache, idx, new_idx, selected_models)
    except Exception:
        logger.error("Failed to compute extend data", exc_info=True)
        extend = None

    summary = _build_progressive_summary(cache, new_idx, color_map)

    return (
        NO,
        extend,
        new_state,
        NO,
        summary,
        f"{new_idx:,} / {total:,}",
        f"Computed in {cache.get('latency_ms', 0):.0f}ms",
        _api_badge(True),
        finished,
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _build_init_figure(
    cache: dict,
    visible_to: int,
    selected_models: list[str],
    color_map: dict[str, str],
) -> go.Figure:
    """Build the initial figure with data up to *visible_to*.

    Trace order is deterministic so that ``_compute_extend_data`` can
    address traces by index:

    * 0          — value line  (row 1)
    * 1 + 2·i   — model *i* anomaly markers  (row 1)
    * 2 + 2·i   — model *i* score line  (row 2)
    """
    full_chart = cache.get("chart_data", {})
    ts = full_chart.get("timestamps", [])[:visible_to]
    vals = full_chart.get("values", [])[:visible_to]
    models_data = full_chart.get("models") or {}

    if not ts or not vals:
        return _empty_fig("No chart data available")

    win_start = max(0, len(ts) - CHART_WINDOW)
    ts_win = ts[win_start:]
    vals_win = vals[win_start:]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.6, 0.4],
        subplot_titles=["Time Series & Anomalies", "Anomaly Scores"],
    )

    # Trace 0: value line
    fig.add_trace(
        go.Scatter(
            x=ts_win,
            y=vals_win,
            mode="lines",
            name="Value",
            line={"color": "#ffffff", "width": 1},
        ),
        row=1,
        col=1,
    )

    # Per-model: fixed pair of (markers, scores) traces
    n_win = len(ts_win)
    for model_slug in sorted(selected_models):
        color = color_map.get(model_slug, "#636efa")
        mdata = models_data.get(model_slug)

        if not isinstance(mdata, dict):
            logger.warning("No data for model %s in API response", model_slug)
            # Add empty placeholder traces to keep trace indices consistent
            fig.add_trace(
                go.Scatter(x=[], y=[], mode="markers", name=model_slug,
                           marker={"color": color, "size": 6, "symbol": "x"},
                           legendgroup=model_slug),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=[], y=[], mode="lines",
                           name=f"{model_slug} (score)",
                           line={"color": color, "width": 1.5},
                           legendgroup=model_slug, showlegend=False),
                row=2, col=1,
            )
            continue

        scores = mdata.get("scores", [])[:visible_to]
        anomalies = mdata.get("anomalies", [])[:visible_to]

        s_win = scores[win_start:]
        a_win = anomalies[win_start:]

        # Pad to window length if API returned fewer points than expected
        if len(a_win) < n_win:
            a_win = a_win + [False] * (n_win - len(a_win))
        if len(s_win) < n_win:
            s_win = s_win + [0] * (n_win - len(s_win))

        # Marker trace — None for non-anomaly points keeps point count
        # aligned with the value line so maxPoints trims uniformly.
        marker_x = [t if a else None for t, a in zip(ts_win, a_win, strict=False)]
        marker_y = [v if a else None for v, a in zip(vals_win, a_win, strict=False)]

        fig.add_trace(
            go.Scatter(
                x=marker_x,
                y=marker_y,
                mode="markers",
                name=model_slug,
                marker={"color": color, "size": 6, "symbol": "x"},
                hovertemplate="%{x}<br>Value: %{y:.4g}<extra>%{fullData.name}</extra>",
                legendgroup=model_slug,
            ),
            row=1,
            col=1,
        )

        # Score line
        score_vals = [s if s is not None else 0 for s in s_win]
        fig.add_trace(
            go.Scatter(
                x=ts_win,
                y=score_vals,
                mode="lines",
                name=f"{model_slug} (score)",
                line={"color": color, "width": 1.5},
                legendgroup=model_slug,
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
        uirevision="live",
    )
    fig.update_xaxes(gridcolor="#333")
    fig.update_yaxes(gridcolor="#333")
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)

    for ann in fig.layout.annotations:
        ann.update(font_color="#aaaaaa", font_size=12)

    return fig


def _compute_extend_data(
    cache: dict,
    old_idx: int,
    new_idx: int,
    selected_models: list[str],
) -> list | None:
    """Return a Plotly ``extendData`` payload for the delta points.

    Format: ``[{x: [[...], ...], y: [[...], ...]}, indices, maxPoints]``
    """
    if new_idx <= old_idx:
        return None

    full_chart = cache.get("chart_data", {})
    ts_delta = full_chart.get("timestamps", [])[old_idx:new_idx]
    vals_delta = full_chart.get("values", [])[old_idx:new_idx]

    if not ts_delta:
        return None

    n_delta = len(ts_delta)
    models_data = full_chart.get("models") or {}
    x_arrays: list[list] = [ts_delta]  # trace 0: value line
    y_arrays: list[list] = [vals_delta]
    indices: list[int] = [0]

    for i, model_slug in enumerate(sorted(selected_models)):
        mdata = models_data.get(model_slug)
        if not isinstance(mdata, dict):
            # No data for this model — send empty/zeroed delta to keep traces aligned
            x_arrays.append([None] * n_delta)
            y_arrays.append([None] * n_delta)
            indices.append(1 + 2 * i)
            x_arrays.append(ts_delta)
            y_arrays.append([0] * n_delta)
            indices.append(2 + 2 * i)
            continue

        scores = mdata.get("scores", [])[old_idx:new_idx]
        anomalies = mdata.get("anomalies", [])[old_idx:new_idx]

        # Pad if API returned fewer points
        if len(anomalies) < n_delta:
            anomalies = anomalies + [False] * (n_delta - len(anomalies))
        if len(scores) < n_delta:
            scores = scores + [0] * (n_delta - len(scores))

        # Marker trace (1 + 2*i) — None where no anomaly
        mk_x = [t if a else None for t, a in zip(ts_delta, anomalies, strict=False)]
        mk_y = [v if a else None for v, a in zip(vals_delta, anomalies, strict=False)]
        x_arrays.append(mk_x)
        y_arrays.append(mk_y)
        indices.append(1 + 2 * i)

        # Score trace (2 + 2*i)
        x_arrays.append(ts_delta)
        y_arrays.append([s if s is not None else 0 for s in scores])
        indices.append(2 + 2 * i)

    return [{"x": x_arrays, "y": y_arrays}, indices, CHART_WINDOW]


def _build_chart_no_api(timestamps: list[str], values: list[float]) -> go.Figure:
    """Build a chart showing only raw data (API offline)."""
    n = len(timestamps)
    win_start = max(0, n - CHART_WINDOW)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps[win_start:],
            y=values[win_start:],
            mode="lines",
            name="Value",
            line={"color": "#ffffff", "width": 1},
        )
    )
    fig.update_layout(
        **_DARK_LAYOUT,
        title="API Offline — showing raw data only",
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#ffffff"),
        uirevision="live",
    )
    return fig


# ---------------------------------------------------------------------------
# Summary cards
# ---------------------------------------------------------------------------


def _build_progressive_summary(
    cache: dict,
    visible_to: int,
    color_map: dict[str, str],
) -> list:
    """Build summary cards from the visible slice of cached results."""
    try:
        models_data = cache.get("chart_data", {}).get("models") or {}
        if not models_data or visible_to <= 0:
            return [html.Span("No detection results yet.", className="text-muted-light")]

        summary: dict[str, dict] = {}
        for slug, mdata in models_data.items():
            if not isinstance(mdata, dict):
                continue
            anomalies = mdata.get("anomalies", [])[:visible_to]
            count = sum(1 for a in anomalies if a)
            ratio = count / visible_to if visible_to > 0 else 0
            summary[slug] = {
                "display_name": slug,
                "anomaly_count": count,
                "anomaly_ratio": ratio,
            }
        return _build_summary_cards(summary, color_map)
    except Exception:
        logger.error("Failed to build summary", exc_info=True)
        return [html.Span("Error computing summary.", className="text-muted-light")]


def _build_summary_cards(summary: dict, color_map: dict[str, str]) -> list:
    """Build a row of stat cards from API summary."""
    if not summary:
        return [html.Span("No detection results yet.", className="text-muted-light")]

    cards = []
    for slug, info in summary.items():
        display_name = info.get("display_name", slug)
        anomaly_count = info.get("anomaly_count", 0)
        anomaly_ratio = info.get("anomaly_ratio", 0)
        color = color_map.get(slug, "#636efa")

        card = dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H6(
                            [
                                html.Span(
                                    style={
                                        "display": "inline-block",
                                        "width": "10px",
                                        "height": "10px",
                                        "borderRadius": "50%",
                                        "backgroundColor": color,
                                        "marginRight": "8px",
                                    }
                                ),
                                display_name,
                            ],
                            className="card-title mb-2",
                        ),
                        html.Div(
                            [
                                _stat_row("Anomalies", str(anomaly_count)),
                                _stat_row("Rate", f"{anomaly_ratio * 100:.2f}%"),
                            ],
                            className="small",
                        ),
                    ]
                ),
                className="card-dark shadow rounded-3 h-100",
            ),
            md=3,
        )
        cards.append(card)

    return [dbc.Row(cards, className="g-3")]


def _stat_row(label: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Span(f"{label}: ", className="text-muted-light"),
            html.Span(value),
        ],
        className="mb-1",
    )


def _api_badge(online: bool | None) -> html.Span:
    """Return API status badge."""
    if online is None:
        return html.Span(
            [html.I(className="bi bi-circle me-1"), "Unknown"],
            className="text-muted-light",
        )
    if online:
        return html.Span(
            [html.I(className="bi bi-circle-fill me-1"), "API Online"],
            className="status-online",
        )
    return html.Span(
        [html.I(className="bi bi-circle-fill me-1"), "API Offline"],
        className="status-offline",
    )
