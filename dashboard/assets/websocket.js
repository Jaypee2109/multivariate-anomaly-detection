/**
 * WebSocket manager for Live Monitoring page.
 *
 * Two clientside callbacks bridge the browser WebSocket API into Dash's
 * reactive callback system:
 *
 *   ws.connect  — triggered by ws-trigger store; opens/closes/controls the WS
 *   ws.drain    — triggered by dcc.Interval; moves JS-buffered chunks into
 *                 ws-buffer dcc.Store so Python callbacks can read them
 */

if (!window.dash_clientside) {
    window.dash_clientside = {};
}

/* Shared state between callbacks (lives outside Dash's store system) */
if (!window._wsState) {
    window._wsState = {
        ws: null,
        chunks: [],
        initMsg: null,
        status: "disconnected",
        error: null,
    };
}

window.dash_clientside.ws = {

    /**
     * Manage WebSocket lifecycle based on ws-trigger store updates.
     *
     * Input:  ws-trigger.data
     * Output: ws-dummy.data  (unused — Dash requires an output)
     */
    connect: function (trigger) {
        if (!trigger || typeof trigger !== "object") {
            return window.dash_clientside.no_update;
        }

        var state = window._wsState;
        var action = trigger.action;

        if (action === "connect") {
            /* Close existing connection */
            if (state.ws && state.ws.readyState <= 1) {
                state.ws.close();
            }
            state.chunks = [];
            state.initMsg = null;
            state.status = "connecting";
            state.error = null;

            var config = trigger.config || {};
            var protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            var apiHost = config.api_host || "localhost:8000";
            var url = protocol + "//" + apiHost + "/ws/stream";

            var ws = new WebSocket(url);
            state.ws = ws;

            ws.onopen = function () {
                state.status = "connected";
                ws.send(JSON.stringify({
                    dataset_path: config.dataset_path,
                    models: config.models || null,
                    chunk_size: config.chunk_size || 5,
                    interval_ms: config.interval_ms || 200,
                }));
            };

            ws.onmessage = function (event) {
                var msg = JSON.parse(event.data);

                if (msg.type === "init") {
                    state.initMsg = msg;
                    state.status = "streaming";
                } else if (msg.type === "chunk") {
                    state.chunks.push(msg);
                } else if (msg.type === "done") {
                    state.status = "done";
                } else if (msg.type === "paused") {
                    state.status = "paused";
                } else if (msg.type === "resumed") {
                    state.status = "streaming";
                } else if (msg.type === "error") {
                    state.status = "error";
                    state.error = msg.detail;
                } else if (msg.type === "reset") {
                    state.chunks = [];
                    state.status = "streaming";
                }
            };

            ws.onerror = function () {
                state.status = "error";
                state.error = "WebSocket connection failed";
            };

            ws.onclose = function () {
                if (state.status !== "done" && state.status !== "error") {
                    state.status = "disconnected";
                }
            };

        } else if (action === "disconnect") {
            if (state.ws && state.ws.readyState <= 1) {
                state.ws.close();
            }
            state.chunks = [];
            state.initMsg = null;
            state.status = "disconnected";
            state.error = null;

        } else if (action === "pause" || action === "resume" || action === "reset") {
            if (state.ws && state.ws.readyState === 1) {
                state.ws.send(JSON.stringify({ action: action }));
            }
            if (action === "reset") {
                state.chunks = [];
                state.initMsg = null;
            }

        } else if (action === "speed") {
            if (state.ws && state.ws.readyState === 1) {
                state.ws.send(JSON.stringify({
                    action: "speed",
                    chunk_size: trigger.chunk_size,
                    interval_ms: trigger.interval_ms,
                }));
            }
        }

        return window.dash_clientside.no_update;
    },

    /**
     * Drain accumulated WebSocket chunks into ws-buffer dcc.Store.
     *
     * Called on every dcc.Interval tick. Moves data from the JS-side
     * buffer (window._wsState.chunks) into the Dash store so Python
     * callbacks can consume it.
     *
     * Input:  lm-interval.n_intervals
     * State:  ws-buffer.data
     * Output: ws-buffer.data
     */
    drain: function (_n, currentBuffer) {
        var state = window._wsState;

        var newBuffer = {
            chunks: (currentBuffer && currentBuffer.chunks) ? currentBuffer.chunks : [],
            status: state.status,
            init: state.initMsg,
            error: state.error,
        };

        /* Move pending chunks from JS buffer into the store */
        if (state.chunks.length > 0) {
            newBuffer.chunks = newBuffer.chunks.concat(state.chunks);
            state.chunks = [];
        }

        return newBuffer;
    },
};
