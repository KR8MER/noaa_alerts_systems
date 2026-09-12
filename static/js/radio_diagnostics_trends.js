/*!
 * EAS Station — SDR Diagnostics "Historical Trends" charts + sparkline strip.
 *
 * Two independent pieces, both reading the same endpoint:
 *   GET /api/radio/diagnostics/trends/<receiver_id>?window=1h|6h|24h|7d
 * (2-tier Redis archive: raw 10 s samples -> 1h, 5 min rollups -> 7d; see
 * app_core/radio/trends.py, sampled by sdr_hardware_service.py every 10 s.)
 *
 * 1. An "at a glance" sparkline strip embedded in every receiver card
 *    (always on, no toggle) — a compact canvas-drawn Signal Strength trace
 *    and a Buffer Drops (overflow+underflow) trace, refreshed every 30 s.
 *    Adapted from the GPS dashboard's drawSparkline() helper, simplified
 *    to take a plain array of numbers (the server already returns a
 *    finished time series) rather than porting its live client-side ring
 *    buffer machinery, which has no equivalent here.
 *
 * 2. A per-receiver "Historical Trends" panel, toggled by the button next
 *    to Live Waterfall / Spectrum Scope: four Chart.js line charts (signal
 *    strength, lock %, sample-rate ratio, buffer overflow/underflow) with
 *    a window selector. Mirrors this page's existing RBDS history chart
 *    (radio_diagnostics.html) in using plain string labels rather than a
 *    Chart.js time scale -- this page does not load the date-fns adapter
 *    GPS's own trends chart file relies on.
 *
 * Deliberately self-contained and independent of the page's inline
 * <script> block (live waterfall / spectrum scope / run-diagnostic
 * handlers) -- no shared state, so this file can be dropped or reloaded
 * without touching that code.
 */
(function () {
    'use strict';

    var TRENDS_ENDPOINT = '/api/radio/diagnostics/trends/';

    function num(v) {
        return (typeof v === 'number' && isFinite(v)) ? v : null;
    }

    // ── Small theme-aware helpers ──────────────────────────────────────
    function cssVar(name, fallback) {
        try {
            var v = getComputedStyle(document.documentElement).getPropertyValue(name);
            return (v && v.trim()) ? v.trim() : fallback;
        } catch (e) {
            return fallback;
        }
    }

    // ── Sparkline canvas (DPR-aware; adapted from gps_dashboard.html) ──
    function fitCanvas(canvas) {
        var dpr = window.devicePixelRatio || 1;
        var rect = canvas.getBoundingClientRect();
        var w = Math.max(1, Math.round(rect.width));
        var h = Math.max(1, Math.round(rect.height));
        if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
            canvas.width = w * dpr;
            canvas.height = h * dpr;
        }
        var ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        return { ctx: ctx, w: w, h: h };
    }

    // `series`: array of { values: number[], color, dashed }. `opts`:
    // { includeZero, yDomain: [lo, hi] }. Values may contain null/NaN to
    // indicate a gap (line breaks there, same convention as the waterfall
    // status line / RBDS chart on this page).
    function drawSparkline(canvas, series, opts) {
        if (!canvas) return;
        var fit = fitCanvas(canvas);
        var ctx = fit.ctx, W = fit.w, H = fit.h;
        ctx.clearRect(0, 0, W, H);

        opts = opts || {};
        var pad = { l: 2, r: 2, t: 3, b: 3 };
        var plotW = W - pad.l - pad.r;
        var plotH = H - pad.t - pad.b;

        var lo = Infinity, hi = -Infinity, anyData = false, n = 0;
        series.forEach(function (s) {
            (s.values || []).forEach(function (v) {
                if (v === null || v === undefined || isNaN(v)) return;
                if (v < lo) lo = v;
                if (v > hi) hi = v;
                anyData = true;
            });
            if ((s.values || []).length > n) n = s.values.length;
        });
        if (!anyData) return;

        if (opts.yDomain) { lo = opts.yDomain[0]; hi = opts.yDomain[1]; }
        if (opts.includeZero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
        if (lo === hi) { lo -= 1; hi += 1; }
        var span = hi - lo;
        lo -= span * 0.08; hi += span * 0.08;

        series.forEach(function (s) {
            var vals = s.values || [];
            var len = vals.length;
            if (len < 1) return;
            ctx.save();
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 1.4;
            if (s.dashed) ctx.setLineDash([3, 2]);
            ctx.beginPath();
            var started = false;
            for (var i = 0; i < len; i++) {
                var v = vals[i];
                var x = pad.l + (len === 1 ? plotW / 2 : (i / (len - 1)) * plotW);
                if (v === null || v === undefined || isNaN(v)) { started = false; continue; }
                var y = pad.t + plotH * (1 - (v - lo) / (hi - lo));
                if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
            }
            ctx.stroke();
            ctx.setLineDash([]);

            // Dot on the latest sample.
            for (var j = len - 1; j >= 0; j--) {
                var lv = vals[j];
                if (lv === null || lv === undefined || isNaN(lv)) continue;
                var lx = pad.l + (len === 1 ? plotW / 2 : (j / (len - 1)) * plotW);
                var ly = pad.t + plotH * (1 - (lv - lo) / (hi - lo));
                ctx.fillStyle = s.color;
                ctx.beginPath();
                ctx.arc(lx, ly, 1.8, 0, Math.PI * 2);
                ctx.fill();
                break;
            }
            ctx.restore();
        });
    }

    // ── At-a-glance sparkline strip ─────────────────────────────────────
    var SPARKLINE_REFRESH_MS = 30000;
    var SPARKLINE_WINDOW = '1h';

    function sparklineOverflowUnderflow(sample) {
        var of = num(sample.overflow_count) || 0;
        var uf = num(sample.underflow_count) || 0;
        return of + uf;
    }

    function loadSparklinesForReceiver(receiverId) {
        var sigCanvas = document.querySelector(
            'canvas[data-sparkline-canvas="signal"][data-receiver-id="' + CSS.escape(receiverId) + '"]'
        );
        var dropCanvas = document.querySelector(
            'canvas[data-sparkline-canvas="drops"][data-receiver-id="' + CSS.escape(receiverId) + '"]'
        );
        if (!sigCanvas && !dropCanvas) return;

        fetch(TRENDS_ENDPOINT + encodeURIComponent(receiverId) + '?window=' + SPARKLINE_WINDOW, { cache: 'no-store' })
            .then(function (resp) { return resp.ok ? resp.json() : null; })
            .then(function (data) {
                if (!data) return;
                var samples = Array.isArray(data.samples) ? data.samples : [];
                if (sigCanvas) {
                    drawSparkline(sigCanvas, [
                        { values: samples.map(function (s) { return num(s.signal_strength); }), color: cssVar('--info-color', '#39a0ff') },
                    ], {});
                }
                if (dropCanvas) {
                    drawSparkline(dropCanvas, [
                        { values: samples.map(sparklineOverflowUnderflow), color: cssVar('--danger-color', '#ff6b6b') },
                    ], { includeZero: true });
                }
            })
            .catch(function () { /* sparklines are ambient decoration, not critical UI */ });
    }

    function loadAllSparklines() {
        var ids = {};
        document.querySelectorAll('canvas[data-sparkline-canvas][data-receiver-id]').forEach(function (el) {
            ids[el.getAttribute('data-receiver-id')] = true;
        });
        Object.keys(ids).forEach(loadSparklinesForReceiver);
    }

    document.addEventListener('DOMContentLoaded', function () {
        loadAllSparklines();
        window.setInterval(function () {
            if (document.visibilityState === 'visible') loadAllSparklines();
        }, SPARKLINE_REFRESH_MS);
    });
    document.addEventListener('visibilitychange', function () {
        if (document.visibilityState === 'visible') loadAllSparklines();
    });

    // ── Historical Trends panel (Chart.js) ──────────────────────────────
    var TRENDS_WINDOWS = ['1h', '6h', '24h', '7d'];
    var trendsCharts = {};              // "receiverId:metric" -> Chart instance
    var trendsActiveReceivers = {};     // receiverId -> true
    var trendsCurrentWindow = {};       // receiverId -> window string
    var trendsRequestGen = {};          // receiverId -> generation counter, guards against
                                         // a slow fetch resolving after a newer rebuild/window
                                         // change and painting stale samples over fresh ones

    function trendsFindContainer(receiverId) {
        return document.querySelector('.trends-container[data-trends-for="' + CSS.escape(receiverId) + '"]');
    }
    function trendsFindLabel(receiverId) {
        return document.querySelector('span[data-trends-label="' + CSS.escape(receiverId) + '"]');
    }

    function trendsDestroyCharts(receiverId) {
        Object.keys(trendsCharts).forEach(function (key) {
            if (key.indexOf(receiverId + ':') === 0) {
                try { trendsCharts[key].destroy(); } catch (e) { /* noop */ }
                delete trendsCharts[key];
            }
        });
    }

    function trendsBuildPanel(container, receiverId) {
        var win = trendsCurrentWindow[receiverId] || '1h';
        var options = TRENDS_WINDOWS.map(function (w) {
            return '<option value="' + w + '"' + (w === win ? ' selected' : '') + '>' + w + '</option>';
        }).join('');
        container.innerHTML =
            '<div class="d-flex flex-wrap align-items-center gap-2 mb-2">' +
                '<span class="badge bg-secondary text-uppercase">Trends</span>' +
                '<label class="small text-muted mb-0 ms-2" for="trendsWindow-' + receiverId + '">Window</label>' +
                '<select class="form-select form-select-sm w-auto" id="trendsWindow-' + receiverId + '" data-trends-window="' + receiverId + '">' +
                    options +
                '</select>' +
                '<span class="small text-muted" data-trends-status="' + receiverId + '">Loading…</span>' +
            '</div>' +
            '<div class="row g-2">' +
                '<div class="col-md-6"><div class="small text-muted mb-1">Signal Strength</div>' +
                    '<div style="position:relative;height:220px;"><canvas data-trends-chart="signal" data-receiver-id="' + receiverId + '"></canvas></div></div>' +
                '<div class="col-md-6"><div class="small text-muted mb-1">Lock %</div>' +
                    '<div style="position:relative;height:220px;"><canvas data-trends-chart="locked" data-receiver-id="' + receiverId + '"></canvas></div></div>' +
                '<div class="col-md-6"><div class="small text-muted mb-1">Sample Rate Ratio (effective / configured)</div>' +
                    '<div style="position:relative;height:220px;"><canvas data-trends-chart="rate" data-receiver-id="' + receiverId + '"></canvas></div></div>' +
                '<div class="col-md-6"><div class="small text-muted mb-1">Buffer Overflow / Underflow</div>' +
                    '<div style="position:relative;height:220px;"><canvas data-trends-chart="buffer" data-receiver-id="' + receiverId + '"></canvas></div></div>' +
            '</div>';
        container.style.display = '';
    }

    function trendsLockedPct(sample) {
        if (typeof sample.locked_pct === 'number') return sample.locked_pct;
        if (typeof sample.locked === 'number') return sample.locked * 100;
        return null;
    }

    function trendsRenderChart(receiverId, metric, labels, datasets, scaleOpts) {
        if (!window.Chart) return;
        var canvas = document.querySelector(
            'canvas[data-trends-chart="' + metric + '"][data-receiver-id="' + CSS.escape(receiverId) + '"]'
        );
        if (!canvas) return;
        var key = receiverId + ':' + metric;
        if (trendsCharts[key]) {
            try { trendsCharts[key].destroy(); } catch (e) { /* noop */ }
            delete trendsCharts[key];
        }
        var text = cssVar('--text-color', '#e6edf3');
        var grid = 'rgba(127,127,127,0.14)';
        trendsCharts[key] = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets.map(function (ds) {
                    return {
                        label: ds.label,
                        data: ds.data,
                        borderColor: ds.color,
                        backgroundColor: ds.color + '26',
                        pointRadius: 0,
                        borderWidth: 1.5,
                        tension: 0.25,
                        spanGaps: true,
                    };
                }),
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: { ticks: { maxTicksLimit: 6, autoSkip: true, color: text }, grid: { color: grid } },
                    y: Object.assign({ beginAtZero: false, ticks: { color: text }, grid: { color: grid } }, scaleOpts || {}),
                },
                plugins: { legend: { display: datasets.length > 1, labels: { boxWidth: 10, color: text } } },
            },
        });
    }

    function trendsFetchAndRender(receiverId) {
        var container = trendsFindContainer(receiverId);
        if (!container) return;
        var win = trendsCurrentWindow[receiverId] || '1h';
        var statusEl = container.querySelector('[data-trends-status="' + receiverId + '"]');
        if (statusEl) statusEl.textContent = 'Loading…';

        var myGen = (trendsRequestGen[receiverId] || 0) + 1;
        trendsRequestGen[receiverId] = myGen;
        var isCurrent = function () { return trendsRequestGen[receiverId] === myGen; };

        fetch(TRENDS_ENDPOINT + encodeURIComponent(receiverId) + '?window=' + encodeURIComponent(win), { cache: 'no-store' })
            .then(function (resp) {
                return resp.json().then(function (data) { return { ok: resp.ok, status: resp.status, data: data }; });
            })
            .then(function (result) {
                if (!isCurrent()) return; // superseded by a newer rebuild/window change
                if (!result.ok) {
                    if (statusEl) statusEl.textContent = 'Error: ' + (result.data.error || ('HTTP ' + result.status));
                    return;
                }
                var data = result.data;
                var samples = Array.isArray(data.samples) ? data.samples : [];
                if (statusEl) {
                    statusEl.textContent = samples.length
                        ? (samples.length + ' samples (' + data.tier + ' tier)')
                        : 'No trend data yet for this window';
                }
                var labels = samples.map(function (s) {
                    var d = s.t ? new Date(s.t) : null;
                    return d ? d.toLocaleTimeString() : '';
                });
                trendsRenderChart(receiverId, 'signal', labels, [
                    { label: 'Signal Strength', data: samples.map(function (s) { return num(s.signal_strength); }), color: '#39a0ff' },
                ]);
                trendsRenderChart(receiverId, 'locked', labels, [
                    { label: 'Lock %', data: samples.map(trendsLockedPct), color: '#39ff8f' },
                    // max: 105, not 100/suggestedMax: a receiver locked 100% of the
                    // window (zero variance -- the common, healthy case) gives Chart.js
                    // no data range to round up from, so suggestedMax does nothing when
                    // data already sits exactly on it -- confirmed live (yMax stayed 100
                    // after switching to suggestedMax). A fixed ceiling above the true
                    // 0-100 range is the only fix that holds regardless of variance.
                ], { min: 0, max: 105 });
                trendsRenderChart(receiverId, 'rate', labels, [
                    { label: 'Sample Rate Ratio', data: samples.map(function (s) { return num(s.sample_rate_ratio); }), color: '#f0ad4e' },
                    // effective/configured is always in [0, 1] by construction (decimation
                    // never produces more samples than were captured) -- without this,
                    // Chart.js auto-scaling a constant series (the common case: a fixed
                    // decimation factor never changes) picks an arbitrary, physically
                    // meaningless range, observed live as -1 to 1.5 for a flat 0.25.
                ], { min: 0, suggestedMax: 1 });
                trendsRenderChart(receiverId, 'buffer', labels, [
                    { label: 'Overflow', data: samples.map(function (s) { return num(s.overflow_count); }), color: '#d9534f' },
                    { label: 'Underflow', data: samples.map(function (s) { return num(s.underflow_count); }), color: '#9b59b6' },
                ], { min: 0 });
            })
            .catch(function (err) {
                if (!isCurrent()) return; // superseded by a newer rebuild/window change
                if (statusEl) statusEl.textContent = 'Fetch error: ' + err.message;
            });
    }

    function trendsStart(receiverId) {
        var container = trendsFindContainer(receiverId);
        if (!container) return;
        trendsActiveReceivers[receiverId] = true;
        trendsBuildPanel(container, receiverId);
        trendsFetchAndRender(receiverId);
        var label = trendsFindLabel(receiverId);
        if (label) label.textContent = 'Hide Historical Trends';
    }

    function trendsStop(receiverId) {
        delete trendsActiveReceivers[receiverId];
        trendsDestroyCharts(receiverId);
        var container = trendsFindContainer(receiverId);
        if (container) {
            container.style.display = 'none';
            container.innerHTML = '';
        }
        var label = trendsFindLabel(receiverId);
        if (label) label.textContent = 'Historical Trends';
    }

    // -- Survive the receiver cards' WebSocket-driven auto-refresh --------
    // `renderReceivers()` in radio_diagnostics.html replaces the entire
    // receiver-card tree (roughly once a second) with freshly-rendered,
    // empty `.trends-container` placeholders -- server markup has no notion
    // of "this panel was open". A Chart.js canvas's drawn pixels are never
    // serialized into HTML, so any attempt to clone-and-restore the old
    // panel markup across that replacement only ever produces a blank,
    // disconnected canvas frozen at whatever width it happened to have at
    // capture time (the "chart disappears almost immediately" / "isn't wide
    // enough" symptoms). Instead, watch the receivers container for its
    // children being replaced and, for any receiver whose trends panel was
    // left open, rebuild it from scratch against the fresh DOM.
    function trendsRebuildActive() {
        Object.keys(trendsActiveReceivers).forEach(function (receiverId) {
            var container = trendsFindContainer(receiverId);
            if (!container) {
                // The receiver itself is gone (removed/renamed) -- drop its
                // state instead of leaving a dangling active flag and
                // detached Chart.js instances behind.
                trendsDestroyCharts(receiverId);
                delete trendsActiveReceivers[receiverId];
                delete trendsRequestGen[receiverId];
                return;
            }
            if (container.firstChild) return; // not yet clobbered by this refresh
            trendsBuildPanel(container, receiverId);
            trendsFetchAndRender(receiverId);
            var label = trendsFindLabel(receiverId);
            if (label) label.textContent = 'Hide Historical Trends';
        });
    }

    document.addEventListener('DOMContentLoaded', function () {
        var receiversContainer = document.getElementById('receiversContainer');
        if (receiversContainer && window.MutationObserver) {
            new MutationObserver(trendsRebuildActive).observe(receiversContainer, { childList: true });
        }
    });

    document.addEventListener('click', function (event) {
        var btn = event.target.closest('button[data-action="historical-trends"]');
        if (!btn) return;
        var receiverId = btn.getAttribute('data-receiver-id');
        if (!receiverId) return;
        if (trendsActiveReceivers[receiverId]) {
            trendsStop(receiverId);
        } else {
            trendsStart(receiverId);
        }
    });

    // Delegated so it keeps working after trendsRebuildActive() rebuilds the
    // panel's <select> as a brand-new element every refresh cycle -- a
    // directly-attached listener would not survive that, a document-level
    // delegated one does.
    document.addEventListener('change', function (event) {
        var select = event.target.closest('select[data-trends-window]');
        if (!select) return;
        var receiverId = select.getAttribute('data-trends-window');
        if (!receiverId) return;
        trendsCurrentWindow[receiverId] = select.value;
        trendsFetchAndRender(receiverId);
    });

    window.addEventListener('beforeunload', function () {
        Object.keys(trendsActiveReceivers).forEach(trendsDestroyCharts);
    });
})();
