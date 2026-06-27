"""
Self-contained HTML dashboard served at GET /dashboard.

Kept in its own file to avoid cluttering api/main.py with a long string.
The page fetches /metrics every 30 seconds and re-renders without a full reload.
No external JS or CSS libraries — works offline and with no CDN dependency.
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agentic AI Platform — Dashboard</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #f4f6f8;
           color: #1a1a2e; padding: 28px 32px; }
    h1  { font-size: 1.35rem; font-weight: 700; }
    .sub { color: #666; font-size: 0.82rem; margin: 4px 0 28px; }
    /* Summary cards */
    .cards { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }
    .card  { background: #fff; border-radius: 10px; padding: 20px 24px;
             flex: 1; min-width: 150px;
             box-shadow: 0 1px 4px rgba(0,0,0,.07); }
    .card-label { font-size: 0.7rem; text-transform: uppercase;
                  letter-spacing: .06em; color: #888; }
    .card-value { font-size: 2.1rem; font-weight: 700; margin-top: 6px; line-height: 1; }
    .card-unit  { font-size: 0.95rem; font-weight: 400; color: #555; margin-left: 2px; }
    /* Section headers */
    .section-title { font-size: 0.9rem; font-weight: 600; margin-bottom: 10px;
                     margin-top: 4px; text-transform: uppercase; letter-spacing: .04em;
                     color: #444; }
    /* Tables */
    .tbl-wrap { background: #fff; border-radius: 10px; overflow: hidden;
                box-shadow: 0 1px 4px rgba(0,0,0,.07); margin-bottom: 24px; }
    table { width: 100%; border-collapse: collapse; }
    th { background: #f0f2f5; font-size: 0.7rem; text-transform: uppercase;
         letter-spacing: .05em; color: #555; padding: 10px 16px; text-align: left; }
    td { padding: 9px 16px; border-top: 1px solid #f0f2f5; font-size: 0.875rem; }
    tr:hover td { background: #fafafa; }
    code { background: #f0f2f5; border-radius: 4px; padding: 1px 5px;
           font-size: 0.8rem; }
    .ok  { color: #16a34a; font-weight: 600; }
    .err { color: #dc2626; font-weight: 600; }
    /* Stage latency mini-bar */
    .stage-pill { display: inline-block; background: #e0e7ff; color: #3730a3;
                  border-radius: 4px; padding: 1px 6px; font-size: 0.75rem;
                  margin: 1px; white-space: nowrap; }
    /* Footer */
    .ts { color: #aaa; font-size: 0.75rem; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>Agentic AI Platform</h1>
  <p class="sub">Live metrics &mdash; auto-refreshes every 30 s</p>

  <div class="cards" id="cards">
    <div class="card"><div class="card-label">Loading</div><div class="card-value">…</div></div>
  </div>

  <p class="section-title">Route Breakdown</p>
  <div class="tbl-wrap">
    <table id="routes-table">
      <thead><tr><th>Route</th><th>Requests</th><th>Avg ms</th><th>p95 ms</th></tr></thead>
      <tbody><tr><td colspan="4" style="color:#bbb">No data yet</td></tr></tbody>
    </table>
  </div>

  <p class="section-title">Recent Requests</p>
  <div class="tbl-wrap">
    <table id="recent-table">
      <thead><tr><th>ID</th><th>Route</th><th>Total ms</th><th>Stage breakdown</th><th>Tokens</th><th>Status</th></tr></thead>
      <tbody><tr><td colspan="6" style="color:#bbb">No requests yet</td></tr></tbody>
    </table>
  </div>

  <p class="ts" id="ts"></p>

  <script>
    function fmt(n) { return typeof n === 'number' ? n.toLocaleString() : n; }

    function stagePills(stage_ms) {
      if (!stage_ms || !Object.keys(stage_ms).length) return '<span style="color:#bbb">—</span>';
      return Object.entries(stage_ms)
        .map(([k, v]) => `<span class="stage-pill">${k}: ${v}ms</span>`)
        .join(' ');
    }

    async function refresh() {
      let data;
      try {
        data = await fetch('/metrics').then(r => r.json());
      } catch (e) {
        document.getElementById('ts').textContent = 'Error fetching metrics: ' + e;
        return;
      }

      /* --- Summary cards --- */
      const routes = data.by_route || {};
      const routeList = Object.values(routes);
      const overallAvg = routeList.length
        ? Math.round(routeList.reduce((s, r) => s + r.avg_ms, 0) / routeList.length)
        : 0;

      document.getElementById('cards').innerHTML = `
        <div class="card">
          <div class="card-label">Total Requests</div>
          <div class="card-value">${fmt(data.total_requests)}</div>
        </div>
        <div class="card">
          <div class="card-label">Error Rate</div>
          <div class="card-value">${(data.error_rate * 100).toFixed(1)}<span class="card-unit">%</span></div>
        </div>
        <div class="card">
          <div class="card-label">Avg Latency</div>
          <div class="card-value">${overallAvg}<span class="card-unit">ms</span></div>
        </div>
        <div class="card">
          <div class="card-label">Total Tokens</div>
          <div class="card-value">${fmt(data.total_tokens)}</div>
        </div>`;

      /* --- Route table --- */
      const routeRows = Object.entries(routes)
        .map(([name, r]) => `<tr>
          <td>${name}</td>
          <td>${r.count}</td>
          <td>${r.avg_ms}</td>
          <td>${r.p95_ms}</td>
        </tr>`).join('') || '<tr><td colspan="4" style="color:#bbb">No data yet</td></tr>';
      document.querySelector('#routes-table tbody').innerHTML = routeRows;

      /* --- Recent requests table --- */
      const recent = data.recent || [];
      const recentRows = recent.map(r => `<tr>
        <td><code>${r.request_id}</code></td>
        <td>${r.route}</td>
        <td>${r.total_ms}</td>
        <td>${stagePills(r.stage_ms)}</td>
        <td>${r.tokens}</td>
        <td class="${r.success ? 'ok' : 'err'}">${r.success ? 'OK' : 'ERR'}</td>
      </tr>`).join('') || '<tr><td colspan="6" style="color:#bbb">No requests yet</td></tr>';
      document.querySelector('#recent-table tbody').innerHTML = recentRows;

      document.getElementById('ts').textContent =
        'Last updated: ' + new Date().toLocaleTimeString();
    }

    refresh();
    setInterval(refresh, 30000);
  </script>
</body>
</html>"""
