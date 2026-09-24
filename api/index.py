"""Vercel serverless entrypoint for the mumpy project.

Exposes:
  GET /            -> landing page (docs + live demo, Arabic/English)
  GET /api/info    -> JSON with mumpy version / env info
  GET /api/demo    -> runs a small end-to-end mumpy computation and returns JSON
  GET /api/health  -> {"ok": true}

mumpy itself is a NumPy-compatible library (not a web app); this file only
gives Vercel a valid Python entrypoint so `vercel build` succeeds and the
deployment serves something useful.
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mumpy as mp
import numpy as np

LANDING_HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>mumpy — NumPy you know. Speed you feel.</title>
<style>
body{font-family:system-ui,'Segoe UI',Tahoma,sans-serif;background:#0f2027;color:#e8f1f5;margin:0;padding:2rem;line-height:1.8}
.wrap{max-width:820px;margin:auto}
.hero{background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);border:1px solid #33505c;border-radius:16px;padding:2rem}
h1{margin:.2rem 0}.tag{color:#9fd8cb}
.card{background:#122b33;border:1px solid #274b57;border-radius:12px;padding:1rem 1.2rem;margin:1rem 0}
code{direction:ltr;display:inline-block;background:#0a1a20;padding:.1rem .5rem;border-radius:6px;color:#ffd479}
pre{direction:ltr;text-align:left;background:#0a1a20;border-radius:10px;padding:1rem;overflow:auto;color:#d9f2e6}
a{color:#7dd3fc}button{background:#4f46e5;color:#fff;border:0;border-radius:8px;padding:.5rem 1.2rem;font-weight:700;cursor:pointer;margin:.2rem}
#out{white-space:pre-wrap;background:#0a1a20;border-radius:10px;padding:1rem;min-height:3rem}
.ltr{direction:ltr;text-align:left}
</style>
</head>
<body><div class="wrap">
<div class="hero">
<h1>&#128640; mumpy <span class="tag">v__VERSION__</span></h1>
<p><b>NumPy you know. Speed you feel. Tools you actually need.</b></p>
<p>&#1605;&#1603;&#1578;&#1576;&#1577; Python &#1605;&#1578;&#1608;&#1575;&#1601;&#1602;&#1577; &#1605;&#1593; NumPy: &#1581;&#1587;&#1575;&#1576; &#1605;&#1578;&#1608;&#1575;&#1586;&#1610; + &#1593;&#1605;&#1604;&#1610;&#1575;&#1578; &#1605;&#1583;&#1605;&#1580;&#1577; + &#1571;&#1583;&#1608;&#1575;&#1578; (DB, DataFrame, ML, DL) &#1576;&#1583;&#1608;&#1606; &#1575;&#1593;&#1578;&#1605;&#1575;&#1583;&#1610;&#1575;&#1578; &#1573;&#1580;&#1576;&#1575;&#1585;&#1610;&#1577; &#1594;&#1610;&#1585; NumPy.</p>
<p class="ltr">pip install mumpy-toolkit &nbsp;|&nbsp; <a href="https://github.com/salim-studio/mumpy">GitHub</a></p>
</div>
<div class="card">
<h3>&#129514; &#1580;&#1585;&#1617;&#1576; &#1581;&#1610; &#1605;&#1576;&#1575;&#1588;&#1585; (&#1610;&#1593;&#1605;&#1604; &#1593;&#1604;&#1609; &#1607;&#1584;&#1575; &#1575;&#1604;&#1587;&#1610;&#1585;&#1601;&#1585;)</h3>
<button onclick="call('/api/info')">/api/info</button>
<button onclick="call('/api/demo')">/api/demo — &#1581;&#1587;&#1575;&#1576; &#1581;&#1602;&#1610;&#1602;&#1610;</button>
<button onclick="call('/api/health')">/api/health</button>
<div id="out">...</div>
</div>
<div class="card"><h3>Quickstart</h3>
<pre>import mumpy as mp
a = mp.arange(10_000_000)
b = mp.sqrt(a)          # multithreaded over all cores
c = mp.fma(a, b, 1.0)   # a*b+c in a single pass</pre>
</div>
<div class="card ltr">
<h3>Live API</h3>
<p><code>GET /api/info</code> — version &amp; environment<br/>
<code>GET /api/demo?n=1000</code> — fused ops + stats + tiny LogisticRegression<br/>
<code>GET /api/health</code> — liveness check</p>
</div>
</div>
<script>
async function call(u){const o=document.getElementById('out');o.textContent='...';try{const r=await fetch(u);o.textContent=await r.text();}catch(e){o.textContent='Error: '+e;}}
</script>
</body></html>
"""


def _demo(n=1000):
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    a = mp.arange(n, dtype=float)
    b = mp.sqrt(a + 1)
    c = mp.fma(a, b, 1.0)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    Xtr, Xte, ytr, yte = mp.ml.train_test_split(X, y, test_size=0.25, random_state=0)
    model = mp.ml.LogisticRegression(lr=0.5, epochs=200).fit(Xtr, ytr)
    acc = float(mp.metrics.accuracy(yte, model.predict(Xte)))
    return {
        "n": n,
        "fma_head": [float(v) for v in np.asarray(c[:5])],
        "fma_mean": float(np.asarray(c).mean()),
        "ml_accuracy": round(acc, 4),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


class handler(BaseHTTPRequestHandler):
    server_version = "mumpy-vercel/1.0"

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj), "application/json")

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path in ("/", "/api", "/api/index"):
                html = LANDING_HTML.replace("__VERSION__", getattr(mp, "__version__", "?"))
                self._send(200, html, "text/html; charset=utf-8")
            elif path == "/api/health":
                self._json(200, {"ok": True})
            elif path == "/api/info":
                self._json(200, mp.info())
            elif path == "/api/demo":
                qs = parse_qs(parsed.query)
                try:
                    n = max(10, min(20000, int(qs.get("n", ["1000"])[0])))
                except ValueError:
                    n = 1000
                self._json(200, _demo(n))
            else:
                self._json(404, {"error": "not found", "hint": "try /, /api/info, /api/demo, /api/health"})
        except Exception as exc:  # never leak a 500 without a body
            self._json(500, {"error": type(exc).__name__, "message": str(exc)[:300]})

    def log_message(self, *args):  # quieter logs on Vercel
        pass
