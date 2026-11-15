"""Measure and cache both one-click demos on the RUNNING server, before a presentation (standard library only).

    python tools/warm_demo.py                         # server at http://127.0.0.1:8000
    python tools/warm_demo.py --base http://host:port

The server also does this by itself in the background at start-up (demo.warm_on_start). Use this to be sure it has
finished, or after changing the model/artifacts. Afterwards both demo buttons answer instantly from the measured
snapshot; the page labels when and with which model each was measured.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def call(base: str, method: str, path: str, timeout: float = 900):
    req = urllib.request.Request(base.rstrip("/") + path, method=method, data=b"" if method == "POST" else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    a = ap.parse_args()
    t0 = time.time()
    print("measuring the demos on", a.base, "(the first time can take a minute or two)...", flush=True)
    while True:
        st, j = call(a.base, "POST", "/api/demo/warm")
        if st == 409 or (st == 200 and j.get("state") == "running"):
            time.sleep(3); continue                    # the start-up warm-up is still running: wait for it
        break
    if st != 200:
        print("FAILED:", st, j.get("detail", j)); return 1
    st, r = call(a.base, "GET", "/api/results", timeout=60)
    s, d = r.get("sample_snapshot") or {}, r.get("demo_snapshot") or {}
    print(f"done in {time.time() - t0:.0f} s")
    print(f"  sample analysis : {'cached' if s else 'MISSING'}  measured {s.get('ran_at', '-')}  "
          f"({s.get('compute_seconds', '?')} s, {s.get('model', '?')})")
    print(f"  noise demo      : {'cached' if d else 'MISSING'}  measured {d.get('ran_at', '-')}  "
          f"({d.get('compute_seconds', '?')} s, {d.get('model', '?')})")
    for kind, path in (("sample", "/api/demo/sample"), ("noise", "/api/demo/run")):
        t = time.time()
        st, _ = call(a.base, "POST", path, timeout=60)
        print(f"  {kind:6s} button now answers in {1000 * (time.time() - t):.0f} ms (HTTP {st})")
    return 0 if s and d else 1


if __name__ == "__main__":
    sys.exit(main())
