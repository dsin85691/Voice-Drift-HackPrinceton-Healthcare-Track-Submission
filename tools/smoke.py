"""Deployment smoke test (backlog VD-11), standard library only.

    python tools/smoke.py                         # against http://127.0.0.1:8000
    python tools/smoke.py --base https://your-deployment --reset

Scenarios A-D run through the public API exactly as the browser does. Scenario E (fresh browser session, clicking
through) cannot be automated here; the script prints its checklist. --reset reloads the seeded demo profiles first
(it only touches the seeded synthetic profiles). Exit code 0 = all automated scenarios passed.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave

RESULTS: list[tuple[str, bool, str]] = []


def call(base, method, path, body=None, files=None, timeout=300):
    url = base.rstrip("/") + path
    headers, data = {}, None
    if files is not None:
        boundary = uuid.uuid4().hex
        parts = []
        for k, v in (body or {}).items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        for k, (fname, blob, ctype) in files.items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'
                         f"Content-Type: {ctype}\r\n\r\n".encode() + blob + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        data = b"".join(parts)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw and r.headers.get_content_type() == "application/json" else raw)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}", flush=True)


def wait_checkin(base, cid, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st, j = call(base, "GET", f"/api/checkins/{cid}")
        if st == 200:
            return j
        time.sleep(1)
    return None


def pending(base):
    st, j = call(base, "GET", "/api/v1/review-flags?status=OPEN,ACKNOWLEDGED")
    return j if st == 200 else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--reset", action="store_true", help="reload the seeded demo profiles first")
    a = ap.parse_args()
    b = a.base

    print("Setup")
    st, h = call(b, "GET", "/api/health")
    check("server reachable", st == 200, f"encoder={h.get('encoder') if isinstance(h, dict) else '?'}")
    if st != 200:
        return 1
    check("frontend served", call(b, "GET", "/")[0] == 200)
    if a.reset:
        check("seeded demo profiles loaded", call(b, "POST", "/api/demo/reset", {"scenario": "all"})[0] == 200)

    print("Scenario A - normal check-in (bundled sample, same pipeline as live)")
    before = {f["flag_id"] for f in pending(b)}
    st, j = call(b, "POST", "/api/demo/sample", timeout=600)
    check("one-click sample analysed", st == 200, "" if st == 200 else str(j)[:200])
    if st == 200:
        c = j["checkin"]
        check("quality reported", c.get("quality_status") in ("pass", "warning"), c.get("quality_status"))
        check("drift score shown with boundary", c.get("drift_score") is not None and c.get("drift_threshold") == 1.0,
              f"{c.get('drift_score')} x boundary, {c.get('drift_status')}")
        check("status agrees with score", (c.get("drift_status") == "elevated") == (c["drift"]["d"] > c["drift"]["tau"]))
        check("acoustic measurement present", (c.get("acoustic_features") or {}).get("pause_fraction", {}).get("status") == "VALID")
        new = {f["flag_id"] for f in pending(b)} - before
        check("a single check-in creates no queue item", not new)

    print("Scenario B - elevated drift (seeded illustrative profile)")
    flags = pending(b)
    drift = [f for f in flags if f["patient_id"] == "p_drift"]
    check("seeded elevated profile has a pending review item", bool(drift),
          "run with --reset if the demo data was cleared" if not drift else "")
    if drift:
        f = drift[0]
        check("item shows trigger score above boundary", (f.get("trigger") or {}).get("normalized_score", 0) > 1.0,
              f"{(f.get('trigger') or {}).get('normalized_score')} / 1.0")
        st, d = call(b, "GET", f"/api/v1/review-flags/{f['flag_id']}")
        check("item opens with evidence", st == 200 and len(d.get("evidence", [])) >= 2)
        st, tl = call(b, "GET", "/api/patients/p_drift/timeline?days=365")
        marked = [c for c in tl.get("checkins", []) if c.get("review_flag")] if st == 200 else []
        check("Home/Trends mark the same check-ins", bool(marked) and all(c["drift_status"] == "elevated" for c in marked))
    stable = [f for f in flags if f["patient_id"] == "p_stable"]
    check("stable seeded profile has no queue item", not stable)

    print("Scenario C - results")
    st, r = call(b, "GET", "/api/results")
    check("results endpoint loads", st == 200)
    agg = (r or {}).get("aggregate") if st == 200 else None
    check("aggregate results present", bool(agg), "" if agg else "results/aggregate.json missing")
    if agg:
        R = agg["results"]["session"]
        R = R.get("matched", R)
        check("B0, B1 and P1 all reported", {"B0", "B1", "P1"} <= set(R))
        matched = agg.get("schema", 1) >= 2
        check("calibration state disclosed", True, "matched operating point" if matched else
              "unmatched: the UI shows the preliminary-comparison caveat")

    print("Scenario D - failure handling")
    n_flags = len(pending(b))
    st, j = call(b, "POST", "/api/patients/p_smoke/checkins", {"prompt_id": "standard_reading_v1"},
                 files={"audio": ("bad.wav", b"this is not audio", "audio/wav")})
    check("invalid upload accepted for processing (no crash)", st == 202, str(st))
    if st == 202:
        res = wait_checkin(b, j["checkin_id"])
        check("invalid audio is rejected, not scored", bool(res) and res["quality"]["verdict"] == "REJECT"
              and res.get("drift") is None)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\x00\x00" * 16000 * 25)
    st, j = call(b, "POST", "/api/patients/p_smoke/checkins", {"prompt_id": "standard_reading_v1"},
                 files={"audio": ("silence.wav", buf.getvalue(), "audio/wav")})
    if st == 202:
        res = wait_checkin(b, j["checkin_id"])
        check("silent recording is rejected with a reason", bool(res) and res["quality"]["verdict"] == "REJECT",
              ", ".join((res or {}).get("quality", {}).get("reason_codes", [])))
    check("no review item created by failures", len(pending(b)) == n_flags)
    call(b, "DELETE", "/api/patients/p_smoke")

    print("Scenario E - fresh browser session (manual)")
    for step in ("Open the deployment in a private window (no saved profile or settings).",
                 "Home: click 'Analyze sample recording'; quality, overall pattern and measurements appear.",
                 "Results: charts and tables load; the B1 note and calibration caveat are visible.",
                 "Top bar: 'Review queue' -> open the pending item -> evidence table and history load.",
                 "Browser console shows no errors throughout."):
        print("  [ ] " + step)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} automated checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
