"""Browser checks for page navigation, the recorder and analysis progress, run against a RUNNING server.

    python tools/browser_checks.py        # needs Playwright + Chrome; uses a fake microphone, uploads nothing real

Each check forces one failure (slow request, dropped event stream, recorder refusing to start, rejected action,
bad address, edge-case values) and confirms the page handles it. It first reloads the synthetic demo profiles
("Load demo: all", which keeps live recordings); every other change is made by request interception, so the
database is otherwise unchanged."""
import json
import re
import urllib.request

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8000/"
ARGS = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"]
results = []


def ok(name, cond, info=""):
    results.append((name, bool(cond), info))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""))


def ctx_for(b, role="patient", pid="p_demo", fast=False, extra_js=""):
    c = b.new_context(viewport={"width": 1280, "height": 900})
    c.grant_permissions(["microphone"], origin=BASE.rstrip("/"))
    js = f"localStorage.setItem('vd_role','{role}');localStorage.setItem('vd_pid','{pid}');"
    js += """
      window.__streams = [];
      const gum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
      navigator.mediaDevices.getUserMedia = async (c) => {
        if (window.__gumDelay) await new Promise(r => setTimeout(r, window.__gumDelay));
        const s = await gum(c); window.__streams.push(s); return s; };
      const of = window.fetch.bind(window);   // in-page delay: slows one request without blocking the test driver
      window.fetch = async (u, o) => { const d = window.__delay || {};
        for (const k in d) if (String(u).includes(k)) await new Promise(r => setTimeout(r, d[k]));
        return of(u, o); };
    """
    if fast:   # make the recorder's clock run 20x so Stop unlocks quickly
        js += "const pn = performance.now.bind(performance); performance.now = () => pn() * 20;"
    c.add_init_script(js + extra_js)
    return c


def api(path, body=None):
    req = urllib.request.Request(BASE + "api" + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def patch_timeline(pg, pid, fn):
    """Serve the real timeline for `pid` with an edit applied (edge-case values without touching the database)."""
    def handler(r):
        resp = r.fetch(); j = resp.json(); fn(j)
        r.fulfill(response=resp, body=json.dumps(j))
    pg.route(f"**/api/patients/{pid}/timeline*", handler)


api("/demo/reset", {"scenario": "all"})

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", args=ARGS)
    errs = []

    # 1. route race: slow first page must not overwrite the second
    c = ctx_for(b); pg = c.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(BASE + "#/about"); pg.wait_for_selector(".ribbon")
    pg.evaluate("window.__delay = {'/timeline': 2500}; location.hash = '#/trends'")
    pg.wait_for_timeout(300)
    pg.evaluate("window.__delay = {}; location.hash = '#/about'")
    pg.wait_for_timeout(4000)
    h1 = pg.inner_text("#app h1")
    ok("1 route race: later page stays", "About" in h1 or "Privacy" in h1, h1)
    ok("1 route race: one ribbon", pg.locator("#app .ribbon").count() == 1, str(pg.locator("#app .ribbon").count()))
    c.close()

    # 2. clock never shows 0:00 at the start
    c = ctx_for(b); pg = c.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(BASE + "#/checkin"); pg.wait_for_selector("#rec")
    pg.click("#rec")
    seen = set()
    for _ in range(12):
        seen.add(pg.inner_text("#clock")); pg.wait_for_timeout(150)
    ok("2 clock: no 0:00 at start", "0:00" not in seen, ",".join(sorted(seen)))
    # 8. cancel goes through route(): ribbon back, mic released, nothing uploaded
    uploads = []
    pg.on("request", lambda r: uploads.append(r.url) if r.method == "POST" and "/checkins" in r.url else None)
    pg.click("#cancel"); pg.wait_for_timeout(800)
    ok("8 cancel: ribbon present", pg.locator("#app .ribbon").count() == 1)
    ok("8 cancel: mic released", pg.evaluate("window.__streams.every(s => s.getTracks().every(t => t.readyState === 'ended'))"))
    ok("8 cancel: no upload", not uploads)
    c.close()

    # 4. leave during the permission prompt: mic released immediately, no recording
    c = ctx_for(b); pg = c.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(BASE + "#/checkin"); pg.wait_for_selector("#rec")
    pg.evaluate("window.__gumDelay = 1500")
    pg.click("#rec"); pg.wait_for_timeout(200)
    pg.evaluate("location.hash = '#/'"); pg.wait_for_timeout(2500)
    st = pg.evaluate("window.__streams.map(s => s.getTracks().map(t => t.readyState).join())")
    ok("4 permission then leave: mic released", st and all(x == "ended" for x in st), str(st))
    c.close()

    # 5. recorder start failure is reported, mic released
    c = ctx_for(b, extra_js="window.MediaRecorder = class { static isTypeSupported(){return true} constructor(){ throw new Error('NotSupportedError: test') } };")
    pg = c.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(BASE + "#/checkin"); pg.wait_for_selector("#rec")
    pg.click("#rec"); pg.wait_for_timeout(1500)
    txt = pg.inner_text("#app")
    ok("5 recorder failure: message", "couldn't start recording" in txt)
    ok("5 recorder failure: Start re-enabled", pg.is_enabled("#rec"))
    ok("5 recorder failure: mic released", pg.evaluate("window.__streams.every(s => s.getTracks().every(t => t.readyState === 'ended'))"))
    c.close()

    # 3/6. upload in flight, leave the page: no false "not saved" message
    c = ctx_for(b, fast=True); pg = c.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.route("**/api/patients/*/checkins", lambda r: r.fulfill(status=200, content_type="application/json", body=json.dumps({"checkin_id": "ck_fake"})))
    pg.goto(BASE + "#/checkin"); pg.wait_for_selector("#rec")
    pg.evaluate("window.__delay = {'/api/patients/p_demo/checkins': 2000}")
    pg.click("#rec"); pg.wait_for_function("!document.getElementById('stop').disabled", timeout=10000)
    pg.click("#stop"); pg.wait_for_selector("#steps", state="attached")
    pg.evaluate("location.hash = '#/about'"); pg.wait_for_timeout(3500)
    txt = pg.inner_text("#app")
    ok("6 upload then leave: no false failure", "didn't reach the server" not in txt and "Privacy" in txt)
    c.close()

    # 7. event stream drops: falls back to the stored result
    c = ctx_for(b, fast=True); pg = c.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.route("**/api/patients/*/checkins", lambda r: r.fulfill(status=200, content_type="application/json", body=json.dumps({"checkin_id": "ck_fake2"})))
    pg.route("**/api/checkins/ck_fake2/events*", lambda r: r.abort())
    hits = {"n": 0}
    def stored(r):
        hits["n"] += 1
        if hits["n"] < 2:
            r.fulfill(status=404, body="{}")
        else:
            r.fulfill(status=200, content_type="application/json", body=json.dumps({"checkin_id": "ck_fake2", "outcome": "accepted", "status": "OK",
                      "quality": {"verdict": "PASS", "message": "Recording quality passed.", "status": "PASS", "reason_codes": []},
                      "message": "Check-in recorded.", "baseline_progress": None, "has_window_analysis": False}))
    pg.route("**/api/checkins/ck_fake2?role=patient", stored)
    pg.route("**/api/checkins/ck_fake2/verification", lambda r: r.fulfill(status=404, body="{}"))
    pg.goto(BASE + "#/checkin"); pg.wait_for_selector("#rec")
    pg.click("#rec"); pg.wait_for_function("!document.getElementById('stop').disabled", timeout=10000)
    pg.click("#stop")
    try:
        pg.wait_for_selector("text=Check-in recorded", timeout=12000); got = True
    except Exception:
        got = False
    ok("7 stream drop: result shown via fallback", got, f"polls={hits['n']}")
    c.close()

    # 9. Mark reviewed failure is reported
    c = ctx_for(b, role="clinician", pid="p_drift"); pg = c.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.route("**/api/alerts/*/ack", lambda r: r.fulfill(status=500, body="{}"))
    def add_alert(r):     # demo data has no unreviewed note: add one to the response (the ack itself is forced to fail)
        resp = r.fetch(); j = resp.json()
        ck = j["checkins"][-1]["checkin_id"]
        j["alerts"] = [{"id": "a_test", "kind": "SAFETY", "checkin_id": ck, "created_at": j["checkins"][-1]["created_at"], "acked_at": None}]
        r.fulfill(response=resp, body=json.dumps(j))
    pg.route("**/api/patients/p_drift/timeline*", add_alert)
    pg.goto(BASE + "#/research/p_drift"); pg.wait_for_timeout(3000)
    found = pg.locator("[data-ack]").count()
    if found:
        pg.locator("[data-ack]").first.click(); pg.wait_for_timeout(1000)
        n = pg.locator(".ackmsg").count()
        ok("9 mark reviewed failure: message", n == 1, pg.locator(".ackmsg").first.inner_text() if n else "nothing shown")
    else:
        ok("9 mark reviewed failure: message", False, "no unreviewed note on any profile")
    c.close()

    # ---------------------------------------------------------------- bug-fix specification (2026-11-15)
    def page(role="patient", pid="p_drift"):
        c = ctx_for(b, role=role, pid=pid); pg = c.new_page()
        pg.on("pageerror", lambda e: errs.append(str(e)))
        return c, pg

    # P1-2 / P1-3: bad day counts and unknown review ids give a static page, never "Reconnecting" polling
    for days in ("-5", "0", "0.5", "abc"):
        c, pg = page()
        health = []
        pg.on("request", lambda r: health.append(r.url) if "/api/health" in r.url else None)
        pg.goto(BASE + f"#/trends?days={days}"); pg.wait_for_selector(".ribbon")
        n0 = len(health); pg.wait_for_timeout(4000)
        txt = pg.inner_text("#app")
        ok(f"P1-3 trends?days={days}: shown, no reconnect", "Voice trends" in txt and "Reconnecting" not in txt and len(health) == n0,
           f"health calls after load: {len(health) - n0}")
        c.close()
    c, pg = page("clinician")
    health = []
    pg.on("request", lambda r: health.append(r.url) if "/api/health" in r.url else None)
    pg.goto(BASE + "#/review/flag_doesnotexist"); pg.wait_for_selector("#app h1")
    pg.wait_for_timeout(4000)
    txt = pg.inner_text("#app")
    ok("P1-2 unknown review: 'not found' + link, no polling", "review wasn't found" in txt and pg.locator('#app a[href="#/review"]').count() >= 1
       and "Reconnecting" not in txt and len(health) <= 1, f"health calls: {len(health)}")
    c.close()

    # P1-4: a review link made before "Load demo" still opens after it
    flags = [f for f in api("/v1/review-flags?status=OPEN,ACKNOWLEDGED") if f["patient_id"] == "p_drift"]
    fid = flags[0]["flag_id"] if flags else None
    api("/demo/reset", {"scenario": "all"})
    c, pg = page("clinician")
    pg.goto(BASE + f"#/review/{fid}"); pg.wait_for_selector("#app h1")
    ok("P1-4 review link survives Load demo", fid and "Speech Change Review" in pg.inner_text("#app h1"), str(fid))
    # P3-6: a review action re-renders through route(): ribbon present, message shown
    pg.route("**/api/v1/review-flags/*/actions", lambda r: r.fulfill(status=200, content_type="application/json", body="{}"))
    pg.locator("[data-act]").first.click(); pg.wait_for_timeout(2500)
    ok("P3-6 review action keeps the ribbon", pg.locator("#app .ribbon").count() == 1 and "Saved" in pg.inner_text("#ractmsg"))
    c.close()

    # P3-7: unknown filter in the address shows the Pending tab
    c, pg = page("clinician")
    pg.goto(BASE + "#/review?status=bogus"); pg.wait_for_selector(".tabs")
    cur = pg.locator('.tabs a[aria-current="page"]').inner_text()
    ok("P3-7 #/review?status=bogus shows Pending", cur == "Pending", cur)
    # P3-3: the Patients list count equals the open reviews the linked queue shows
    pg.goto(BASE + "#/research"); pg.wait_for_selector("th:has-text('Open reviews')")
    row = pg.locator("tbody tr", has_text="Sam (simulated")         # not "Sample speaker"
    shown = row.locator("td").last.inner_text().strip()
    n_open = len([f for f in api("/v1/review-flags?status=OPEN,ACKNOWLEDGED") if f["patient_id"] == "p_drift"])
    ok("P3-3 'Open reviews' matches the queue", shown == str(n_open) and "Open reviews" in pg.inner_text("thead"), f"{shown} vs {n_open}")
    # P3-5: no "your" on clinician pages
    hits = {}
    for h in ("#/review", f"#/review/{fid}", "#/research", "#/research/p_drift", "#/trends", "#/results"):
        pg.goto(BASE + h); pg.wait_for_selector(".ribbon"); pg.wait_for_timeout(500)
        found = re.findall(r"[^.\n]{0,30}\byour\b[^.\n]{0,30}", pg.inner_text("#app"), re.I)
        if found: hits[h] = found[:2]
    ok("P3-5 clinician pages never say 'your'", not hits, json.dumps(hits)[:300])
    c.close()

    # P4-1 / P4-6 / P4-7: Sam's Home (patient)
    c, pg = page()
    pg.goto(BASE + "#/"); pg.wait_for_selector(".vd-hero")
    txt = pg.inner_text("#app")
    stat = pg.inner_text(".attention-stat strong")
    n_above = int(stat.split("/")[0])
    want = "Several recent recordings different" if n_above >= 2 else "One recording different"
    ok("P4-1 Home heading matches the count", want in pg.inner_text("#attn-h"), f"{stat!r} -> {pg.inner_text('#attn-h')!r}")
    ok("P4-6 instruction paragraph appears once", txt.count("Read the passage aloud at your normal speaking pace") == 1)
    n = api("/patients/p_drift/timeline?days=365")["baseline"]["n"]
    ok("P4-7 baseline count is the stored one", f"from {n} recording" in txt)
    ok("P4-5 seeded times read as daytime", not re.search(r"\b[1-6]:\d\d\s?AM", txt), (re.findall(r"\d+:\d\d\s?[AP]M", txt) or [""])[0])
    c.close()

    # P3-4: a repeat request on a synthetic profile offers no record button
    c, pg = page()
    patch_timeline(pg, "p_drift", lambda j: j.update(repeat_requested=True))
    pg.goto(BASE + "#/"); pg.wait_for_selector(".vd-hero")
    box = pg.locator(".callout", has_text="Repeat check-in requested")
    ok("P3-4 synthetic repeat request: no record button", box.count() == 1 and box.locator("a, button").count() == 0
       and "can't be recorded here" in box.inner_text())
    c.close()

    # P4-4: rounding to zero, zero boundary, extreme score
    c, pg = page()
    def edge(j):
        sc = [x for x in j["checkins"] if x.get("drift_score") is not None]
        sc[-1]["speaking_rate_change_pct"] = -0.4
    patch_timeline(pg, "p_drift", edge)
    pg.goto(BASE + "#/"); pg.wait_for_selector(".metric-grid")
    sr = pg.locator(".metric-card").first.locator(".metric-value").inner_text()
    ok("P4-4 -0.4% shows 'No change'", "No change" in sr and "0%" not in sr, sr)
    c.close()
    c, pg = page("clinician")
    def extreme(j):
        sc = [x for x in j["checkins"] if x.get("drift") and x["drift"].get("tau")]
        sc[-1]["drift"]["d"] = sc[-1]["drift"]["tau"] * 40          # 40x the boundary
        sc[-2]["drift"]["tau"] = 0                                   # missing boundary
    patch_timeline(pg, "p_drift", extreme)
    pg.goto(BASE + "#/research/p_drift"); pg.wait_for_selector("svg.chart")
    grid = pg.locator("svg.chart line.grid").count()
    ok("P4-4 extreme score: <= 9 gridlines + off-scale marker", grid <= 9 and "off scale (40.0)" in pg.inner_html("svg.chart"), f"{grid} lines")
    sel = pg.locator('.row.small a[data-ck]').nth(-2)
    sel.click(); pg.wait_for_timeout(400)
    ok("P4-4 zero boundary shows no 'Infinity'", "Infinity" not in pg.inner_text("#app"))
    c.close()

    # P4-2: privacy statements agree (reading passage never sent)
    c, pg = page()
    pg.goto(BASE + "#/about"); pg.wait_for_selector("h1")
    about = pg.inner_text("#app")
    pg.goto(BASE + "#/checkin"); pg.wait_for_selector("h1")
    ok("P4-2 About and Check-in agree", "Reading-passage check-ins (the standard check-in) are never sent" in about
       and "Sent to Anthropic's API to find quotes" not in about)
    c.close()

    # P4-3: delete names the profile and refreshes the page (the DELETE itself is intercepted)
    c, pg = page()
    dialogs = []
    pg.on("dialog", lambda d: (dialogs.append(d.message), d.accept()))
    pg.route("**/api/patients/p_drift", lambda r: r.fulfill(status=200, content_type="application/json", body='{"deleted":"p_drift"}')
             if r.request.method == "DELETE" else r.continue_())
    pg.goto(BASE + "#/about"); pg.wait_for_selector("#del")
    pg.click("#del"); pg.wait_for_timeout(2000)
    ok("P4-3 delete confirmation names the profile", dialogs and "Sam" in dialogs[0] and "p_drift" not in dialogs[0], dialogs[0] if dialogs else "")
    ok("P4-3 page refreshed with 'Deleted'", "Deleted all stored data for Sam" in pg.inner_text("#delmsg") and pg.locator("#app .ribbon").count() == 1)
    c.close()

    # P2-5 + P5-1: the upload names the real container and sends the time-zone offset (upload intercepted)
    c = ctx_for(b, fast=True); pg = c.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
    sent = {}
    def capture(r):
        sent["body"] = r.request.post_data_buffer or b""
        r.fulfill(status=409, content_type="application/json", body='{"detail":"test stop"}')
    pg.route("**/api/patients/p_demo/checkins", capture)
    pg.goto(BASE + "#/checkin"); pg.wait_for_selector("#rec")
    pg.click("#rec"); pg.wait_for_function("!document.getElementById('stop').disabled", timeout=10000)
    pg.click("#stop"); pg.wait_for_timeout(2000)
    body = sent.get("body", b"")
    ok("P2-5 upload filename matches the recorder", b'filename="checkin.webm"' in body)
    ok("P5-1 upload sends tz_offset_min", b'name="tz_offset_min"' in body)
    c.close()

    # P1-5 server half, seen from the browser: unknown check-in stream is a 404, not endless keep-alives
    c, pg = page()
    pg.goto(BASE + "#/about"); pg.wait_for_selector("h1")
    st = pg.evaluate("fetch('/api/checkins/c_nope/events?role=patient').then(r => r.status)")
    ok("P1-5 unknown check-in stream: 404", st == 404, str(st))
    c.close()

    ok("no uncaught page errors", not errs, "; ".join(errs[:3]))
    b.close()

print(f"\n{sum(r[1] for r in results)}/{len(results)} passed")
raise SystemExit(0 if all(r[1] for r in results) else 1)
