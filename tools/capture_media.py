"""Capture the Devpost screenshots from a RUNNING server (real models, demo data loaded) into devpost_media/.

    python tools/capture_media.py [--base http://127.0.0.1:8000] [--browser chrome|msedge]

Before running: start the server, run tools/warm_demo.py, and click "Load demo: all" (or POST /api/demo/reset).
Each shot opens a fresh browser context with the view (patient/clinician) and profile set exactly as the app stores
them, so nothing is faked: every image is the live app.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "devpost_media"


def main() -> int:
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--browser", default="chrome")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    OUT.mkdir(exist_ok=True)
    flags = json.load(urllib.request.urlopen(f"{base}/api/v1/review-flags?status=OPEN,ACKNOWLEDGED"))
    sam_flag = next((f["flag_id"] for f in flags if f["patient_id"] == "p_drift"), None)

    shots = [  # (file, role, profile, hash, how to frame it)
        ("01_noise_demo_chart.png", "clinician", "p_drift", "#/results", ("parent_of", "#demoout svg")),
        ("02_home_patient.png", "patient", "p_drift", "#/", ("span", ".vd-hero", ".metric-grid")),
        ("03_trend_chart.png", "patient", "p_drift", "#/trends", ("card_with", "svg.chart")),
        ("04_checkin_page.png", "patient", "p_demo", "#/checkin", ("el", ".checkin-layout")),
        ("05_review_item.png", "clinician", "p_drift", f"#/review/{sam_flag}" if sam_flag else "#/review", ("top", 1150)),
        ("06_results_table.png", "clinician", "p_drift", "#/results", ("numbers_table",)),
    ]
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=a.browser, headless=True)
        for name, role, pid, hash_, frame in shots:
            ctx = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
            ctx.add_init_script(f"localStorage.setItem('vd_role','{role}');localStorage.setItem('vd_pid','{pid}');"
                                "localStorage.setItem('vd_results_mode','auto');")
            page = ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{base}/{hash_}", wait_until="networkidle")
            page.wait_for_timeout(1200)                      # fonts + late renders
            kind, *args = frame
            path = str(OUT / name)
            if kind == "el":
                page.locator(args[0]).first.screenshot(path=path)
            elif kind == "parent_of":
                page.locator(args[0]).first.locator("xpath=..").screenshot(path=path)
            elif kind == "card_with":
                page.locator("section.card").filter(has=page.locator(args[0])).first.screenshot(path=path)
            elif kind == "numbers_table":
                sec = page.locator("section", has=page.locator("h2", has_text="The numbers")).first
                sec.locator(".tablewrap").first.screenshot(path=path)
            elif kind == "span":
                b1, b2 = page.locator(args[0]).first.bounding_box(), page.locator(args[1]).first.bounding_box()
                page.screenshot(path=path, full_page=True, clip={"x": b1["x"] - 8, "y": b1["y"] - 8,
                                                                  "width": b1["width"] + 16,
                                                                  "height": b2["y"] + b2["height"] - b1["y"] + 16})
            elif kind == "top":
                page.screenshot(path=path, full_page=True, clip={"x": 0, "y": 0, "width": 1440, "height": args[0]})
            print(f"{name:28s} {'OK' if not errors else 'PAGE ERRORS: ' + '; '.join(errors)}", flush=True)
            ctx.close()
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
