"""GPU budget ledger + watchdog (spec section 4). Hard cap 5.0 h; watchdog trips at 4.75 h total.

CLI (run before any GPU stage):  python tools/budget.py --drive <dir>
Library: `with budget.stage("G1", drive_dir): ...` writes the ledger row, starts a watchdog thread that
heartbeats every 30 s and unassigns the runtime on kill-rule/total breach, and always closes the row +
unassigns in `finally`.

Disconnects: if the runtime dies, `finally` never runs and the row stays open. An open row whose heartbeat is
older than STALE_S is treated as ended at its last heartbeat (and is closed that way when the next stage starts),
so a dead session doesn't keep inflating the hours used. An open row with a FRESH heartbeat is still running
somewhere: refuse to start another stage on top of it.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

CAP_H = 5.0
TRIP_H = 4.75
STALE_S = 120          # no heartbeat for 2 min = that runtime is gone
HEARTBEAT_S = 30
# stage -> (planned_h, kill_h)
RULES = {"G0": (0.25, 0.25), "G1": (0.75, 1.0), "G2": (0.5, 0.75), "G3": (0.25, 0.25), "G4": (0.25, 0.25),
         # T1: GPU training/eval (step 6). NOT in the spec's plan: it draws on the 3.0 h reserve, which the
         # spec says needs a team decision. Metered like any other stage.
         "T1": (0.5, 0.75),
         # L1: lightweight projection experiment extraction (~2,400 clips): projection-head spec v0.1
         "L1": (0.25, 0.5),
         # L2: aggregate-experiment noise sweep (eval speakers only, ~480 clips)
         "L2": (0.1, 0.25),
         # L3: evidence-spec change conditions (slower delivery, change + noise), eval speakers only, ~360 clips
         "L3": (0.1, 0.25)}
FIELDS = ["stage", "start_utc", "end_utc", "gpu_name", "last_seen_utc", "note"]
_lock = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ledger(drive: Path) -> Path:
    p = Path(drive) / "budget" / "ledger.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(",".join(FIELDS) + "\n")
    else:  # upgrade an older 4-column ledger in place
        with open(p, newline="") as f:
            header = next(csv.reader(f), [])
        if header != FIELDS:
            with open(p, newline="") as f:
                rows = list(csv.DictReader(f))
            _write(p, rows)
    return p


def _write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def read_rows(drive: Path) -> list[dict]:
    with open(_ledger(drive), newline="") as f:
        return list(csv.DictReader(f))


def _ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _effective_end(r: dict, now: datetime) -> datetime:
    """Closed row: its end. Open row: now if its heartbeat is fresh (still running), else its last heartbeat.
    Open row with no heartbeat at all (pre-heartbeat ledger or died before the first beat): its kill-rule
    limit, i.e. the most it could have run before the watchdog would have stopped it (conservative)."""
    if r.get("end_utc"):
        return _ts(r["end_utc"])
    seen = _ts(r.get("last_seen_utc"))
    start = _ts(r["start_utc"])
    if seen is None:
        cap = start + timedelta(hours=RULES.get(r["stage"], (0, 1.0))[1])
        return min(now, cap)
    return now if (now - seen).total_seconds() <= STALE_S else seen


def is_running(r: dict, now: datetime | None = None) -> bool:
    now = now or _now()
    if r.get("end_utc"):
        return False
    seen = _ts(r.get("last_seen_utc"))
    return seen is not None and (now - seen).total_seconds() <= STALE_S


def hours_used(drive: Path, now: datetime | None = None) -> float:
    now = now or _now()
    return sum(max(0.0, (_effective_end(r, now) - _ts(r["start_utc"])).total_seconds()) / 3600
               for r in read_rows(drive))


def close_stale(drive: Path, now: datetime | None = None) -> list[str]:
    """Close open rows from runtimes that died (no fresh heartbeat). Returns the stages closed."""
    now = now or _now()
    path = _ledger(drive)
    with _lock:
        rows = read_rows(drive)
        closed = []
        for r in rows:
            if not r.get("end_utc") and not is_running(r, now):
                r["end_utc"] = _effective_end(r, now).isoformat()
                r["note"] = "closed after disconnect"
                closed.append(r["stage"])
        if closed:
            _write(path, rows)
    return closed


def _update_row(drive: Path, start_iso: str, **fields) -> None:
    path = _ledger(drive)
    with _lock:
        rows = read_rows(drive)
        for r in rows:
            if r["start_utc"] == start_iso:
                r.update(fields)
        _write(path, rows)


def _gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def _unassign() -> None:  # pragma: no cover - Colab only
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception:
        pass


@contextlib.contextmanager
def stage(name: str, drive: str | Path, unassign=_unassign, heartbeat_s: float = HEARTBEAT_S,
          unassign_on_exit: bool = True):
    """unassign_on_exit=False only for chaining stages in one session (the LAST stage must unassign).
    The watchdog still unassigns on a kill-rule or total-budget breach either way."""
    drive = Path(drive)
    closed = close_stale(drive)
    if closed:
        print(f"[budget] closed stale ledger rows from a disconnected runtime: {closed}")
    live = [r["stage"] for r in read_rows(drive) if is_running(r)]
    if live:
        raise RuntimeError(f"stage(s) {live} still heartbeating in another runtime: stop it before starting {name} "
                           "(Colab: Runtime > Manage sessions)")
    used = hours_used(drive)
    if used >= TRIP_H:
        raise RuntimeError(f"GPU ledger at {used:.2f} h >= {TRIP_H} h: refusing to start {name}")
    path = _ledger(drive)
    start = _now()
    with _lock:
        rows = read_rows(drive)
        rows.append({"stage": name, "start_utc": start.isoformat(), "end_utc": "", "gpu_name": _gpu_name(),
                     "last_seen_utc": start.isoformat(), "note": ""})
        _write(path, rows)
    done = threading.Event()

    def watchdog():
        kill_h = RULES[name][1]
        while not done.wait(heartbeat_s):
            try:
                _update_row(drive, start.isoformat(), last_seen_utc=_now().isoformat())
            except OSError:
                pass  # Drive hiccup: try again next beat
            el = (_now() - start).total_seconds() / 3600
            if el > kill_h or hours_used(drive) > TRIP_H:
                unassign()
                return

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        yield
    finally:
        done.set()
        end = _now().isoformat()
        _update_row(drive, start.isoformat(), end_utc=end, last_seen_utc=end)
        if unassign_on_exit:
            unassign()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default=".")
    a = ap.parse_args()
    drive = Path(a.drive)
    now = _now()
    for r in read_rows(drive):
        state = "done" if r.get("end_utc") else ("RUNNING" if is_running(r, now) else "OPEN (disconnected)")
        h = (_effective_end(r, now) - _ts(r["start_utc"])).total_seconds() / 3600
        print(f"  {r['stage']:<3} {r['start_utc'][:19]}  {h:5.2f} h  {state:<20} {r.get('gpu_name', '')}")
    used = hours_used(drive, now)
    print(f"GPU hours used: {used:.2f} / {CAP_H:.1f}   remaining: {CAP_H - used:.2f}   (watchdog trips at {TRIP_H})")


if __name__ == "__main__":
    main()
