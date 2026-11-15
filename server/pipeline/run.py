"""Check-in orchestrator (spec 5.1). Models load once into a Runtime; each check-in runs in a thread."""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import scoring, store
from ..config import ROOT, get_config, resolve
from . import acoustic, asr, decode, drift, language, noise, quality, safety, windows

log = logging.getLogger("vd.run")


def demo_enrollment() -> dict:
    """Profiles created by this demo app are enrolled with the explicit hackathon stand-in 'demo_consent'
    (alerting spec E5); it is never real-world authorization, and only applies in demo notification mode."""
    if get_config().monitoring["notification_mode"] != "demo":
        return {"monitoring_enrollment_status": "not_enrolled", "review_consent_status": "none"}
    return {"monitoring_enrollment_status": "enrolled", "review_consent_status": "demo_consent",
            "assigned_reviewer_id": get_config().monitoring["default_reviewer"]}


class EventBus:
    """Per-check-in ordered event list with replay, so late SSE subscribers miss nothing."""

    def __init__(self):
        self._events: dict[str, list[dict]] = {}
        self._cv = threading.Condition()

    def emit(self, cid: str, kind: str, data: dict | None = None):
        with self._cv:
            self._events.setdefault(cid, []).append({"event": kind, "data": data or {}})
            self._cv.notify_all()

    def get(self, cid: str) -> list[dict]:
        with self._cv:
            return list(self._events.get(cid, []))

    def wait(self, cid: str, n_seen: int, timeout: float = 15.0) -> list[dict]:
        with self._cv:
            self._cv.wait_for(lambda: len(self._events.get(cid, [])) > n_seen, timeout)
            return list(self._events.get(cid, []))[n_seen:]


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=ROOT, timeout=2).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


class Runtime:
    def __init__(self):
        self.cfg = get_config()
        self.dm = drift.load_drift_model()
        self.dm_raw = drift.load_drift_model(raw=True)
        self.acoustic = acoustic.AcousticBranch()
        if "fallback" in (self.acoustic.encoder_name or "") and self.dm.version != "fallback":
            # Trained drift constants (PCA, tau) were fitted to WavLM/projection embeddings. Applying them to the
            # spectral fallback features would produce scores that look calibrated but mean nothing.
            log.warning("encoder is %s: ignoring trained drift artifacts %s, using the fallback drift space",
                        self.acoustic.encoder_name, self.dm.version)
            self.dm = self.dm_raw = drift.fallback_model()
        self.reference = acoustic.ReferenceRisk()
        self.cohort = acoustic.CohortRisk()
        self.asr = asr.ASR()
        self.language = language.LanguageBranch()
        self.bus = EventBus()
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.git = git_sha()
        self.tmp = resolve(self.cfg.server["tmp_dir"])
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.sweep_tmp()
        log.info("runtime ready: encoder=%s asr=%s artifacts=%s tau=%.2f llm=%s", self.acoustic.encoder_name,
                 self.asr.engine, self.dm.version, self.dm.tau, self.language.reachable)

    def sweep_tmp(self) -> int:
        """Startup sweep: delete leftover audio older than tmp_sweep_s."""
        n, cutoff = 0, time.time() - float(self.cfg.server["tmp_sweep_s"])
        for f in self.tmp.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                n += 1
        return n

    def health(self) -> dict:
        gpu = False
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            pass
        return {"models_loaded": True, "gpu": gpu, "artifacts_version": self.dm.version,
                "llm_reachable": self.language.reachable, "encoder": self.acoustic.encoder_name,
                "asr": self.asr.engine, "reference_risk": self.reference.model is not None,
                "cohort_risk": self.cohort.available}

    # ------------------------------------------------------------------ the pipeline
    def process(self, cid: str, pid: str, audio_path: Path, prompt_id: str = "default",
                interventions: list | None = None, verification_of: str | None = None,
                meta: dict | None = None) -> None:
        """Runs a full check-in. Always deletes the audio file, even on error.
        interventions: known noise events marked during recording (seconds from start), for the visualization."""
        t_start = time.perf_counter()
        lat: dict[str, int] = {}
        emit = lambda k, d=None: self.bus.emit(cid, k, d)
        ms = lambda t: int((time.perf_counter() - t) * 1000)
        try:
            try:
                self._process(cid, pid, audio_path, emit, lat, t_start, ms, interventions, prompt_id, verification_of,
                              meta)
            except Exception as e:  # never leave the client hanging
                if isinstance(e, decode.DecodeError):
                    log.warning("check-in %s: %s", cid, str(e)[:120])
                else:
                    log.exception("check-in %s failed", cid)
                store.log_event(cid, "error", 0, repr(e)[:300])
                self._store_failure(cid, pid, repr(e), prompt_id, verification_of, meta)
                emit("result", self._result(cid))
        finally:
            try:
                Path(audio_path).unlink(missing_ok=True)
            except OSError:
                log.error("could not delete %s", audio_path)

    def _result(self, cid: str) -> dict:
        with store.session() as s:
            return scoring.result_json(s.get(store.Checkin, cid))

    def _store_failure(self, cid: str, pid: str, err: str, prompt_id: str = "default",
                       verification_of: str | None = None, meta: dict | None = None) -> None:
        from .. import protocol
        reason = "unreadable audio" if "unreadable" in err else "processing error"
        with store.session() as s:
            if s.get(store.Checkin, cid) is None:
                s.add(store.Checkin(id=cid, patient_id=pid, status="BASELINE",
                                    quality={"verdict": "REJECT", "status": "REJECT", "speech_s": 0, "snr_db": 0,
                                             "clipping": 0, "reasons": [reason],
                                             "reason_codes": [quality.REASON_CODES[reason]]},
                                    versions={"git": self.git},
                                    extra={"error": err, "protocol": protocol.meta(prompt_id), **(meta or {}),
                                           **({"verification_of": verification_of, "checkin_type": "verification"}
                                              if verification_of else {})}))
                s.commit()

    def window_analysis(self, s, pid: str, x, q, emb, interventions=None, before=None) -> dict:
        """Raw-vs-projected window analysis of one recording against the patient's frozen baseline."""
        prev = [c for c in scoring.history(s, pid) if before is None or c.created_at < before]
        base = scoring.baseline_rows(prev, int(self.cfg.drift["baseline_n"]))
        return windows.analyze(x=x, quality=q, emb=emb, baseline=base,
                               encoder=emb.encoder if emb else self.acoustic.encoder_name,
                               wcfg=dict(self.cfg.windows), interventions=interventions,
                               population_meta=self.dm.meta, baseline_n=int(self.cfg.drift["baseline_n"]))

    def replay(self, pid: str, clean_path: Path, *, kind: str, start_s: float, end_s: float, snr_db: float,
               seed: int = 0, noise_path: Path | None = None) -> dict:
        """Controlled replay (Mode 1): the SAME speech analysed clean and with a known noise interval.
        Analysis only: nothing is stored as a check-in, so longitudinal drift is unaffected. Audio is deleted."""
        try:
            x = decode.decode_audio(clean_path)
            nf = decode.decode_audio(noise_path) if noise_path else None
        finally:
            for p in (clean_path, noise_path):
                if p:
                    Path(p).unlink(missing_ok=True)
        q_clean = quality.assess(x, self.cfg)
        nz = noise.make_noise(kind, len(x), seed, ref=x, noise_file=nf)
        xn = noise.mix_interval(x, nz, start_s, end_s, snr_db, q_clean.speech_mask)
        iv = [{"type": f"controlled_{kind}_noise", "start_seconds": start_s, "end_seconds": end_s,
               "snr_db": snr_db, "source": "replay"}]
        out = {"mode": "controlled_replay", "patient_id": pid, "noise": {"kind": kind, "snr_db": snr_db,
               "start_seconds": start_s, "end_seconds": end_s, "seed": seed}, "illustrative": False}
        with store.session() as s:
            for name, sig in (("clean", x), ("noisy", xn)):
                q = quality.assess(sig, self.cfg)
                emb = self.acoustic.embed(sig, q.speech_mask)   # always: windows decide validity (state D/E)
                a = self.window_analysis(s, pid, sig, q, emb, iv)
                if name == "clean":   # same interval, outlined only: no noise was added to this version
                    a["interventions"] = [{**w, "type": "reference_interval_no_noise"} for w in a["interventions"]]
                out[name] = windows.contract(a, "controlled_experiment")
        return out

    def _process(self, cid, pid, audio_path, emit, lat, t_start, ms, interventions=None, prompt_id="default",
                 verification_of=None, meta=None):
        from .. import protocol
        from . import evidence
        cfg = self.cfg
        t = time.perf_counter()
        x = decode.decode_audio(audio_path)
        dur = len(x) / decode.SR
        q = quality.assess(x, cfg)
        if dur < cfg.audio["min_s"]:
            q = quality.Quality("REJECT", q.speech_s, q.snr_db, q.clipping, ["recording too short"], q.speech_mask,
                                q.duration_s, q.rms)
        elif dur > cfg.audio["max_s"]:
            q = quality.Quality("REJECT", q.speech_s, q.snr_db, q.clipping, ["recording too long"], q.speech_mask,
                                q.duration_s, q.rms)
        lat["decode"] = ms(t)
        store.log_event(cid, "quality", lat["decode"])
        emit("stage", {"stage": "quality", "ms": lat["decode"], "data": q.to_json()})

        # Speech-to-text and the acoustic embedding are independent, so they run at the SAME time (this halves the
        # wall-clock time of a check-in on CPU). The safety screen still gates everything: if the transcript trips it,
        # the embedding is simply discarded, so what is stored and shown is exactly as before.
        t_par = time.perf_counter()

        def acoustic_job():
            t0 = time.perf_counter()
            e = self.acoustic.embed(x, q.speech_mask)
            d = self.reference.score(x)
            lat["acoustic"] = ms(t0)
            return e, d

        fa = self.pool.submit(acoustic_job) if q.verdict != "REJECT" else None
        t = time.perf_counter()
        tr = self.asr.transcribe(x)
        lat["asr"] = ms(t)
        store.log_event(cid, "transcript", lat["asr"])
        emit("stage", {"stage": "transcript", "ms": lat["asr"], "data": {"available": tr.available}})

        lex = safety.lexicon_match(tr.text)  # gates everything after it
        scored = q.verdict != "REJECT" and not lex

        emb = dam = lang = None
        if scored:
            def llm_job():
                t0 = time.perf_counter()
                r = self.language.analyze(tr.text)
                lat["llm"] = ms(t0)
                store.log_event(cid, "evidence", lat["llm"], None if r["available"] else "unavailable")
                emit("stage", {"stage": "evidence", "ms": lat["llm"], "data": {}})
                return r

            # A fixed reading passage says nothing about how the person feels: never mine it for symptom language.
            reading = protocol.get(prompt_id)["task_type"] == "read_speech"
            fl = None if reading else self.pool.submit(llm_job)
            emb, dam = fa.result()
            store.log_event(cid, "scores", lat["acoustic"])
            emit("stage", {"stage": "scores", "ms": lat["acoustic"], "data": {}})
            if reading:
                emit("stage", {"stage": "evidence", "ms": 0, "data": {"skipped": True}})
                lang = {"available": False, "evidence": [], "evidence_dropped": 0, "summary": "",
                        "safety_flag": False, "not_applicable": "reading passage: no symptom-language analysis"}
            else:
                lang = fl.result()
        else:
            if fa is not None:            # safety path: the embedding finishes in the background and is discarded
                fa.add_done_callback(lambda f: f.exception())
            for st in ("scores", "evidence"):
                emit("stage", {"stage": st, "ms": 0, "data": {"skipped": True}})
        lat["parallel"] = ms(t_par)

        sf = safety.screen(tr.text, bool(lang and lang["safety_flag"]))
        words_timing = asr.timing_features(tr.words, q.speech_s) if tr.available else None

        ck = store.Checkin(
            id=cid, patient_id=pid, quality=q.to_json(), transcript=tr.text,
            reference_risk=dam if (dam and dam.get("source") != "unavailable") else dam,
            evidence=(lang or {}).get("evidence", []),
            versions={"artifacts": self.dm.version, "git": self.git, "asr": tr.engine,
                      "encoder": emb.encoder if emb else self.acoustic.encoder_name},
        )
        ck.extra = {"evidence_dropped": (lang or {}).get("evidence_dropped", 0),
                    "evidence_available": bool(lang and lang["available"]) or not scored,
                    "language_analysis": (lang or {}).get("not_applicable") or "performed",
                    "summary": (lang or {}).get("summary", ""),
                    "window_count": emb.window_count if emb else 0,
                    "protocol": protocol.meta(prompt_id)}
        if verification_of:
            ck.extra.update(verification_of=verification_of, checkin_type="verification")
        ck.extra.update(meta or {})              # e.g. {"source": "demo", "synthetic_voice": true}
        if scored:   # same extraction for every recording; rejected recordings carry no measurements at all
            ck.extra["acoustic"] = evidence.measure(q.speech_mask, words_timing, tr.available)
        if emb is not None:   # clinician-only cohort reference; dropped for SAFETY by result_json
            ck.extra["cohort_risk"] = self.cohort.score(emb.logit)
        if emb is not None and emb.W is not None:       # per-window vectors: needed for window-level calibration
            ck.window_vecs, ck.window_vecs_raw = store.mat_to_blob(emb.W), store.mat_to_blob(emb.W_raw)
        with store.session() as s:
            if s.get(store.Patient, pid) is None:
                s.add(store.Patient(id=pid, display_name=pid, consent_at=store.utcnow(), **demo_enrollment()))
            try:   # visualization only: a failure here must never break the check-in
                ck.extra["window_analysis"] = self.window_analysis(s, pid, x, q, emb, interventions, ck.created_at)
            except Exception:
                log.exception("window analysis failed for %s", cid)
            scoring.apply_scoring(s, ck, vec=emb.v if emb else None, vec_raw=emb.v_raw if emb else None,
                                  timing=words_timing, safety=sf, dm=self.dm)
            if verification_of:
                from .. import verification
                ck.extra["verification"] = verification.evaluate(s, ck)
            lat["total"] = ms(t_start)
            ck.extra["latency_ms"] = lat
            s.add(ck)
            s.commit()
            res = scoring.result_json(ck)
        try:   # review-flag engine runs on the SAVED check-in; never blocks or breaks the check-in itself
            from .. import monitoring
            with store.session() as s:
                monitoring.evaluate(s, pid, cid)
                s.commit()
        except Exception:
            log.exception("monitoring evaluation failed for %s", cid)
        emit("result", res)

