// Voice Drift Monitor frontend: dependency-free ES module (no build step), served by FastAPI.
const $app = document.getElementById("app");
const DEMO = new URLSearchParams(location.search).get("demo") === "1";
const MIN_S = 20, MAX_S = 60;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const api = async (path, opts) => {
  const r = await fetch("/api" + path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
};
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const fmtDate = (iso) => new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
const fmtDT = (iso) => new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
const badge = (s) => `<span class="badge ${esc(s)}">${esc(s)}</span>`;
const simTag = (r) => (r.simulated ? `<span class="sim">simulated</span>` : "");

let cleanup = () => {};
function route() {
  cleanup(); cleanup = () => {};
  const h = location.hash.replace(/^#/, "") || "/";
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", h.startsWith("/clinician") === (a.dataset.nav === "clinician")));
  const w = h.match(/^\/windows\/(.+)$/);
  if (w) return viewWindows(decodeURIComponent(w[1]));
  if (h.startsWith("/lab")) return viewLab();
  const m = h.match(/^\/clinician\/(.+)$/);
  if (m) return viewDetail(decodeURIComponent(m[1]));
  if (h.startsWith("/clinician")) return viewList();
  return viewCheckin();
}
window.addEventListener("hashchange", route);
route();

// ------------------------------------------------------------------ check-in (patient)
async function viewCheckin() {
  const prompt = await api("/prompt").catch(() => ({ text: "Tell me about your day." }));
  let patients = [];
  try { patients = await api("/patients"); } catch {}
  const pid0 = localStorage.getItem("vd_pid") || "p_demo";
  $app.innerHTML = `
    <h1>Daily check-in</h1>
    <div class="banner">Your voice recording stays on this computer and is deleted after processing. Only the written transcript is sent to a language model. Live check-ins are for consenting team members only.</div>
    <div class="card recwrap">
      ${DEMO || patients.length ? `<label class="small muted">Patient <select id="pid"></select></label>` : ""}
      <p class="prompt">${esc(prompt.text)}</p>
      <div class="ring"><svg width="180" height="180" viewBox="0 0 180 180"><circle cx="90" cy="90" r="80" fill="none" stroke="var(--line)" stroke-width="10"/><circle id="arc" cx="90" cy="90" r="80" fill="none" stroke="var(--accent)" stroke-width="10" stroke-linecap="round" stroke-dasharray="502.65" stroke-dashoffset="502.65"/></svg><div class="t" id="clock">1:00</div></div>
      <div class="meter" aria-label="input level"><div id="lvl"></div></div>
      <div class="hint" id="hint">&nbsp;</div>
      <div class="row"><button class="primary" id="rec">Start recording</button><button id="stop" disabled>Stop (after 0:20)</button><button id="cancel" disabled>Cancel</button></div>
      ${DEMO ? `<div class="row"><button id="noisemark" disabled>Mark noise start</button><span class="small muted" id="noiselog">Demo: press when background noise starts and stops, so the timeline shows it.</span></div>` : ""}
      <div class="small muted">Speak naturally for up to a minute. You can stop after 20 seconds.</div>
    </div>`;
  const sel = document.getElementById("pid");
  if (sel) {
    const opts = patients.length ? patients.map((p) => p.id) : ["p_demo"];
    if (!opts.includes("p_demo")) opts.unshift("p_demo");
    sel.innerHTML = opts.map((id) => `<option ${id === pid0 ? "selected" : ""}>${esc(id)}</option>`).join("");
  }
  const rec = document.getElementById("rec"), stop = document.getElementById("stop"), cancel = document.getElementById("cancel");
  const arc = document.getElementById("arc"), clock = document.getElementById("clock");
  const lvl = document.getElementById("lvl"), hint = document.getElementById("hint");
  let stream, mr, chunks = [], ctx, raf, t0, timer, stopped = false;
  const marks = [];               // known noise interventions, seconds from recording start
  const noiseBtn = document.getElementById("noisemark");
  const elapsed = () => (t0 ? (performance.now() - t0) / 1000 : 0);
  if (noiseBtn) noiseBtn.onclick = () => {
    const open = marks.length && marks[marks.length - 1].end_seconds == null;
    if (open) { marks[marks.length - 1].end_seconds = +elapsed().toFixed(2); noiseBtn.textContent = "Mark noise start"; }
    else { marks.push({ type: "controlled_background_noise", start_seconds: +elapsed().toFixed(2), end_seconds: null, source: "live_marker" }); noiseBtn.textContent = "Mark noise end"; }
    document.getElementById("noiselog").textContent = marks.map((m) => `noise ${m.start_seconds}s - ${m.end_seconds ?? "..."}s`).join(", ");
  };

  const finish = () => { cancelAnimationFrame(raf); clearInterval(timer); stream?.getTracks().forEach((t) => t.stop()); ctx?.close(); };
  cleanup = () => { stopped = true; finish(); if (mr && mr.state !== "inactive") mr.stop(); };

  rec.onclick = async () => {
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } }); }
    catch (e) { hint.textContent = "Microphone unavailable: " + e.message; return; }
    rec.disabled = true;
    cancel.disabled = false;
    if (noiseBtn) noiseBtn.disabled = false;
    ctx = new AudioContext();
    const an = ctx.createAnalyser(); an.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(an);
    const buf = new Float32Array(an.fftSize);
    const tick = () => {
      an.getFloatTimeDomainData(buf);
      let s = 0, pk = 0; for (const v of buf) { s += v * v; pk = Math.max(pk, Math.abs(v)); }
      const db = 20 * Math.log10(Math.sqrt(s / buf.length) + 1e-9);
      const pct = Math.max(0, Math.min(1, (db + 60) / 50));
      lvl.style.width = pct * 100 + "%";
      lvl.style.background = pk > 0.97 ? "var(--red)" : db < -50 ? "var(--amber)" : "var(--green)";
      hint.textContent = pk > 0.97 ? "Too loud - move back a little" : db < -52 ? "Too quiet - speak up or move closer" : "";
      raf = requestAnimationFrame(tick);
    };
    tick();
    mr = new MediaRecorder(stream, MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? { mimeType: "audio/webm;codecs=opus" } : undefined);
    mr.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    mr.onstop = () => { finish(); if (!stopped) upload(new Blob(chunks, { type: mr.mimeType || "audio/webm" })); };
    mr.start(250);
    t0 = performance.now();
    timer = setInterval(() => {
      const el = (performance.now() - t0) / 1000;
      const left = Math.max(0, MAX_S - el);
      clock.textContent = `${Math.floor(left / 60)}:${String(Math.ceil(left) % 60).padStart(2, "0")}`;
      arc.style.strokeDashoffset = 502.65 * (1 - el / MAX_S);
      stop.disabled = el < MIN_S;
      stop.textContent = el < MIN_S ? `Stop (after 0:${String(Math.ceil(MIN_S - el)).padStart(2, "0")})` : "Stop & submit";
      if (el >= MAX_S) mr.stop();
    }, 200);
  };
  stop.onclick = () => mr && mr.state !== "inactive" && mr.stop();
  // Cancel: stop at any time and DISCARD the recording. Nothing is uploaded, nothing is stored.
  cancel.onclick = () => {
    stopped = true;                       // onstop sees this and skips the upload
    if (mr && mr.state !== "inactive") mr.stop();
    finish();
    chunks = [];
    viewCheckin();                        // back to a fresh check-in screen
  };

  async function upload(blob) {
    const pid = sel ? sel.value : "p_demo";
    localStorage.setItem("vd_pid", pid);
    const fd = new FormData();
    fd.append("audio", blob, "checkin.webm"); fd.append("prompt_id", "default");
    const end = +elapsed().toFixed(2);
    marks.forEach((m) => { if (m.end_seconds == null) m.end_seconds = end; });   // close an open noise interval
    if (marks.length) fd.append("interventions", JSON.stringify(marks));
    $app.innerHTML = `<h1>Processing</h1><div class="card"><ul class="steps" id="steps"></ul></div>`;
    try {
      const { checkin_id } = await (await fetch(`/api/patients/${encodeURIComponent(pid)}/checkins`, { method: "POST", body: fd })).json();
      watch(checkin_id, pid);
    } catch (e) { $app.innerHTML = `<div class="card">Upload failed: ${esc(e.message)}</div>`; }
  }
}

const STEPS = [["quality", "Checking audio quality"], ["transcript", "Transcribing"], ["scores", "Analyzing your voice"], ["evidence", "Reading the transcript"]];
function watch(cid, pid) {
  const ul = document.getElementById("steps");
  ul.innerHTML = STEPS.map(([k, l]) => `<li id="s_${k}"><span class="tick">&#9675;</span><span>${l}</span><span class="ms"></span></li>`).join("");
  const es = new EventSource(`/api/checkins/${cid}/events?role=patient`);
  cleanup = () => es.close();
  es.addEventListener("stage", (e) => {
    const d = JSON.parse(e.data), li = document.getElementById("s_" + d.stage);
    if (li) { li.classList.add("done"); li.querySelector(".tick").innerHTML = "&#10003;"; li.querySelector(".ms").textContent = d.ms ? (d.ms / 1000).toFixed(1) + " s" : ""; }
  });
  es.addEventListener("result", (e) => { es.close(); patientResult(JSON.parse(e.data), pid); });
  es.onerror = () => { /* EventSource retries; result replays from the server */ };
}

function patientResult(r) {
  if (r.status === "SAFETY") {
    $app.innerHTML = `<div class="safety-panel"><div class="big">You are not alone.</div><p>${esc(r.safety.message)}</p>
      <p class="big"><a href="tel:988">Call or text 988</a></p><p>Please contact your clinician today.</p></div>`;
    return;
  }
  const prog = r.baseline_progress;
  $app.innerHTML = `<h1>${r.quality.verdict === "REJECT" ? "Let's try that again" : "Check-in recorded"}</h1>
    <div class="card"><p>${esc(r.message || "")}</p>
      ${prog ? `<div class="meter" style="width:100%"><div style="width:${(prog[0] / prog[1]) * 100}%;background:var(--accent)"></div></div><p class="small muted">Baseline ${prog[0]}/${prog[1]}</p>` : ""}
      ${simTag(r)}</div>
    <a class="btn" href="#/">New check-in</a> <a class="btn" href="#/clinician">Clinician view</a>`;
}

// ------------------------------------------------------------------ clinician list
async function viewList() {
  const [list] = await Promise.all([api("/patients")]);
  $app.innerHTML = `<h1>Patients</h1>
    ${DEMO ? demoBar() : ""}
    <div class="card"><table><thead><tr><th>Patient</th><th>Status</th><th>Last check-in</th><th>Open alerts</th></tr></thead><tbody>
      ${list.map((p) => `<tr class="click" data-id="${esc(p.id)}"><td>${esc(p.display_name)}${p.simulated ? '<span class="sim">simulated</span>' : ""}</td><td>${badge(p.status)}</td><td>${p.last_checkin ? fmtDT(p.last_checkin) : "-"}</td><td>${p.open_alerts}</td></tr>`).join("") || `<tr><td colspan="4" class="muted">No patients yet. ${DEMO ? "Use the demo controls above." : ""}</td></tr>`}
    </tbody></table></div>`;
  $app.querySelectorAll("tr.click").forEach((tr) => (tr.onclick = () => (location.hash = "#/clinician/" + encodeURIComponent(tr.dataset.id))));
  wireDemoBar();
}

function demoBar() {
  return `<div class="card row"><strong>Demo</strong>
    ${["stable", "drift", "all", "blank"].map((s) => `<button data-reset="${s}">Reset: ${s}</button>`).join("")}
    <a class="btn" href="?demo=1#/lab">Replay lab</a></div>`;
}
function wireDemoBar() {
  $app.querySelectorAll("[data-reset]").forEach((b) => (b.onclick = async () => { b.disabled = true; await post("/demo/reset", { scenario: b.dataset.reset }); route(); }));
}

// ------------------------------------------------------------------ patient detail
async function viewDetail(pid) {
  let tl;
  try { tl = await api(`/patients/${encodeURIComponent(pid)}/timeline?days=60`); }
  catch { $app.innerHTML = `<div class="card">Unknown patient. <a href="#/clinician">Back</a></div>`; return; }
  const cks = tl.checkins;
  const cal = await api("/calibration").catch(() => ({ available: false }));
  let sel = cks.length ? cks[cks.length - 1].checkin_id : null;
  const render = () => {
    const c = cks.find((x) => x.checkin_id === sel);
    $app.innerHTML = `<a href="#/clinician">&larr; Patients</a>
      <h1>${esc(tl.patient.display_name)} ${tl.patient.simulated ? '<span class="sim">simulated</span>' : ""}</h1>
      ${alertsHtml(tl)}
      <div class="card"><h2>Drift vs. personal baseline</h2>${chart(tl, sel)}
        <div class="small muted">Dashed line = threshold &tau;. First ${tl.baseline ? tl.baseline.n : 5} passing check-ins form the baseline (grey). &#9679; scored &nbsp; &#9675; low-weight (WARN) &nbsp; &times; rejected. Acted days are labeled simulated.</div></div>
      ${c ? checkinHtml(c) : '<div class="card muted">No check-ins yet.</div>'}
      ${calibrationHtml(cal)}
      ${DEMO && c ? `<div class="card" id="cmp"><h2>Robustness: same recording, two feature sets</h2><div class="muted small">Loading...</div></div>` : ""}
      ${DEMO ? demoBar() : ""}`;
    $app.querySelectorAll("[data-ck]").forEach((el) => (el.onclick = (ev) => { ev.preventDefault(); sel = el.dataset.ck; render(); }));
    $app.querySelectorAll("[data-ack]").forEach((b) => (b.onclick = async () => {
      const note = $app.querySelector(`#note_${b.dataset.ack}`).value;
      await post(`/alerts/${b.dataset.ack}/ack`, { clinician: "dr_demo", note });
      viewDetail(pid);
    }));
    wireDemoBar();
    if (DEMO && c) compare(c);
  };
  render();
}

function alertsHtml(tl) {
  if (!tl.alerts.length) return "";
  const rows = [...tl.alerts].reverse().map((a) => `<div class="alert ${a.kind} ${a.acked_at ? "acked" : ""}">
      <div class="row"><strong>${esc(a.kind)}</strong><span class="muted small">${fmtDT(a.created_at)}</span>
      <a href="#" data-ck="${esc(a.checkin_id)}" class="small">view check-in</a></div>
      ${a.acked_at ? `<div class="small muted">Acknowledged by ${esc(a.acked_by)}${a.note ? ": " + esc(a.note) : ""}</div>`
        : `<div class="row" style="margin-top:6px"><input id="note_${a.id}" placeholder="Note" size="28"><button data-ack="${a.id}">Acknowledge</button></div>`}
    </div>`).join("");
  return `<div class="card"><h2>Alerts</h2>${rows}</div>`;
}

function highlight(text, evidence) {
  let html = esc(text);
  for (const e of evidence) {
    const q = esc(e.quote).replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
    html = html.replace(new RegExp(q, "i"), (m) => `<mark title="${esc(e.item)} (${esc(e.direction)})">${m}</mark>`);
  }
  return html;
}

function checkinHtml(c) {
  const chips = (c.timing_chips || []).map((x) => `<span class="chip">${esc(x.text)}</span>`).join("");
  const t = c.timing;
  const rr = c.reference_risk;
  const hasScores = c.status !== "SAFETY";
  return `<div class="card"><div class="row"><h2 style="margin:0">Check-in ${fmtDT(c.created_at)}</h2>${badge(c.status)}${simTag(c)}${c.acted ? '<span class="sim">acted</span>' : ""}</div>
    <p class="small muted">Quality: ${esc(c.quality.verdict)} &middot; speech ${c.quality.speech_s}s &middot; SNR ${c.quality.snr_db} dB${c.environment ? " &middot; " + esc(c.environment) : ""}</p>
    ${c.status === "SAFETY" ? `<div class="safety-panel"><div class="big">Safety alert (${esc(c.safety.source)})</div><p>High-priority: contact the patient. No scores are displayed for safety check-ins.</p></div>` : ""}
    ${hasScores && c.drift ? `<p><strong>Drift D = ${c.drift.d}</strong> (&tau; = ${c.drift.tau}) ${c.drift.exceeds ? "&mdash; exceeds threshold" : ""}${c.drift.confirmed ? " &mdash; <strong>confirmed</strong>" : ""}</p>` : hasScores ? '<p class="muted">Building baseline.</p>' : ""}
    ${chips ? `<div>${chips}</div>` : ""}
    ${t && t.speech_rate_wpm != null ? `<p class="small muted">Speech ${t.speech_rate_wpm} wpm &middot; pauses ${t.pause_rate_pm}/min &middot; mean pause ${t.mean_pause_s}s</p>` : ""}
    ${hasScores && rr ? `<p class="small">Population reference (not combined with drift): ${rr.level ? `<strong>${esc(rr.level)}</strong> (${esc(rr.source)})` : "unavailable"}</p>` : ""}
    ${hasScores ? cohortHtml(c.cohort_risk) : ""}
    ${hasScores && c.has_window_analysis ? `<p><a class="btn" href="#/windows/${encodeURIComponent(c.checkin_id)}">Inside this recording: raw vs projected, window by window &rarr;</a></p>` : ""}
    ${c.summary ? `<p><em>${esc(c.summary)}</em></p>` : ""}
    ${hasScores ? `<h3>Evidence${c.evidence.length ? "" : " (none)"}</h3>${c.evidence.map((e) => `<blockquote><span class="chip">${esc(e.item)} &middot; ${esc(e.direction)}</span><br>&ldquo;${esc(e.quote)}&rdquo;</blockquote>`).join("")}
    <h3>Transcript</h3><div class="transcript">${c.transcript ? highlight(c.transcript, c.evidence) : '<span class="muted">Transcript unavailable.</span>'}</div>` : ""}
  </div>`;
}

// ------------------------------------------------------------------ cohort reference (clinician only)
// Calibrated output of our own classifier, relative to the Androids research cohort. Never shown to patients,
// never combined with drift, never sets the status. Bands, not a bare number, lead the display.
function cohortHtml(cr) {
  if (!cr) return "";
  if (!cr.available) return `<p class="small muted">Cohort reference: unavailable &mdash; ${esc(cr.reason || "")}</p>`;
  const m = cr.calibration || {};
  return `<div class="cohort"><div class="row"><strong>Cohort reference</strong>
      <span class="band ${esc(cr.band)}">${esc(cr.band)}</span>
      <span class="small muted">calibrated p = ${cr.prob.toFixed(2)}</span></div>
    <div class="small muted">${esc(cr.label)}. Platt-calibrated on held-out Androids sessions
      (AUC ${m.auc} [${(m.auc_ci || []).join(", ")}], ECE ${m.ece}). Reference only &mdash; not a diagnosis,
      not combined with drift. See &ldquo;How reliable is the cohort reference?&rdquo; below.</div></div>`;
}

function calibrationHtml(cal) {
  if (!cal || !cal.available) return "";
  const m = cal.metrics;
  return `<details class="card"><summary><strong>How reliable is the cohort reference?</strong>
      <span class="small muted">&nbsp;${cal.gate_passed ? "shown" : "hidden: did not beat chance (E1 gate)"}</span></summary>
    <div class="grid2" style="margin-top:12px"><div>${reliabilitySvg(m.reliability)}
      <div class="small muted">Reliability diagram: bars = observed share of depressed speakers in each predicted-probability bin
      (held-out sessions); dashed diagonal = perfect calibration. Bar opacity shows how many sessions are in the bin.</div></div>
    <div><table><tbody>
      <tr><td>Method</td><td>${esc(cal.method)} scaling, ${cal.session_windows}-window sessions (~${cal.session_windows * 8} s)</td></tr>
      <tr><td>Held-out data</td><td>${cal.n_sessions} sessions from ${cal.n_speakers} speakers</td></tr>
      <tr><td>AUC</td><td>${m.auc} <span class="muted">[${m.auc_ci.join(", ")}]</span></td></tr>
      <tr><td>ECE</td><td>${m.ece} <span class="muted">[${m.ece_ci.join(", ")}]</span>, uncalibrated ${m.ece_uncalibrated}</td></tr>
      <tr><td>Brier</td><td>${m.brier} <span class="muted">[${m.brier_ci.join(", ")}]</span>, uncalibrated ${m.brier_uncalibrated}</td></tr>
      <tr><td>Bands</td><td>moderate &ge; ${cal.bands.moderate}, elevated &ge; ${cal.bands.elevated}</td></tr>
    </tbody></table>
    <ul class="small muted">${(cal.caveats || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div></div></details>`;
}

function reliabilitySvg(bins) {
  const W = 260, H = 220, L = 34, B = 26, T = 8, R = 8;
  const x = (v) => L + v * (W - L - R), y = (v) => T + (1 - v) * (H - T - B);
  const nmax = Math.max(1, ...bins.map((b) => b.n));
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Reliability diagram">`;
  for (const v of [0, 0.5, 1]) s += `<line class="axis" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 4}" y="${y(v) + 4}" text-anchor="end">${v}</text><text x="${x(v)}" y="${H - 8}" text-anchor="middle">${v}</text>`;
  for (const b of bins) {
    if (!b.n || b.frac_pos == null) continue;
    const bw = (W - L - R) / bins.length;
    s += `<rect x="${x(b.bin_lo) + 1}" y="${y(b.frac_pos)}" width="${bw - 2}" height="${y(0) - y(b.frac_pos)}" fill="var(--accent)" opacity="${0.35 + 0.65 * (b.n / nmax)}"><title>predicted ${b.bin_lo}-${b.bin_hi}: observed ${b.frac_pos} (n=${b.n})</title></rect>`;
  }
  s += `<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="var(--muted)" stroke-dasharray="4 4"/>`;
  return s + "</svg>";
}

// SVG drift chart: x = check-in index, y = D. tau dashed, baseline points grey.
function chart(tl, sel) {
  const W = 900, H = 260, L = 40, R = 12, T = 12, B = 32;
  const cks = tl.checkins;
  if (!cks.length) return '<div class="muted">No data.</div>';
  const ds = cks.map((c) => c.drift?.d).filter((d) => d != null);
  const ymax = Math.max(tl.tau * 1.6, ...ds, 2) * 1.08;
  const x = (i) => L + (cks.length === 1 ? 0.5 : i / (cks.length - 1)) * (W - L - R);
  const y = (v) => T + (1 - v / ymax) * (H - T - B);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Drift score over time with threshold">`;
  s += `<rect x="${L}" y="${y(1)}" width="${W - L - R}" height="${y(0) - y(1)}" fill="var(--band)"/>`;
  for (let v = 0; v <= Math.floor(ymax); v++) s += `<line class="axis" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  s += `<line x1="${L}" x2="${W - R}" y1="${y(tl.tau)}" y2="${y(tl.tau)}" stroke="var(--red)" stroke-dasharray="6 4" stroke-width="1.5"/><text x="${W - R}" y="${y(tl.tau) - 5}" text-anchor="end" style="fill:var(--red)">&tau; = ${tl.tau.toFixed(2)}</text>`;
  const pts = [];
  cks.forEach((c, i) => {
    const cx = x(i);
    s += `<text x="${cx}" y="${H - 10}" text-anchor="middle">${fmtDate(c.created_at)}</text>`;
    const isSel = c.checkin_id === sel;
    if (c.quality.verdict === "REJECT") { s += `<g data-ck="${c.checkin_id}" style="cursor:pointer"><path d="M${cx - 5} ${y(0) - 12}l10 10m0 -10l-10 10" stroke="var(--muted)" stroke-width="2"/><title>Rejected: ${esc((c.quality.reasons || []).join(", "))}</title></g>`; return; }
    if (c.status === "SAFETY") { s += `<g data-ck="${c.checkin_id}" style="cursor:pointer"><circle cx="${cx}" cy="${y(0) - 8}" r="7" fill="var(--safety)"/></g>`; return; }
    if (c.drift) {
      pts.push([cx, y(c.drift.d)]);
      const col = c.drift.confirmed ? "var(--red)" : c.drift.exceeds ? "var(--amber)" : "var(--accent)";
      const warn = c.quality.verdict === "WARN";
      s += `<g data-ck="${c.checkin_id}" style="cursor:pointer"><circle cx="${cx}" cy="${y(c.drift.d)}" r="${isSel ? 8 : 6}" fill="${warn ? "var(--card)" : col}" stroke="${col}" stroke-width="2.5"/><title>${fmtDate(c.created_at)}: D=${c.drift.d}${c.simulated ? " (simulated)" : ""}${c.acted ? " (acted)" : ""}</title></g>`;
    } else {
      s += `<g data-ck="${c.checkin_id}" style="cursor:pointer"><circle cx="${cx}" cy="${y(0.5)}" r="${isSel ? 8 : 6}" fill="var(--base)"/><title>Baseline check-in ${fmtDate(c.created_at)}</title></g>`;
    }
    if (c.acted) s += `<text x="${cx}" y="${y(0) - 24}" text-anchor="middle" style="fill:var(--amber)">acted</text>`;
  });
  if (pts.length > 1) s = s.replace("</svg>", "") + `<polyline fill="none" stroke="var(--accent)" stroke-width="1.5" opacity=".5" points="${pts.map((p) => p.join(",")).join(" ")}"/>`;
  return s + "</svg>";
}

// ------------------------------------------------------------------ robustness panel (?demo=1)
async function compare(c) {
  const box = document.getElementById("cmp");
  if (!box) return;
  try {
    const r = await post("/demo/compare", { checkin_id: c.checkin_id });
    if (!r.available) { box.innerHTML = `<h2>Robustness</h2><p class="muted">${esc(r.reason)}</p>`; return; }
    const max = Math.max(r.raw.d, r.invariant.d, r.raw.tau) * 1.25;
    const bar = (label, o) => `<div class="bar"><span>${label}: D = ${o.d}${o.exceeds ? " (above &tau;)" : " (below &tau;)"}</span><div class="fill" style="width:${(o.d / max) * 100}%;background:${o.exceeds ? "var(--red)" : "var(--green)"}"></div><div class="tauline" style="left:${(o.tau / max) * 100}%"></div></div>`;
    box.innerHTML = `<h2>Robustness: same recording, two feature sets</h2><div class="bars">${bar("Raw WavLM features", r.raw)}${bar("Invariance head", r.invariant)}</div><p class="small muted">The vertical line is &tau;. Both numbers come from one recording and one personal baseline.</p>`;
  } catch (e) { box.innerHTML = `<h2>Robustness</h2><p class="muted">${esc(e.message)}</p>`; }
}

// ------------------------------------------------------------------ window-level raw vs projected (visualization spec v0.1)
// Both curves are each model's window distance divided by ITS OWN calibrated window threshold, so one fixed
// line at 1 serves both and never moves during a recording. Rejected windows are gaps (never zero).
const fmt = (v, d = 2) => (v == null ? "&ndash;" : Number(v).toFixed(d));

async function viewWindows(cid) {
  let a;
  try { a = await api(`/checkins/${encodeURIComponent(cid)}/windows`); }
  catch (e) { $app.innerHTML = `<div class="card">No window analysis for this check-in. <a href="javascript:history.back()">Back</a></div>`; return; }
  $app.innerHTML = `<a href="#/clinician/${encodeURIComponent(a.patient_id)}">&larr; Patient</a>
    <h1>Inside this recording</h1>
    <p class="muted">Live recording &middot; measured model outputs${a.simulated ? ' <span class="sim">simulated</span>' : ""}</p>
    ${windowChart(a)}`;
}

function windowChart(a, title = "") {
  const W = 900, L = 46, R = 14, HA = 96, HB = 240, dur = Math.max(a.duration_seconds, 1);
  const x = (t) => L + (t / dur) * (W - L - R);
  const ivBands = (h) => (a.interventions || []).map((iv) => iv.type === "reference_interval_no_noise"
    ? `<rect x="${x(iv.start_seconds)}" y="1" width="${x(iv.end_seconds) - x(iv.start_seconds)}" height="${h - 2}" fill="none" stroke="var(--amber)" stroke-dasharray="5 4"><title>Same interval as the noisy version - no noise added here</title></rect>`
    : `<rect x="${x(iv.start_seconds)}" y="0" width="${x(iv.end_seconds) - x(iv.start_seconds)}" height="${h}" fill="var(--amber)" opacity=".13"><title>Known noise intervention ${iv.start_seconds}-${iv.end_seconds}s${iv.snr_db != null ? ` at ${iv.snr_db} dB SNR` : ""}</title></rect>`).join("");
  const winLines = (h) => a.windows.map((w) => `<line x1="${x(w.start_seconds)}" x2="${x(w.start_seconds)}" y1="0" y2="${h}" stroke="var(--line)"/>`).join("");
  const timeAxis = (h) => { let s = ""; for (let t = 0; t <= dur; t += dur > 40 ? 10 : 5) s += `<text x="${x(t)}" y="${h - 4}" text-anchor="middle">${t}s</text>`; return s; };

  // ---- Panel A: recording timeline (envelope, speech, windows, quality, interventions)
  const env = a.envelope || { hz: 10, db: [], speech: [] };
  const ey = (db) => 8 + (1 - (Math.max(-70, Math.min(0, db)) + 70) / 70) * (HA - 30);
  let pA = `<svg class="chart" viewBox="0 0 ${W} ${HA}" width="100%" role="img" aria-label="Recording timeline">${ivBands(HA - 18)}`;
  env.speech.forEach((sp, i) => { if (sp) pA += `<rect x="${x(i / env.hz)}" y="4" width="${x(1 / env.hz) - x(0) + 0.5}" height="${HA - 26}" fill="var(--band)"/>`; });
  if (env.db.length) pA += `<path d="M${x(0)} ${ey(-70)} ${env.db.map((d, i) => `L${x(i / env.hz)} ${ey(d)}`).join(" ")} L${x(env.db.length / env.hz)} ${ey(-70)}Z" fill="var(--muted)" opacity=".55"/>`;
  a.windows.forEach((w) => {
    if (w.quality === "REJECT") pA += `<rect x="${x(w.start_seconds)}" y="${HA - 22}" width="${x(w.end_seconds) - x(w.start_seconds)}" height="5" fill="var(--red)"><title>Window not scored: ${esc(w.reasons.join(", "))}</title></rect>`;
    else if (w.quality === "WARN") pA += `<rect x="${x(w.start_seconds)}" y="${HA - 22}" width="${x(w.end_seconds) - x(w.start_seconds)}" height="5" fill="var(--amber)"><title>Low local SNR estimate (${w.snr_estimate_db} dB)</title></rect>`;
  });
  pA += winLines(HA - 22) + timeAxis(HA) + `<text x="4" y="14">audio</text></svg>`;

  // ---- Panel B: model comparison
  let pB = "";
  const st = a.state;
  if (st === "C" || st === "E") {
    const valid = a.windows.filter((w) => w.raw_ratio != null);
    const peak = Math.max(2, ...valid.map((w) => Math.max(w.raw_ratio, w.projected_ratio)));
    const logy = peak > 6;   // big excursions (e.g. 80x) would flatten the threshold line: switch to a log axis
    const ymax = logy ? Math.pow(10, Math.ceil(Math.log10(peak * 1.1))) : peak * 1.12, lmin = -1;
    const y = logy
      ? (v) => 10 + (1 - (Math.log10(Math.max(v, 0.1)) - lmin) / (Math.log10(ymax) - lmin)) * (HB - 40)
      : (v) => 10 + (1 - v / ymax) * (HB - 40);
    pB = `<svg class="chart" viewBox="0 0 ${W} ${HB}" width="100%" role="img" aria-label="Normalized drift per window, raw versus projected">${ivBands(HB - 26)}`;
    a.windows.forEach((w) => {   // disagreement shading (observation, not proof of denoising)
      if (w.raw_ratio == null) return;
      const rawOnly = w.raw_ratio > 1 && w.projected_ratio <= 1, projOnly = w.projected_ratio > 1 && w.raw_ratio <= 1;
      if (rawOnly || projOnly) pB += `<rect x="${x(w.start_seconds)}" y="0" width="${x(w.end_seconds) - x(w.start_seconds)}" height="${HB - 26}" fill="${rawOnly ? "#8a5cf6" : "var(--muted)"}" opacity=".16"><title>Model disagreement: ${rawOnly ? "raw above its threshold, projected below" : "projected above its threshold, raw below"}</title></rect>`;
    });
    const ticks = logy ? [0.1, 1, 10, 100, 1000, 10000].filter((v) => v <= ymax) : Array.from({ length: Math.floor(ymax / (ymax > 4 ? 1 : 0.5)) + 1 }, (_, i) => i * (ymax > 4 ? 1 : 0.5));
    ticks.forEach((v) => { pB += `<line class="axis" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`; });
    if (logy) pB += `<text x="${L + 4}" y="12">log scale</text>`;
    pB += winLines(HB - 26);
    pB += `<line x1="${L}" x2="${W - R}" y1="${y(1)}" y2="${y(1)}" stroke="var(--ink)" stroke-width="1.6" stroke-dasharray="7 5"/><text x="${W - R}" y="${y(1) - 6}" text-anchor="end" style="fill:var(--ink)">fixed threshold = 1</text>`;
    const line = (key, col, dash) => {   // break the line at missing windows
      let d = "", pen = false, dots = "";
      a.windows.forEach((w) => {
        const v = w[key], cx = x((w.start_seconds + w.end_seconds) / 2);
        if (v == null) { pen = false; return; }
        d += `${pen ? "L" : "M"}${cx} ${y(v)} `; pen = true;
        dots += `<circle cx="${cx}" cy="${y(v)}" r="4.5" fill="${col}"><title>${w.start_seconds}-${w.end_seconds}s | ${key === "raw_ratio" ? "raw" : "projected"} = ${v.toFixed(2)} x threshold | speech ${w.speech_seconds}s | local SNR est. ${w.snr_estimate_db ?? "n/a"} dB | ${w.quality}</title></circle>`;
      });
      return `<path d="${d}" fill="none" stroke="${col}" stroke-width="2.4" ${dash ? 'stroke-dasharray="2 0"' : ""}/>${dots}`;
    };
    pB += line("raw_ratio", "var(--red)") + line("projected_ratio", "var(--accent)") + timeAxis(HB) + `</svg>`;
    pB += `<div class="row small legend"><span><i style="background:var(--red)"></i>Raw encoder drift</span><span><i style="background:var(--accent)"></i>Projected (invariance head) drift</span><span><i style="background:#8a5cf6;opacity:.4"></i>Model disagreement</span><span><i style="background:var(--amber);opacity:.4"></i>Known noise intervention</span></div>`;
  } else if (st === "B") {
    const mini = (key, col, label) => {
      const vals = a.windows.map((w) => w[key]).filter((v) => v != null);
      const ymax = Math.max(1e-6, ...vals) * 1.15, H = 140, y = (v) => 8 + (1 - v / ymax) * (H - 30);
      let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" width="100%">${ivBands(H - 22)}<text x="4" y="14">${label} (uncalibrated cosine distance)</text>`;
      let d = "", pen = false;
      a.windows.forEach((w) => { const v = w[key]; const cx = x((w.start_seconds + w.end_seconds) / 2); if (v == null) { pen = false; return; } d += `${pen ? "L" : "M"}${cx} ${y(v)} `; pen = true; s += `<circle cx="${cx}" cy="${y(v)}" r="4" fill="${col}"><title>${v.toExponential(2)}</title></circle>`; });
      return s + `<path d="${d}" fill="none" stroke="${col}" stroke-width="2"/>${timeAxis(H)}</svg>`;
    };
    pB = `<div class="banner">${esc(a.state_message)} The two models live on different scales, so they are drawn separately and no threshold is shown.</div>${mini("raw_distance", "var(--red)", "Raw encoder")}${mini("projected_distance", "var(--accent)", "Projected")}`;
  } else if (st === "A") {
    pB = `<div class="banner"><strong>Personal baseline not ready.</strong> ${a.baseline_progress[0]}/${a.baseline_progress[1]} usable baseline recordings. ${esc(a.baseline_note || "")} No drift is shown until the baseline is complete.</div>`;
  } else {
    pB = `<div class="banner"><strong>Recording quality insufficient for reliable drift analysis.</strong> ${esc((a.recording_reasons || []).join(", ") || "")}</div>`;
  }

  // ---- Panel D: summary (no clinical risk, no severity)
  const s = a.summary || {}, cal = a.calibration;
  const pD = `<table class="small"><tbody>
    <tr><td>Recording quality</td><td>${esc(a.recording_quality)}${a.recording_reasons && a.recording_reasons.length ? " (" + esc(a.recording_reasons.join(", ")) + ")" : ""}</td></tr>
    <tr><td>Windows analysed</td><td>${s.valid_windows} valid of ${s.total_windows} (8 s each)</td></tr>
    <tr><td>Max normalized raw drift</td><td>${fmt(s.max_raw_ratio)}</td></tr>
    <tr><td>Max normalized projected drift</td><td>${fmt(s.max_projected_ratio)}</td></tr>
    <tr><td>Raw-only threshold crossings</td><td>${s.raw_only_crossings ?? "&ndash;"} <span class="muted">(projected-only: ${s.projected_only_crossings ?? "&ndash;"}, both: ${s.both_crossings ?? "&ndash;"})</span></td></tr>
    <tr><td>Baseline</td><td>${a.baseline_ready ? "ready" : `not ready (${a.baseline_progress.join("/")})`}</td></tr>
    <tr><td>Calibration</td><td>${cal ? `${esc(cal.source)}, ${esc(cal.metric)} distance, q${Math.round(cal.quantile * 100)}, version ${esc(cal.version)}` : "not calibrated"}</td></tr>
    <tr><td>Encoder</td><td>${esc(a.encoder)}</td></tr></tbody></table>`;
  return `<div class="card">${title ? `<h2>${title}</h2>` : ""}
    <div class="small muted" style="margin-bottom:6px">${esc(a.state_message)}</div>
    ${pA}${pB}
    <p class="small muted">Shading marks <strong>model disagreement</strong>, an observation. It shows the models responded differently in that window, not that noise was removed. Only a controlled comparison (Replay lab) supports a causal reading. Windows with too little speech are left as gaps.</p>
    ${pD}</div>`;
}

async function viewLab() {
  let patients = [];
  try { patients = await api("/patients"); } catch {}
  $app.innerHTML = `<h1>Replay lab</h1>
    <div class="banner">Controlled replay: the same speech is analysed twice, as recorded and with a known noise added over a known interval. Nothing is stored as a check-in, and the audio is deleted after analysis. Results are <strong>measured model outputs</strong>, including when they don't favour the projection.</div>
    <div class="card"><div class="grid2">
      <label>Patient (whose baseline to compare against)<br><select id="lp">${patients.map((p) => `<option value="${esc(p.id)}">${esc(p.display_name)}</option>`).join("")}</select></label>
      <label>Clean speech recording (wav/webm, authorized)<br><input type="file" id="la" accept="audio/*"></label>
      <label>Noise<br><select id="lk"><option value="babble_self">Babble (from this recording, reversed)</option><option value="speech_shaped">Speech-shaped steady noise</option><option value="pink">Pink noise</option><option value="file">Upload a noise file</option></select></label>
      <label>Noise file (only for "Upload")<br><input type="file" id="ln" accept="audio/*"></label>
      <label>Noise from (s) <input id="ls" type="number" value="24" min="0" step="1" style="width:6em"> to (s) <input id="le" type="number" value="48" min="1" step="1" style="width:6em"></label>
      <label>SNR (dB) <input id="lsnr" type="number" value="5" step="1" style="width:6em"> &nbsp; seed <input id="lseed" type="number" value="0" style="width:5em"></label>
    </div><p><button class="primary" id="lrun">Run replay</button> <span class="small muted" id="lmsg"></span></p></div>
    <div id="lout"></div>`;
  document.getElementById("lrun").onclick = async () => {
    const f = document.getElementById("la").files[0], msg = document.getElementById("lmsg");
    if (!f) { msg.textContent = "Choose a clean recording first."; return; }
    const fd = new FormData();
    fd.append("patient_id", document.getElementById("lp").value); fd.append("audio", f, f.name);
    fd.append("kind", document.getElementById("lk").value);
    fd.append("start_seconds", document.getElementById("ls").value); fd.append("end_seconds", document.getElementById("le").value);
    fd.append("snr_db", document.getElementById("lsnr").value); fd.append("seed", document.getElementById("lseed").value);
    const nf = document.getElementById("ln").files[0];
    if (nf && document.getElementById("lk").value === "file") fd.append("noise_file", nf, nf.name);
    msg.textContent = "Analysing both versions...";
    try {
      const r = await fetch("/api/demo/replay", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.status);
      msg.textContent = "";
      const n = j.noise;
      document.getElementById("lout").innerHTML =
        windowChart(j.clean, "Clean recording (no noise added; dashed box = the interval the noisy version changes)") +
        windowChart(j.noisy, `Same recording + ${esc(n.kind)} noise, ${n.start_seconds}-${n.end_seconds}s at ${n.snr_db} dB SNR (seed ${n.seed})`);
    } catch (e) { msg.textContent = "Replay failed: " + e.message; }
  };
}
