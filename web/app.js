// Voice Drift frontend (UI spec v1.0). Dependency-free ES module served by FastAPI; no build step.
// All numbers shown come from the backend. Missing values are shown as missing, never as zero.
const $app = document.getElementById("app");
const DEMO = new URLSearchParams(location.search).get("demo") === "1";
const MIN_S = 20, MAX_S = 60;

// ------------------------------------------------------------------ helpers
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
async function api(path, opts) {
  const r = await fetch("/api" + path, opts);
  if (!r.ok) { const e = new Error(`${path}: ${r.status}`); e.status = r.status; throw e; }
  return r.json();
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const fmtDate = (iso) => new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
const fmtDT = (iso) => new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "&ndash;" : Number(v).toFixed(d));
const sameDay = (iso) => new Date(iso).toDateString() === new Date().toDateString();
let stored = null;
try { stored = localStorage.getItem("vd_pid"); } catch {}
// No saved choice yet: open on the (synthetic, labelled) drift profile so the demo starts on a changed voice.
const S = { pid: stored || null, health: null };
const setPid = (p) => { S.pid = p; try { localStorage.setItem("vd_pid", p); } catch {} };

// Lucide-style outline icons, drawn in-house (24x24, stroke = currentColor)
const IC = {
  home: '<path d="M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
  mic: '<rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v4M8 22h8"/>',
  analysis: '<polyline points="3 12 7 12 10 4 14 20 17 12 21 12"/>',
  trends: '<polyline points="3 17 9 11 13 15 21 7"/><polyline points="15 7 21 7 21 13"/>',
  report: '<path d="M6 2h9l5 5v15H6z"/><path d="M14 2v6h6M9 13h8M9 17h8"/>',
  settings: '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
  check: '<polyline points="4 12 10 18 20 6"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
};
const icon = (n, size = 20) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${IC[n]}</svg>`;

// Neutral wording (UI spec section 10): no risk, deterioration or diagnosis language, no alarm red.
const STATUS = {
  BASELINE: ["Building baseline", "gray", "info"],
  GREEN: ["Within your usual range", "teal", "check"],
  AMBER: ["Different from your baseline", "amber", "info"],
  RED: ["Sustained difference from baseline", "navy", "info"],
  SAFETY: ["Support resources shown", "navy", "shield"],
};
const QUALITY = { PASS: ["Good recording", "teal", "check"], WARN: ["Usable, some background noise", "amber", "info"], REJECT: ["Not usable", "gray", "x"] };
const MODES = { real_recording: "Real recording", controlled_experiment: "Controlled experiment", synthetic_demo: "Synthetic demo data" };
const pill = ([label, tone, ic]) => `<span class="pill ${tone}">${icon(ic, 14)}${esc(label)}</span>`;
const statusPill = (s) => pill(STATUS[s] || [s, "gray", "info"]);
const qualityPill = (q) => pill(QUALITY[q] || [q || "unknown", "gray", "info"]);
const modeTag = (m) => (m ? `<span class="mode ${esc(m)}">${esc(MODES[m] || m)}</span>` : "");
const ALERT_NAMES = { DRIFT: "Sustained difference from baseline", REVIEW: "Review suggested", MISSED: "Several unusable recordings", SAFETY: "Support resources shown" };

async function health() {
  if (!S.health) S.health = await api("/health").catch(() => ({}));
  return S.health;
}
const modelLabel = (h) => (h.encoder && !h.encoder.includes("fallback") ? h.encoder.replace("+", " + ") : "Fallback features (no trained model loaded)");

// ------------------------------------------------------------------ shell + routing
const NAV = [
  ["home", "Home", "#/", "home"], ["checkin", "Check-In", "#/checkin", "mic"], ["trends", "Trends", "#/trends", "trends"],
  ["results", "Results", "#/results", "analysis"], ["about", "About & Privacy", "#/about", "shield"],
];
function renderNav(active) {
  document.getElementById("sidenav").innerHTML = NAV.map(([k, label, href, ic]) => href
    ? `<a class="navlink" href="${href}" ${k === active ? 'aria-current="page"' : ""}>${icon(ic)}${label}</a>`
    : `<span class="navlink disabled" aria-disabled="true">${icon(ic)}${label}<span class="tag">Planned</span></span>`).join("");
  document.getElementById("bottomnav").innerHTML = NAV.filter(([k]) => ["home", "checkin", "trends", "results"].includes(k))
    .map(([k, label, href, ic]) => `<a href="${href}" ${k === active ? 'aria-current="page"' : ""}>${icon(ic, 22)}${label}</a>`).join("");
}
async function renderTopbar() {
  const [pts, h] = await Promise.all([api("/patients").catch(() => []), health()]);
  const ids = pts.map((p) => p.id);
  const opts = (ids.includes("p_demo") ? pts : [{ id: "p_demo", display_name: "Demo speaker (live)", data_mode: "real_recording" }, ...pts]);
  if (!S.pid) S.pid = ids.includes("p_drift") ? "p_drift" : "p_demo";
  const cur = opts.find((p) => p.id === S.pid) || opts[0];
  if (cur && cur.id !== S.pid) setPid(cur.id);
  document.getElementById("topbar").innerHTML = `
    <label for="profile" class="small">Profile</label>
    <select id="profile">${opts.map((p) => `<option value="${esc(p.id)}" ${p.id === S.pid ? "selected" : ""}>${esc(p.display_name)}</option>`).join("")}</select>
    ${modeTag(cur && cur.data_mode)}
    <span class="spacer"></span>
    <span class="pill gray" title="Speech encoder in use">${icon("analysis", 14)}${esc(modelLabel(h))}</span>
    <span class="mode">Research prototype</span>`;
  document.getElementById("sidefoot").innerHTML = `Not a diagnostic system.<br>${esc(modelLabel(h))}`;
  document.getElementById("profile").onchange = (e) => { setPid(e.target.value); route(); };
}

let cleanup = () => {};
async function route() {
  cleanup(); cleanup = () => {};
  const h = location.hash.replace(/^#/, "").split("?")[0] || "/";   // "#/trends?days=7" -> "/trends"
  const seg = h.split("/").filter(Boolean);
  const top = seg[0] || "home";
  const active = { clinician: "about", research: "about", settings: "about", windows: "results", analysis: "results" }[top] || top;
  renderNav(active);
  await renderTopbar();
  $app.innerHTML = '<p class="muted">Loading&hellip;</p>';
  try {
    if (top === "home") await viewHome();
    else if (top === "checkin") await viewCheckin();
    else if (top === "analysis" && seg[1] === "lab") await viewLab();
    else if (top === "analysis" || top === "windows") await viewAnalysis(seg[1] ? decodeURIComponent(seg[1]) : null);
    else if (top === "trends") await viewTrends();
    else if (top === "results") await viewResults();
    else if (top === "about" || top === "settings") await viewAbout();
    else if ((top === "research" || top === "clinician") && seg[1]) await viewResearchDetail(decodeURIComponent(seg[1]));
    else if (top === "research" || top === "clinician") await viewResearch();
    else $app.innerHTML = `<div class="card"><h1>Page not found</h1><a href="#/">Go to Home</a></div>`;
  } catch (e) {
    $app.innerHTML = `<div class="callout caution" role="alert"><h3>The service is unavailable</h3>
      <p>We couldn't load this page (${esc(e.message)}). Check that the Voice Drift server is running, then reload.</p></div>`;
  }
  $app.focus({ preventScroll: true });
}
window.addEventListener("hashchange", route);
route();

async function timeline(pid = S.pid, days = 365) {
  try { return await api(`/patients/${encodeURIComponent(pid)}/timeline?days=${days}`); }
  catch (e) { if (e.status === 404) return null; throw e; }
}
const scored = (c) => c.drift && c.drift.tau > 0;
const ratio = (c) => (scored(c) ? c.drift.d / c.drift.tau : null);

// ------------------------------------------------------------------ Home (patient)
async function viewHome() {
  const tl = await timeline();
  const cks = tl ? tl.checkins : [];
  const need = (tl && tl.baseline_n_required) || 5;
  const passes = cks.filter((c) => c.quality.verdict === "PASS" && c.status !== "SAFETY").length;
  const ready = !!(tl && tl.baseline);
  const today = cks.filter((c) => sameDay(c.created_at) && c.quality.verdict !== "REJECT");
  const last = cks[cks.length - 1];
  const h = await health();
  const res = await api("/results").catch(() => ({}));
  const synthetic = tl && tl.patient.data_mode === "synthetic_demo";
  const lastChips = last && (last.timing_chips || []).length ? last.timing_chips.map((x) => sentence(x.text)).join(" ") : "";
  $app.innerHTML = `
    <h1>Voice Drift</h1>
    <p class="lead" style="max-width:78ch"><strong>Changes in how someone speaks can be tracked between clinic visits with a one-minute recording at home.</strong>
      But a noisy kitchen or a new phone can change a voice recording as much as the speaker does, so Voice Drift checks whether a change comes from the speaker or from the room.</p>
    ${headlineHtml(res.aggregate)}
    ${synthetic ? `<div class="callout caution" style="margin:16px 0 20px"><h3>Synthetic demo profile</h3><p class="small">Every value on this profile is generated for the demo; the last two days simulate deliberately slower, flatter speech. Switch profile at the top to see real recordings.</p></div>` : ""}
    <div class="grid two">
      <section class="card hero stack" aria-labelledby="ci-h">
        <h2 id="ci-h">Daily check-in</h2>
        <p>${today.length ? `${icon("check", 18)} Today's check-in is done. You can record another if you like.` : "You haven't checked in today."}</p>
        <p class="muted small">About 1 minute. Find a quiet spot, then talk naturally about your day: how you slept, your energy, what you're looking forward to.</p>
        <a class="btn primary big" href="#/checkin">${icon("mic", 22)}Start voice check-in</a>
      </section>
      <section class="card stack" aria-labelledby="bl-h">
        <h2 id="bl-h">Your baseline</h2>
        <div class="dots" aria-hidden="true">${Array.from({ length: need }, (_, i) => `<span class="${i < Math.min(passes, need) ? "on" : ""}"></span>`).join("")}</div>
        <p><strong>${Math.min(passes, need)} of ${need}</strong> good recordings${ready ? " &middot; baseline ready" : ""}</p>
        <p class="small muted">${ready
          ? `Set on ${fmtDate(tl.baseline.frozen_at)}. New check-ins are compared with these first ${need} recordings, using a boundary based on normal day-to-day variation.`
          : `Your first ${need} good recordings set your personal baseline. Until then, no comparisons are shown.`}</p>
        ${ready && h.encoder && h.encoder.includes("fallback") ? `<p class="small callout caution">No trained speech model is loaded, so comparisons use simpler fallback features and are experimental.</p>` : ""}
      </section>
    </div>
    <div class="grid two" style="margin-top:20px">
      <section class="card" aria-labelledby="tr-h">
        <div class="row"><h2 id="tr-h">Recent voice trend</h2><span class="spacer"></span><a href="#/trends">All trends &rarr;</a></div>
        ${ready ? sessionChart(cks.slice(-14), { height: 190 }) : `<p class="muted">Your trend appears once your baseline is ready (${Math.min(passes, need)}/${need}).</p>`}
      </section>
      <section class="card stack" aria-labelledby="ra-h">
        <h2 id="ra-h">Recent activity</h2>
        ${last ? `<dl class="meta">
            <dt>Last check-in</dt><dd>${fmtDT(last.created_at)}</dd>
            <dt>Recording</dt><dd>${qualityPill(last.quality.verdict)}</dd>
            <dt>Comparison</dt><dd>${last.status === "SAFETY" ? statusPill("SAFETY") : statusPill(last.status)}</dd>
            ${lastChips ? `<dt>Speech timing</dt><dd>${esc(lastChips)}</dd>` : ""}
          </dl>
          ${last.has_window_analysis ? `<a class="btn" href="#/analysis/${encodeURIComponent(last.checkin_id)}">${icon("analysis")}View analysis</a>` : ""}`
          : `<p class="muted">No check-ins yet.</p>`}
      </section>
    </div>`;
}

// ------------------------------------------------------------------ headline number (from results/aggregate.json only)
const pct = (r) => (r && r.rate != null ? `${Math.round(r.rate * 100)}%` : "&ndash;");
const ci = (r) => (r && r.ci && r.ci[0] != null ? `${Math.round(r.ci[0] * 100)}–${Math.round(r.ci[1] * 100)}%` : "");
function headlineHtml(agg) {
  const h = agg && agg.headline;
  if (!h) return `<div class="card" style="margin:6px 0 20px"><p class="small muted" style="margin:0">${icon("info", 16)} Headline result pending: the aggregate experiment hasn't been run yet. <a href="#/results">How it is measured &rarr;</a></p></div>`;
  return `<section class="card" style="margin:6px 0 20px" aria-label="Headline result"><div class="row" style="align-items:baseline">
      <div><div class="small muted">False alerts when noise is added (unchanged speech)</div>
        <div class="kpi">${pct(h.false_alert_raw)} <small>raw model</small> &rarr; ${pct(h.false_alert_projection)} <small>with the noise-invariant projection</small></div></div>
      <span class="spacer"></span>
      <div class="small"><div class="muted">Detects a controlled change in pausing</div><strong>${pct(h.change_sensitivity_raw)} raw &rarr; ${pct(h.change_sensitivity_projection)} projection</strong></div>
    </div><p class="tiny muted" style="margin:8px 0 0">${esc(h.level)}, ${agg.meta.eval_speakers} held-out speakers, ${esc(h.noise_condition)}; synthetic speech change. <a href="#/results">Evidence and limits &rarr;</a></p></section>`;
}

// ------------------------------------------------------------------ session-level chart (Home + Trends)
// One point per accepted check-in: session drift / session boundary. Never a window-level threshold.
function sessionChart(cks, { height = 240 } = {}) {
  if (!cks.length) return '<p class="muted">No check-ins in this period.</p>';
  const W = 760, H = height, L = 58, R = 12, T = 12, B = 46;
  const vals = cks.map(ratio).filter((v) => v != null);
  const ymax = Math.max(2, ...vals) * 1.1;
  const x = (i) => L + (cks.length === 1 ? 0.5 : i / (cks.length - 1)) * (W - L - R);
  const y = (v) => T + (1 - v / ymax) * (H - T - B);
  const lane = H - B + 14;
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Difference from baseline per check-in. Boundary at 1.">`;
  for (let v = 0; v <= ymax; v += ymax > 4 ? 1 : 0.5) s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  s += `<line x1="${L}" x2="${W - R}" y1="${y(1)}" y2="${y(1)}" stroke="var(--navy)" stroke-width="1.5" stroke-dasharray="6 5"/><text x="${W - R}" y="${y(1) - 6}" text-anchor="end">usual-range boundary (1)</text>`;
  let path = "", pen = false;
  cks.forEach((c, i) => {
    const cx = x(i), v = ratio(c);
    s += `<text x="${cx}" y="${H - 6}" text-anchor="middle">${fmtDate(c.created_at)}</text>`;
    if (v == null) {
      pen = false;
      const what = c.quality.verdict === "REJECT" ? "not usable" : c.status === "SAFETY" ? "support resources shown" : "baseline";
      s += c.quality.verdict === "REJECT"
        ? `<g><path d="M${cx - 5} ${lane - 5}l10 10m0-10l-10 10" stroke="var(--gray)" stroke-width="2"/><title>${fmtDate(c.created_at)}: ${what}</title></g>`
        : `<g><circle cx="${cx}" cy="${lane}" r="5" fill="none" stroke="var(--gray)" stroke-width="2"/><title>${fmtDate(c.created_at)}: ${what}</title></g>`;
      return;
    }
    path += `${pen ? "L" : "M"}${cx} ${y(v)} `; pen = true;
    const above = v > 1;
    s += above
      ? `<g><path d="M${cx} ${y(v) - 7}l7 7-7 7-7-7z" fill="var(--amber)"/><title>${fmtDate(c.created_at)}: ${v.toFixed(2)} (above boundary)${c.simulated ? ", simulated" : ""}${c.acted ? ", acted" : ""}</title></g>`
      : `<g><circle cx="${cx}" cy="${y(v)}" r="6" fill="var(--teal)"/><title>${fmtDate(c.created_at)}: ${v.toFixed(2)}${c.simulated ? ", simulated" : ""}</title></g>`;
    if (c.acted) s += `<text x="${cx}" y="${y(v) - 12}" text-anchor="middle">acted</text>`;
  });
  s += `<path d="${path}" fill="none" stroke="var(--teal)" stroke-width="2" opacity=".55"/>`;
  s += `<text x="14" y="${(H - B) / 2}" transform="rotate(-90 14 ${(H - B) / 2})" text-anchor="middle">difference (&times; boundary)</text></svg>`;
  const acted = cks.some((c) => c.acted);
  const nAbove = vals.filter((v) => v > 1).length;
  const nRej = cks.filter((c) => c.quality.verdict === "REJECT").length;
  const nBase = cks.filter((c) => ratio(c) == null && c.quality.verdict !== "REJECT").length;
  return `<figure>${s}
    <div class="legend"><span>${icon("check", 12)} <span class="box" style="background:var(--teal);border-radius:50%"></span>within boundary</span>
      <span><span class="box" style="background:var(--amber);transform:rotate(45deg)"></span>above boundary</span>
      <span>&#9675; baseline / not compared</span><span>&times; not usable</span></div>
    <p class="small muted">Vertical axis: how far each check-in's voice is from your baseline, as a multiple of the usual-range boundary (1 = the edge of normal day-to-day variation). Horizontal axis: check-in date.
      ${vals.length} compared check-in${vals.length === 1 ? "" : "s"}, ${nAbove} above the boundary, ${nBase} baseline or not compared, ${nRej} not usable. Each point is one whole check-in (session level).</p>
    ${acted ? `<p class="small"><span class="mode synthetic_demo">acted</span> marks simulated days that stand in for deliberately slower, flatter ("acted") speech. The values are generated for the demo; nobody's real speech changed.</p>` : ""}</figure>`;
}

// ------------------------------------------------------------------ Check-in (patient)
async function viewCheckin() {
  const prompt = await api("/prompt").catch(() => ({ text: "Tell me about your day so far." }));
  $app.innerHTML = `
    <h1>Daily voice check-in</h1>
    <p class="lead">About 1 minute. Speak naturally; there are no right or wrong answers.</p>
    <div class="grid two">
      <section class="card stack" style="text-align:center;align-items:center;display:flex;flex-direction:column">
        <p class="prompt"><strong>${esc(prompt.text)}</strong></p>
        <div class="ring"><svg width="190" height="190" viewBox="0 0 190 190" aria-hidden="true"><circle cx="95" cy="95" r="84" fill="none" stroke="var(--line)" stroke-width="10"/><circle id="arc" cx="95" cy="95" r="84" fill="none" stroke="var(--teal)" stroke-width="10" stroke-linecap="round" stroke-dasharray="527.8" stroke-dashoffset="527.8"/></svg><div class="t" id="clock" aria-hidden="true">1:00</div></div>
        <div class="meter" role="meter" aria-label="Microphone level" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" id="lvlwrap"><div id="lvl"></div></div>
        <p id="hint" class="small" aria-live="polite">&nbsp;</p>
        <div class="row" style="justify-content:center">
          <button class="primary big" id="rec">${icon("mic", 22)}Start recording</button>
          <button id="stop" disabled>Stop &amp; submit (after 0:20)</button>
          <button id="cancel" disabled>Cancel</button>
        </div>
        ${DEMO ? `<div class="row"><button id="noisemark" disabled>Mark noise start</button><span class="small muted" id="noiselog">Demo: mark when background noise starts and stops.</span></div>` : ""}
        <p id="status" role="status" class="small muted">Ready. When you press Start, your browser will ask to use the microphone.</p>
      </section>
      <aside class="card stack">
        <h2>Before you start</h2>
        <ul class="small">
          <li>Find a quiet place, away from TV or conversations.</li>
          <li>Hold the device about an arm's length away, or use a headset.</li>
          <li>Speak for up to 60 seconds. You can stop after 20 seconds.</li>
        </ul>
        <div class="callout info small"><h3>${icon("shield", 16)} Your recording</h3>
          Audio is sent only to this computer's Voice Drift server, analysed there, and the audio file is deleted afterwards.
          A written transcript may be sent to a language-model service to find quotes. See <a href="#/about">About &amp; Privacy</a>.</div>
      </aside>
    </div>`;
  const $ = (id) => document.getElementById(id);
  const rec = $("rec"), stop = $("stop"), cancel = $("cancel"), arc = $("arc"), clock = $("clock"), lvl = $("lvl"), hint = $("hint"), status = $("status");
  let stream, mr, chunks = [], ctx, raf, t0, timer, stopped = false;
  const marks = [];
  const noiseBtn = $("noisemark");
  const elapsed = () => (t0 ? (performance.now() - t0) / 1000 : 0);
  const say = (t) => { status.textContent = t; };
  if (noiseBtn) noiseBtn.onclick = () => {
    const open = marks.length && marks[marks.length - 1].end_seconds == null;
    if (open) { marks[marks.length - 1].end_seconds = +elapsed().toFixed(2); noiseBtn.textContent = "Mark noise start"; }
    else { marks.push({ type: "controlled_background_noise", start_seconds: +elapsed().toFixed(2), end_seconds: null, source: "live_marker" }); noiseBtn.textContent = "Mark noise end"; }
    $("noiselog").textContent = marks.map((m) => `noise ${m.start_seconds}s to ${m.end_seconds ?? "..."}s`).join(", ");
  };
  const finish = () => { cancelAnimationFrame(raf); clearInterval(timer); stream?.getTracks().forEach((t) => t.stop()); ctx?.close(); };
  cleanup = () => { stopped = true; finish(); if (mr && mr.state !== "inactive") mr.stop(); };

  rec.onclick = async () => {
    if (!navigator.mediaDevices?.getUserMedia) { showMicProblem("unsupported"); return; }
    rec.disabled = true;
    say("Waiting for microphone permission. Choose “Allow” in your browser's prompt so we can record your check-in.");
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } }); }
    catch (e) { rec.disabled = false; showMicProblem(e.name); return; }
    cancel.disabled = false; if (noiseBtn) noiseBtn.disabled = false;
    ctx = new AudioContext();
    const an = ctx.createAnalyser(); an.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(an);
    const buf = new Float32Array(an.fftSize);
    const wrap = $("lvlwrap");
    const tick = () => {
      an.getFloatTimeDomainData(buf);
      let s2 = 0, pk = 0; for (const v of buf) { s2 += v * v; pk = Math.max(pk, Math.abs(v)); }
      const db = 20 * Math.log10(Math.sqrt(s2 / buf.length) + 1e-9);
      const pct = Math.max(0, Math.min(1, (db + 60) / 50));
      lvl.style.width = pct * 100 + "%"; wrap.setAttribute("aria-valuenow", Math.round(pct * 100));
      lvl.style.background = pk > 0.97 ? "var(--coral)" : db < -50 ? "var(--amber)" : "var(--teal)";
      hint.textContent = pk > 0.97 ? "Too loud: move back a little." : db < -52 ? "Too quiet: speak up or move closer." : "";
      raf = requestAnimationFrame(tick);
    };
    tick();
    mr = new MediaRecorder(stream, MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? { mimeType: "audio/webm;codecs=opus" } : undefined);
    mr.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    mr.onstop = () => { finish(); if (!stopped) upload(new Blob(chunks, { type: mr.mimeType || "audio/webm" })); };
    mr.start(250);
    t0 = performance.now();
    say("Recording. Speak naturally.");
    status.insertAdjacentHTML("afterbegin", '<span class="recdot" aria-hidden="true"></span> ');
    timer = setInterval(() => {
      const el = elapsed(), left = Math.max(0, MAX_S - el);
      clock.textContent = `${Math.floor(left / 60)}:${String(Math.ceil(left) % 60).padStart(2, "0")}`;
      arc.style.strokeDashoffset = 527.8 * (1 - el / MAX_S);
      stop.disabled = el < MIN_S;
      stop.textContent = el < MIN_S ? `Stop & submit (after 0:${String(Math.ceil(MIN_S - el)).padStart(2, "0")})` : "Stop & submit";
      if (el >= MAX_S) mr.stop();
    }, 200);
  };
  stop.onclick = () => mr && mr.state !== "inactive" && mr.stop();
  cancel.onclick = () => { stopped = true; if (mr && mr.state !== "inactive") mr.stop(); finish(); chunks = []; viewCheckin(); };

  function showMicProblem(kind) {
    const msg = {
      NotAllowedError: ["Microphone access was blocked", "Allow microphone access for this site: select the lock or site-settings icon in your browser's address bar, set Microphone to Allow, then reload this page."],
      NotFoundError: ["No microphone found", "Connect a microphone or headset, then try again."],
      NotReadableError: ["The microphone is in use", "Close other apps that are using the microphone (calls, recorders), then try again."],
      unsupported: ["Recording isn't available here", "Open Voice Drift in an up-to-date browser on http://127.0.0.1 or an HTTPS address."],
    }[kind] || ["We couldn't start the microphone", "Check your microphone and try again."];
    say(msg[0] + ".");
    $app.querySelector(".card").insertAdjacentHTML("beforeend", `<div class="callout caution" role="alert"><h3>${msg[0]}</h3><p class="small">${msg[1]}</p></div>`);
  }

  async function upload(blob) {
    const fd = new FormData();
    fd.append("audio", blob, "checkin.webm"); fd.append("prompt_id", "default");
    const end = +elapsed().toFixed(2);
    marks.forEach((m) => { if (m.end_seconds == null) m.end_seconds = end; });
    if (marks.length) fd.append("interventions", JSON.stringify(marks));
    $app.innerHTML = `<h1>Checking your recording</h1><div class="card"><ul class="steps" id="steps"></ul><p id="pstatus" role="status" class="small muted">Uploading&hellip;</p></div>`;
    try {
      const r = await fetch(`/api/patients/${encodeURIComponent(S.pid)}/checkins`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(r.status);
      watch((await r.json()).checkin_id);
    } catch (e) {
      renderOutcome({ outcome: "failed", message: "The upload didn't reach the server, so this recording was not saved. Please check that Voice Drift is running and record again." });
    }
  }
}

const STEPS = [["quality", "Checking recording quality"], ["transcript", "Transcribing"], ["scores", "Comparing with your baseline"], ["evidence", "Reading the transcript"]];
function watch(cid) {
  const ul = document.getElementById("steps");
  ul.innerHTML = STEPS.map(([k, l]) => `<li id="s_${k}"><span aria-hidden="true">&#9675;</span><span>${l}</span><span class="ms"></span></li>`).join("");
  const es = new EventSource(`/api/checkins/${cid}/events?role=patient`);
  cleanup = () => es.close();
  es.addEventListener("stage", (e) => {
    const d = JSON.parse(e.data), li = document.getElementById("s_" + d.stage);
    if (!li) return;
    li.firstElementChild.outerHTML = icon("check", 18);
    li.querySelector(".ms").textContent = d.ms ? (d.ms / 1000).toFixed(1) + " s" : "skipped";
    document.getElementById("pstatus").textContent = `${li.children[1].textContent}: done`;
  });
  es.addEventListener("result", (e) => { es.close(); renderOutcome(JSON.parse(e.data)); });
}

function renderOutcome(r) {
  if (r.outcome === "safety") {
    $app.innerHTML = `<div class="support" role="alert"><p class="big">You are not alone.</p><p>${esc(r.safety.message)}</p>
      <p class="big"><a href="tel:988">Call or text 988</a></p><p>Please also contact your clinician today.</p></div>`;
    return;
  }
  if (r.outcome === "rejected" || r.outcome === "failed") {
    $app.innerHTML = `<h1>${r.outcome === "failed" ? "This check-in couldn't be analysed" : "Let's try that again"}</h1>
      <div class="callout caution" role="alert"><h3>${r.outcome === "failed" ? "Something went wrong" : "We couldn't use this recording"}</h3><p>${esc(r.message)}</p></div>
      <p style="margin-top:16px"><a class="btn primary big" href="#/checkin" onclick="setTimeout(()=>dispatchEvent(new HashChangeEvent('hashchange')))">${icon("mic", 22)}Record again</a></p>`;
    return;
  }
  const prog = r.baseline_progress;
  $app.innerHTML = `<h1>${icon("check", 26)} Check-in recorded</h1>
    <div class="card stack">
      <p>${esc(r.message || "")}</p>
      ${prog ? `<div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="${prog[1]}" aria-valuenow="${prog[0]}" aria-label="Baseline progress"><div style="width:${(prog[0] / prog[1]) * 100}%"></div></div><p class="small muted">Baseline ${prog[0]} of ${prog[1]}</p>` : ""}
      <div class="row">
        ${r.has_window_analysis ? `<a class="btn primary" href="#/analysis/${encodeURIComponent(r.checkin_id)}">${icon("analysis")}View analysis</a>` : ""}
        <a class="btn" href="#/trends">${icon("trends")}View trends</a><a class="btn" href="#/">Home</a>
      </div>
      ${r.simulated ? '<span class="mode synthetic_demo">Synthetic demo data</span>' : ""}
    </div>`;
}

// ------------------------------------------------------------------ Analysis Lab
async function viewAnalysis(cid) {
  const tl = await timeline();
  const recs = (tl ? tl.checkins : []).filter((c) => c.has_window_analysis && c.status !== "SAFETY");
  const tabs = `<nav class="tabs" aria-label="Analysis sources"><a href="#/analysis" aria-current="page">Your recordings</a><a href="#/analysis/lab">Controlled experiment</a></nav>`;
  if (!recs.length) {
    $app.innerHTML = `<h1>Voice Analysis Lab</h1>${tabs}<div class="card"><h2>No analysed recordings yet</h2>
      <p class="muted">Each new check-in on this profile is analysed window by window. ${tl && tl.patient.data_mode === "synthetic_demo" ? "Synthetic demo profiles have no window-level data." : ""}
      You can also run a <a href="#/analysis/lab">controlled experiment</a> with a prepared recording.</p></div>`;
    return;
  }
  const sel = recs.find((c) => c.checkin_id === cid) || recs[recs.length - 1];
  const a = await api(`/checkins/${encodeURIComponent(sel.checkin_id)}/windows`);
  $app.innerHTML = `<h1>Voice Analysis Lab</h1>
    <p class="lead">Does a change come from the speaker or from the recording environment? The same recording is scored by two models, window by window.</p>
    ${tabs}
    <div class="row" style="margin-bottom:16px"><label for="recsel">Recording</label>
      <select id="recsel">${recs.map((c) => `<option value="${esc(c.checkin_id)}" ${c.checkin_id === sel.checkin_id ? "selected" : ""}>${fmtDT(c.created_at)}</option>`).join("")}</select></div>
    ${analysisBlock(a, { key: "rec", recordingLabel: fmtDT(a.created_at) })}`;
  document.getElementById("recsel").onchange = (e) => { location.hash = `#/analysis/${encodeURIComponent(e.target.value)}`; };
  wireAnalysis(a, "rec");
}

function analysisBlock(a, { key, title = "", recordingLabel = "" }) {
  const cal = a.calibration || {}, s = a.summary || {}, m = a.models || {};
  const calibrated = a.state === "C" || a.state === "E";
  const wins = a.windows || [];
  const nRaw = wins.filter((w) => w.raw_ratio > 1).length, nProj = wins.filter((w) => w.projected_ratio > 1).length;
  const maxDist = (k) => Math.max(...wins.map((w) => w[k]).filter((v) => v != null));
  const summaryCard = (name, tone, ratioMax, n, distKey) => `<div class="card"><div class="small muted">${name}</div>
      ${calibrated ? `<div class="kpi">${fmt(ratioMax)}<small> &times; its boundary (max)</small></div><div class="small">Above its boundary in <strong>${n}</strong> of ${s.valid_windows} usable windows</div>`
                   : `<div class="kpi">${Number.isFinite(maxDist(distKey)) ? maxDist(distKey).toExponential(2) : "&ndash;"}<small> max distance</small></div><div class="small muted">Uncalibrated: no boundary crossings reported</div>`}</div>`;
  return `<section class="stack" aria-label="${esc(title || "Recording analysis")}">
    ${title ? `<h2>${esc(title)}</h2>` : ""}
    ${!m.projection_trained ? `<div class="callout caution small"><h3>No trained projection loaded</h3>The second line uses a classical noise-floor subtraction instead of the trained noise-invariant projection. Treat the comparison as a placeholder.</div>` : ""}
    <div class="grid three">
      <div class="card"><h3>Context</h3><dl class="meta small">
        <dt>Recording</dt><dd>${esc(recordingLabel || "Prepared recording")}</dd>
        <dt>Duration</dt><dd>${fmt(a.duration_seconds, 0)} s (${wins.length} windows of 8 s)</dd>
        <dt>Data</dt><dd>${modeTag(a.data_mode)}</dd>
        <dt>Baseline</dt><dd>${a.baseline && a.baseline.ready ? `ready (${esc(a.baseline.version || "")})` : `not ready (${(a.baseline_progress || [0, 5]).join("/")})`}</dd>
        <dt>Window boundaries</dt><dd>${cal.ready ? `calibrated (${esc(cal.source || "")})` : "not calibrated"}</dd>
      </dl></div>
      ${summaryCard("Raw WavLM drift", "coral", s.max_raw_ratio, nRaw, "raw_distance")}
      ${summaryCard("Noise-invariant projection drift", "teal", s.max_projected_ratio, nProj, "projected_distance")}
    </div>
    <div class="card">
      <div class="row" role="group" aria-label="Chart display options">
        <label class="check"><input type="checkbox" id="${key}_raw" checked> Raw WavLM</label>
        <label class="check"><input type="checkbox" id="${key}_proj" checked> Projected</label>
        <label class="check"><input type="checkbox" id="${key}_iv" checked> Known noise intervals</label>
      </div>
      <div id="${key}_chart"></div>
      <div class="callout info small" style="margin-top:12px" id="${key}_explain"></div>
    </div>
    <details class="card"><summary>Window-by-window values (table)</summary><div class="tablewrap">${windowTable(a)}</div>
      <p class="small muted">${esc(a.remainder_rule || "")}. Each window is compared with your baseline's overall reference, not with the same seconds of an earlier recording, because different days contain different words.</p></details>
    <details class="card"><summary>Calibration and model details</summary><dl class="meta small">
      <dt>Encoder</dt><dd>${esc(m.encoder || a.encoder)}</dd><dt>Projection</dt><dd>${esc(m.projection || "")}</dd>
      <dt>Distance</dt><dd>cosine, per 8 s window</dd>
      <dt>Raw window boundary</dt><dd>${cal.raw_threshold != null ? cal.raw_threshold : "not calibrated"}</dd>
      <dt>Projected window boundary</dt><dd>${cal.projected_threshold != null ? cal.projected_threshold : "not calibrated"}</dd>
      <dt>Calibration source</dt><dd>${esc(cal.source || "none")}${cal.quantile ? `, ${Math.round(cal.quantile * 100)}th percentile of normal variation` : ""}</dd>
      <dt>Calibration version</dt><dd>${esc(cal.version || "&ndash;")}</dd><dt>Baseline version</dt><dd>${esc((a.baseline || {}).version || "&ndash;")}</dd>
      <dt>Thresholds during this recording</dt><dd>fixed (frozen when it was analysed)</dd></dl></details>
  </section>`;
}

function wireAnalysis(a, key) {
  const box = document.getElementById(`${key}_chart`), ex = document.getElementById(`${key}_explain`);
  const draw = () => {
    const show = { raw: document.getElementById(`${key}_raw`).checked, proj: document.getElementById(`${key}_proj`).checked, iv: document.getElementById(`${key}_iv`).checked };
    box.innerHTML = driftChart(a, show, key);
  };
  ["raw", "proj", "iv"].forEach((k) => (document.getElementById(`${key}_${k}`).onchange = draw));
  draw();
  ex.innerHTML = explain(a);
}

// Plain-language interpretation built only from the returned numbers (spec 5.3 F). No health statements.
function explain(a) {
  const wins = a.windows || [], valid = wins.filter((w) => w.raw_ratio != null);
  const gaps = wins.filter((w) => w.quality === "REJECT").length;
  const t = (w) => `${w.start_seconds}–${w.end_seconds} s`;
  const out = [];
  if (a.state === "A") out.push(`Your personal baseline isn't ready (${(a.baseline_progress || []).join("/")} usable recordings), so no comparison is made. ${esc(a.baseline_note || "")}`);
  else if (a.state === "D") out.push(`Recording quality was insufficient for this comparison${a.recording_reasons && a.recording_reasons.length ? ` (${esc(a.recording_reasons.join(", "))})` : ""}. No scores are reported.`);
  else if (a.state === "B") out.push("Window boundaries are not calibrated yet, so no boundary crossings are reported. Distances are drawn on separate scales because the two models measure distance differently.");
  else {
    const rawX = valid.filter((w) => w.raw_ratio > 1), projX = valid.filter((w) => w.projected_ratio > 1);
    const rawOnly = valid.filter((w) => w.raw_ratio > 1 && w.projected_ratio <= 1), projOnly = valid.filter((w) => w.projected_ratio > 1 && w.raw_ratio <= 1);
    out.push(`The raw model exceeded its boundary in <strong>${rawX.length}</strong> of ${valid.length} usable windows; the projected model exceeded its own boundary in <strong>${projX.length}</strong>.`);
    if (rawOnly.length) out.push(`Raw-only threshold crossing in ${rawOnly.length} window${rawOnly.length > 1 ? "s" : ""} (${rawOnly.map(t).join(", ")}).`);
    if (projOnly.length) out.push(`Projected-only threshold crossing in ${projOnly.length} window${projOnly.length > 1 ? "s" : ""} (${projOnly.map(t).join(", ")}).`);
    const ivs = (a.interventions || []).filter((iv) => iv.type !== "reference_interval_no_noise");
    if (ivs.length && rawOnly.length) {
      const ov = rawOnly.filter((w) => ivs.some((iv) => w.start_seconds < iv.end_seconds && w.end_seconds > iv.start_seconds)).length;
      out.push(`${ov} of the raw-only windows overlap ${ivs.length > 1 ? "intervals" : "the interval"} marked as background noise (${ivs.map((iv) => `${iv.start_seconds}–${iv.end_seconds} s`).join(", ")}).`);
    }
    if (!rawX.length && !projX.length) out.push("Neither model crossed its boundary in this recording.");
  }
  if (gaps) out.push(`${gaps} window${gaps > 1 ? "s" : ""} had too little usable speech and ${gaps > 1 ? "are" : "is"} shown as a gap.`);
  out.push("A disagreement shows the models responded differently, not that noise was removed. These are properties of the model scores, not health measurements.");
  return out.map((p) => `<p style="margin:4px 0">${p}</p>`).join("");
}

function windowTable(a) {
  const obs = (w) => w.raw_ratio == null ? (w.quality === "REJECT" ? `Not usable: ${esc((w.reasons || []).join(", "))}` : "Not compared")
    : w.raw_ratio > 1 && w.projected_ratio <= 1 ? "Raw-only threshold crossing"
    : w.projected_ratio > 1 && w.raw_ratio <= 1 ? "Projected-only threshold crossing"
    : w.raw_ratio > 1 ? "Both above boundary" : "Both within boundary";
  return `<table><caption class="sr-only">Window-level values for this recording</caption><thead><tr>
    <th scope="col">Window</th><th scope="col">Usable</th><th scope="col" class="num">Speech (s)</th><th scope="col" class="num">Local SNR est. (dB)</th>
    <th scope="col" class="num">Raw (&times; boundary)</th><th scope="col" class="num">Projected (&times; boundary)</th><th scope="col">Observation</th></tr></thead><tbody>
    ${(a.windows || []).map((w) => `<tr><td>${w.start_seconds}–${w.end_seconds} s</td><td>${w.quality === "REJECT" ? "No" : w.quality === "WARN" ? "Yes, noisy" : "Yes"}</td>
      <td class="num">${fmt(w.speech_seconds, 1)}</td><td class="num">${w.snr_estimate_db == null ? "&ndash;" : fmt(w.snr_estimate_db, 1)}</td>
      <td class="num">${w.raw_ratio == null ? (w.raw_distance == null ? "&ndash;" : `${w.raw_distance.toExponential(2)} (uncal.)`) : fmt(w.raw_ratio)}</td>
      <td class="num">${w.projected_ratio == null ? (w.projected_distance == null ? "&ndash;" : `${w.projected_distance.toExponential(2)} (uncal.)`) : fmt(w.projected_ratio)}</td>
      <td>${obs(w)}</td></tr>`).join("")}</tbody></table>`;
}

// Panel A (audio timeline) + Panel B (normalized drift) on one shared time axis.
function driftChart(a, show, key) {
  const W = 900, L = 48, R = 16, HA = 100, HB = 250, dur = Math.max(a.duration_seconds, 1);
  const x = (t) => L + (t / dur) * (W - L - R);
  const wins = a.windows || [];
  const ivs = show.iv ? a.interventions || [] : [];
  const defs = `<defs><pattern id="${key}_hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="8" stroke="var(--amber)" stroke-width="3" opacity=".45"/></pattern></defs>`;
  const ivBands = (h) => ivs.map((iv) => iv.type === "reference_interval_no_noise"
    ? `<rect x="${x(iv.start_seconds)}" y="1" width="${x(iv.end_seconds) - x(iv.start_seconds)}" height="${h - 2}" fill="none" stroke="var(--amber)" stroke-dasharray="5 4"><title>Same interval as the noisy version (no noise added here)</title></rect>`
    : `<rect x="${x(iv.start_seconds)}" y="0" width="${x(iv.end_seconds) - x(iv.start_seconds)}" height="${h}" fill="url(#${key}_hatch)"><title>Known noise interval ${iv.start_seconds}–${iv.end_seconds} s${iv.snr_db != null ? ` at ${iv.snr_db} dB` : ""}</title></rect>`).join("");
  const winLines = (h) => wins.map((w) => `<line x1="${x(w.start_seconds)}" x2="${x(w.start_seconds)}" y1="0" y2="${h}" stroke="var(--line)"/>`).join("");
  const axis = (h) => { let s = ""; for (let t = 0; t <= dur; t += dur > 40 ? 10 : 5) s += `<text x="${x(t)}" y="${h - 4}" text-anchor="middle">${t} s</text>`; return s; };

  const env = a.envelope || { hz: 10, db: [], speech: [] };
  const ey = (db) => 8 + (1 - (Math.max(-70, Math.min(0, db)) + 70) / 70) * (HA - 34);
  let pA = `<svg class="chart" viewBox="0 0 ${W} ${HA}" role="img" aria-label="Audio timeline: loudness, detected speech, 8-second windows${ivs.length ? ", and known noise intervals" : ""}">${defs}${ivBands(HA - 20)}`;
  env.speech.forEach((sp, i) => { if (sp) pA += `<rect x="${x(i / env.hz)}" y="4" width="${x(1 / env.hz) - x(0) + 0.5}" height="${HA - 30}" fill="var(--mint)"/>`; });
  if (env.db.length) pA += `<path d="M${x(0)} ${ey(-70)} ${env.db.map((d, i) => `L${x(i / env.hz)} ${ey(d)}`).join(" ")} L${x(env.db.length / env.hz)} ${ey(-70)}Z" fill="var(--gray)" opacity=".55"/>`;
  wins.forEach((w) => { if (w.quality !== "PASS") pA += `<rect x="${x(w.start_seconds)}" y="${HA - 25}" width="${x(w.end_seconds) - x(w.start_seconds)}" height="5" fill="${w.quality === "REJECT" ? "var(--gray)" : "var(--amber)"}"><title>${w.quality === "REJECT" ? "Not usable: " + esc((w.reasons || []).join(", ")) : "Usable, low local SNR estimate"}</title></rect>`; });
  pA += winLines(HA - 22) + axis(HA) + `<text x="4" y="14">audio</text></svg>`;

  let pB = "";
  if (a.state === "C" || a.state === "E") {
    const valid = wins.filter((w) => w.raw_ratio != null);
    const peak = Math.max(2, ...valid.flatMap((w) => [show.raw ? w.raw_ratio : 0, show.proj ? w.projected_ratio : 0]));
    const logy = peak > 6, ymax = logy ? Math.pow(10, Math.ceil(Math.log10(peak * 1.1))) : peak * 1.12;
    const y = logy ? (v) => 12 + (1 - (Math.log10(Math.max(v, 0.1)) + 1) / (Math.log10(ymax) + 1)) * (HB - 44) : (v) => 12 + (1 - v / ymax) * (HB - 44);
    pB = `<svg class="chart" viewBox="0 0 ${W} ${HB}" role="img" aria-label="Raw versus noise-invariant drift per window, each relative to its own boundary at 1">${defs}${ivBands(HB - 28)}`;
    wins.forEach((w) => {
      if (w.raw_ratio == null) return;
      const rawOnly = w.raw_ratio > 1 && w.projected_ratio <= 1, projOnly = w.projected_ratio > 1 && w.raw_ratio <= 1;
      if (rawOnly || projOnly) pB += `<rect x="${x(w.start_seconds)}" y="0" width="${x(w.end_seconds) - x(w.start_seconds)}" height="${HB - 28}" fill="var(--navy)" opacity="${rawOnly ? 0.08 : 0.04}"/><text x="${x((w.start_seconds + w.end_seconds) / 2)}" y="11" text-anchor="middle" style="fill:var(--navy)">${rawOnly ? "raw-only" : "projected-only"}</text>`;
    });
    const ticks = logy ? [0.1, 1, 10, 100, 1000].filter((v) => v <= ymax) : Array.from({ length: Math.floor(ymax / (ymax > 4 ? 1 : 0.5)) + 1 }, (_, i) => i * (ymax > 4 ? 1 : 0.5));
    ticks.forEach((v) => { pB += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`; });
    if (logy) pB += `<text x="${L + 4}" y="24">log scale</text>`;
    pB += winLines(HB - 28);
    pB += `<line x1="${L}" x2="${W - R}" y1="${y(1)}" y2="${y(1)}" stroke="var(--navy)" stroke-width="2" stroke-dasharray="8 5"/><text x="${W - R}" y="${y(1) - 6}" text-anchor="end" style="fill:var(--navy)">boundary = 1 (fixed)</text>`;
    const series = (k, col, dash, marker, label) => {
      let d = "", pen = false, dots = "";
      wins.forEach((w) => {
        const v = w[k], cx = x((w.start_seconds + w.end_seconds) / 2);
        if (v == null) { pen = false; return; }
        d += `${pen ? "L" : "M"}${cx} ${y(v)} `; pen = true;
        const tip = `<title>${w.start_seconds}–${w.end_seconds} s: ${label} ${v.toFixed(2)} × its boundary</title>`;
        dots += marker === "square" ? `<rect x="${cx - 5}" y="${y(v) - 5}" width="10" height="10" fill="${col}">${tip}</rect>` : `<circle cx="${cx}" cy="${y(v)}" r="5.5" fill="${col}">${tip}</circle>`;
      });
      return `<path d="${d}" fill="none" stroke="${col}" stroke-width="2.6" ${dash ? `stroke-dasharray="${dash}"` : ""}/>${dots}`;
    };
    if (show.raw) pB += series("raw_ratio", "var(--coral)", "7 5", "square", "raw");
    if (show.proj) pB += series("projected_ratio", "var(--teal)", "", "circle", "projected");
    pB += axis(HB) + `<text x="4" y="${HB / 2}" transform="rotate(-90 12 ${HB / 2})">&times; boundary</text></svg>`;
    pB += `<div class="legend"><span><span class="sw" style="border-top:3px dashed var(--coral)"></span>&#9632; Raw WavLM (dashed)</span><span><span class="sw" style="border-top:3px solid var(--teal)"></span>&#9679; Projected (solid)</span>
      <span><span class="box" style="background:var(--navy);opacity:.15"></span>Model disagreement</span>${ivs.length ? `<span><span class="box" style="background:url(); background-color:#F6E7C9"></span>Known noise interval (hatched)</span>` : ""}</div>`;
  } else if (a.state === "B") {
    const mini = (k, col, label) => {
      const vals = wins.map((w) => w[k]).filter((v) => v != null), H = 150;
      if (!vals.length) return "";
      const ymax = Math.max(...vals) * 1.15 || 1, y = (v) => 10 + (1 - v / ymax) * (H - 34);
      let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${label}, uncalibrated distance">${defs}${ivBands(H - 22)}<text x="4" y="14">${label} (uncalibrated distance)</text>`, d = "", pen = false;
      wins.forEach((w) => { const v = w[k], cx = x((w.start_seconds + w.end_seconds) / 2); if (v == null) { pen = false; return; } d += `${pen ? "L" : "M"}${cx} ${y(v)} `; pen = true; s += `<circle cx="${cx}" cy="${y(v)}" r="4.5" fill="${col}"><title>${v.toExponential(2)}</title></circle>`; });
      return s + `<path d="${d}" fill="none" stroke="${col}" stroke-width="2"/>${axis(H)}</svg>`;
    };
    pB = `${show.raw ? mini("raw_distance", "var(--coral)", "Raw WavLM") : ""}${show.proj ? mini("projected_distance", "var(--teal)", "Projected") : ""}`;
  }
  return `<figure><figcaption>Raw vs. Noise-Invariant Speech Drift</figcaption>${pA}${pB}
    <p class="small muted">Time within this recording (s). Y-axis: each model's distance from your baseline divided by its own window boundary, so one fixed line at 1 serves both.</p></figure>`;
}

async function viewLab() {
  $app.innerHTML = `<h1>Voice Analysis Lab</h1>
    <nav class="tabs" aria-label="Analysis sources"><a href="#/analysis">Your recordings</a><a href="#/analysis/lab" aria-current="page">Controlled experiment</a></nav>
    ${await labFormHtml()}`;
  wireLab();
}

async function labFormHtml() {
  let pts = [];
  try { pts = await api("/patients"); } catch {}
  const opts = pts.length ? pts : [{ id: S.pid, display_name: S.pid }];
  return `<div class="callout info small" style="margin-bottom:16px"><h3>Controlled experiment</h3>The same speech is analysed twice: as recorded, and with a known noise added over a known interval at a known level.
      Nothing is stored as a check-in, and the audio is deleted after analysis. Results are measured model outputs, including when they don't favour the projection.</div>
    <form class="card" id="labform"><div class="grid half">
      <div class="stack"><label for="lp">Compare against the baseline of</label><select id="lp">${opts.map((p) => `<option value="${esc(p.id)}" ${p.id === S.pid ? "selected" : ""}>${esc(p.display_name)}</option>`).join("")}</select></div>
      <div class="stack"><label for="la">Clean speech recording (wav or webm, with consent)</label><input type="file" id="la" accept="audio/*" required></div>
      <div class="stack"><label for="lk">Noise</label><select id="lk"><option value="babble_self">Babble (made from this recording, reversed)</option><option value="speech_shaped">Speech-shaped steady noise</option><option value="pink">Pink noise</option><option value="file">Upload a noise file</option></select></div>
      <div class="stack"><label for="ln">Noise file (only for "Upload")</label><input type="file" id="ln" accept="audio/*"></div>
      <div class="row"><label for="ls">Noise from (s)</label><input id="ls" type="number" value="24" min="0" step="1" style="width:6em"><label for="le">to (s)</label><input id="le" type="number" value="48" min="1" step="1" style="width:6em"></div>
      <div class="row"><label for="lsnr">Level (SNR, dB)</label><input id="lsnr" type="number" value="10" step="1" style="width:6em"><label for="lseed">Seed</label><input id="lseed" type="number" value="0" style="width:5em"></div>
    </div><p class="row"><button class="primary" type="submit">Run comparison</button><span class="small muted" id="lmsg" role="status"></span></p></form>
    <div id="lout" class="stack" style="margin-top:20px"></div>`;
}

function wireLab() {
  document.getElementById("labform").onsubmit = async (ev) => {
    ev.preventDefault();
    const f = document.getElementById("la").files[0], msg = document.getElementById("lmsg");
    if (!f) { msg.textContent = "Choose a clean recording first."; return; }
    const fd = new FormData();
    fd.append("patient_id", document.getElementById("lp").value); fd.append("audio", f, f.name);
    fd.append("kind", document.getElementById("lk").value);
    fd.append("start_seconds", document.getElementById("ls").value); fd.append("end_seconds", document.getElementById("le").value);
    fd.append("snr_db", document.getElementById("lsnr").value); fd.append("seed", document.getElementById("lseed").value);
    const nf = document.getElementById("ln").files[0];
    if (nf && document.getElementById("lk").value === "file") fd.append("noise_file", nf, nf.name);
    msg.textContent = "Analysing both versions…";
    try {
      const r = await fetch("/api/demo/replay", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.status);
      msg.textContent = "Done.";
      const n = j.noise;
      document.getElementById("lout").innerHTML =
        analysisBlock(j.noisy, { key: "noisy", title: `With ${n.kind.replace("_", " ")} noise, ${n.start_seconds}–${n.end_seconds} s at ${n.snr_db} dB (seed ${n.seed})`, recordingLabel: f.name + " + noise" }) +
        analysisBlock(j.clean, { key: "clean", title: "Same recording, no noise added (dashed box = the interval changed above)", recordingLabel: f.name });
      wireAnalysis(j.noisy, "noisy"); wireAnalysis(j.clean, "clean");
    } catch (e) { msg.textContent = "The comparison failed: " + e.message; }
  };
}

// ------------------------------------------------------------------ Results (judge-facing evidence page)
const COND_LABEL = (agg, c) => (agg.conditions && agg.conditions[c]) || c;
function verdict(h) {
  const fr = h.false_alert_raw, fp = h.false_alert_projection, sr = h.change_sensitivity_raw, sp = h.change_sensitivity_projection;
  if (!fr || fr.rate == null || !fp || fp.rate == null) return "";
  const overlap = (a, b) => a.ci && b.ci && a.ci[0] != null && b.ci[0] != null && a.ci[0] <= b.ci[1] && b.ci[0] <= a.ci[1];
  const parts = [];
  parts.push(fp.rate < fr.rate ? "The projection produced fewer false alerts under added noise" : "The projection did not reduce false alerts under added noise");
  if (sr && sp && sr.rate != null && sp.rate != null)
    parts.push(sp.rate >= sr.rate - 0.05 ? "and kept its sensitivity to the controlled change" : `but it also detected the controlled change less often (${pct(sr)} → ${pct(sp)})`);
  let s = parts.join(" ") + ".";
  if (overlap(fr, fp)) s += " The confidence intervals for false alerts overlap, so this difference is not established with this sample.";
  return s;
}
function resultsTable(agg, level) {
  const R = agg.results[level], models = Object.keys(R);
  const conds = Object.keys(R[models[0]]);
  const better = (c) => (c === "pause_change" ? "higher is better" : "lower is better");
  return `<table><caption class="sr-only">${level}-level rates</caption><thead><tr><th scope="col">Condition</th>${models.map((m) => `<th scope="col" class="num">${m === "B0" ? "Raw WavLM" : m === "P1" ? "Projection (P1)" : "Projection, invariance only (B1)"}</th>`).join("")}</tr></thead><tbody>
    ${conds.map((c) => `<tr><td>${esc(COND_LABEL(agg, c))}<br><span class="tiny muted">${c === "pause_change" ? "share detected" : "share flagged (false alerts)"}, ${better(c)}</span></td>
      ${models.map((m) => `<td class="num"><strong>${pct(R[m][c])}</strong><br><span class="tiny muted">${ci(R[m][c])} &middot; n=${R[m][c].n}</span></td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
async function viewResults() {
  const res = await api("/results");
  const agg = res.aggregate, h = agg && agg.headline, snap = res.demo_snapshot, d = res.demo;
  const canRun = d.clip_present && d.profile_baseline_ready;
  const why = !d.clip_present ? `Add a consented 40–60 s recording at <code>demo/clip.wav</code> to enable it.`
    : !d.profile_baseline_ready ? `The clip is compared with the baseline of <strong>${esc(d.profile)}</strong>, which needs 5 good check-ins from the same speaker first.` : "";
  $app.innerHTML = `<h1>Does the projection ignore the room but not the speaker?</h1>
    ${h ? `<p class="lead" style="max-width:80ch"><strong>Under ${esc(h.noise_condition)}, raw WavLM flagged ${pct(h.false_alert_raw)} of unchanged sessions as different; the noise-invariant projection flagged ${pct(h.false_alert_projection)}.
        For a controlled change in pausing, raw WavLM detected ${pct(h.change_sensitivity_raw)} and the projection ${pct(h.change_sensitivity_projection)}.</strong></p>
        <div class="callout info" style="margin-bottom:20px"><p style="margin:0">${esc(verdict(h))}</p></div>`
      : `<div class="callout caution" style="margin-bottom:20px"><h3>Results pending</h3><p class="small">The aggregate experiment hasn't been run on this machine's data yet. On Colab: <code>python -m research.lite.aggregate</code>, then copy <code>results/aggregate.json</code> into the project. Nothing on this page is filled in until then.</p></div>`}

    <section class="card stack"><div class="row"><h2 style="margin:0">See it on one recording</h2><span class="spacer"></span>
      <button class="primary" id="rundemo" ${canRun ? "" : "disabled"}>${icon("analysis")}Run the demo</button></div>
      <p class="small muted">A bundled clip is analysed twice, as recorded and with ${esc(d.noise.replace("_", " "))} noise added from ${d.start_s} to ${d.end_s} s at ${d.snr_db} dB, against one speaker's baseline. One example: the numbers below are the evidence. ${why}</p>
      <p id="demomsg" role="status" class="small"></p>
      <div id="demoout">${snap ? demoSnapshotHtml(snap) : '<p class="muted small">No demo has been run yet.</p>'}</div>
    </section>

    ${agg ? `<section class="card" style="margin-top:20px"><h2>The numbers</h2>
      <p class="small muted">Session level (3 windows, about 24 s, the way the app scores a check-in). Each model is judged against its own boundary, set from <em>training</em> speakers' clean audio only. 95% intervals: bootstrap over ${agg.meta.eval_speakers} held-out speakers.</p>
      <div class="tablewrap">${resultsTable(agg, "session")}</div>
      <details><summary>Window level (single 8 s windows)</summary><div class="tablewrap">${resultsTable(agg, "window")}</div></details>
      <p class="tiny muted">Generated ${esc(agg.generated_at)} &middot; WavLM layer ${agg.meta.wavlm_layer} &middot; ${esc(agg.meta.distance)} distance &middot; boundary = ${Math.round(agg.meta.boundary_quantile * 100)}th percentile of normal variation.</p></section>` : ""}

    <div class="grid half" style="margin-top:20px">
      <section class="card stack"><h2>Method</h2><ul class="small">
        <li>Frozen <strong>WavLM-base</strong> speech encoder; a small <strong>projection</strong> (768&rarr;128) trained, without diagnosis labels, to give the same output for clean and noisy versions of the same speech (invariance, variance and covariance terms) while still predicting pause fraction, pitch variability and speech timing.</li>
        <li>Training noise: MUSAN noise, music and babble, simulated rooms and microphone coloring. Evaluation noise comes from <strong>different noise files</strong> and <strong>different speakers</strong>.</li>
        <li>A session's difference from baseline is the cosine distance from the mean of that speaker's other sessions. Crossing the model's boundary counts as an alert.</li>
        <li>The controlled speech change inserts short pauses of the recording's own room tone after each phrase. The words and recording conditions stay the same.</li>
      </ul></section>
      <section class="card stack"><h2>Limitations</h2><ul class="small">
        ${((agg && agg.limitations) || ["Results are pending, so no evidence is shown yet."]).map((x) => `<li>${esc(x)}</li>`).join("")}
        <li>This is a research prototype. A difference from baseline describes speech and recording conditions; it is not a diagnosis or a clinical alert.</li>
      </ul></section>
    </div>

    <section class="card stack" style="margin-top:20px"><h2>References</h2><ol class="small">
      <li>Tao, Esposito &amp; Vinciarelli. The Androids Corpus: a new publicly available benchmark for speech-based depression detection. Interspeech 2023. <a href="https://www.isca-archive.org/interspeech_2023/tao23_interspeech.pdf">isca-archive.org</a></li>
      <li>Chen et al. WavLM: large-scale self-supervised pre-training for full stack speech processing. IEEE JSTSP 2022. <a href="https://arxiv.org/abs/2110.13900">arXiv:2110.13900</a></li>
      <li>Bardes, Ponce &amp; LeCun. VICReg: variance-invariance-covariance regularization for self-supervised learning. ICLR 2022. <a href="https://arxiv.org/abs/2105.04906">arXiv:2105.04906</a></li>
      <li>Snyder, Chen &amp; Povey. MUSAN: a music, speech, and noise corpus. 2015. <a href="https://arxiv.org/abs/1510.08484">arXiv:1510.08484</a></li>
      <li>Ko et al. A study on data augmentation of reverberant speech for robust speech recognition. ICASSP 2017 (OpenSLR 28 room impulse responses). <a href="https://www.openslr.org/28/">openslr.org/28</a></li>
    </ol></section>

    <details class="card" style="margin-top:20px"><summary>Try your own recording (controlled experiment)</summary>${await labFormHtml()}</details>`;
  wireLab();
  if (snap) wireSnapshot(snap);
  document.getElementById("rundemo").onclick = async (e) => {
    e.target.disabled = true;
    const msg = document.getElementById("demomsg");
    msg.textContent = "Analysing the clip, clean and with noise…";
    try {
      const r = await fetch("/api/demo/run", { method: "POST" });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.status);
      msg.textContent = "Done.";
      document.getElementById("demoout").innerHTML = demoSnapshotHtml(j);
      wireSnapshot(j);
    } catch (err) { msg.textContent = "The demo failed: " + err.message; }
    e.target.disabled = false;
  };
}
function demoSnapshotHtml(j) {
  const n = j.noise;
  return `<p class="tiny muted">Measured output of the run at ${esc(fmtDT(j.ran_at))} with ${esc(j.model || "")}.</p>
    ${analysisBlock(j.noisy, { key: "dnoisy", title: `With ${n.kind.replace("_", " ")} noise, ${n.start_seconds}–${n.end_seconds} s at ${n.snr_db} dB`, recordingLabel: "Bundled demo clip + noise" })}
    <details class="card"><summary>Same clip with no noise added</summary>${analysisBlock(j.clean, { key: "dclean", recordingLabel: "Bundled demo clip" })}</details>`;
}
function wireSnapshot(j) { wireAnalysis(j.noisy, "dnoisy"); wireAnalysis(j.clean, "dclean"); }

// ------------------------------------------------------------------ Trends (session level)
async function viewTrends() {
  const days = Number(new URLSearchParams(location.hash.split("?")[1] || "").get("days")) || 30;
  const tl = await timeline(S.pid, Math.min(days, 365));
  const cks = tl ? tl.checkins : [];
  const sp = (d, label) => `<a href="#/trends?days=${d}" ${d === days ? 'aria-current="page"' : ""}>${label}</a>`;
  const chips = cks.filter((c) => (c.timing_chips || []).length).slice(-6).reverse();
  $app.innerHTML = `<h1>Voice trends</h1>
    <p class="lead">One point per check-in, compared with your baseline. These describe your speech and recordings; they are not health measurements.</p>
    <nav class="tabs" aria-label="Time range">${sp(7, "7 days")}${sp(30, "30 days")}${sp(90, "90 days")}${sp(365, "All")}</nav>
    ${!tl ? '<div class="card muted">No check-ins on this profile yet.</div>' : `
    <section class="card"><h2>Difference from baseline</h2>${tl.baseline ? sessionChart(cks) : `<p class="muted">Shown once your baseline is ready.</p>`}</section>
    <div class="grid half" style="margin-top:20px">
      <section class="card"><h2>Speech timing</h2>
        ${chips.length ? chips.map((c) => `<p class="small"><strong>${fmtDate(c.created_at)}:</strong> ${c.timing_chips.map((x) => esc(sentence(x.text))).join(" ")}</p>`).join("") : '<p class="muted small">No timing differences of note, or timing not available (needs transcription).</p>'}
        <div class="tablewrap">${timingTable(cks)}</div></section>
      <section class="card"><h2>Recording quality</h2><div class="tablewrap"><table><thead><tr><th>Date</th><th>Quality</th><th class="num">Speech (s)</th><th class="num">SNR (dB)</th></tr></thead><tbody>
        ${cks.slice().reverse().map((c) => `<tr><td>${fmtDT(c.created_at)}</td><td>${qualityPill(c.quality.verdict)}</td><td class="num">${fmt(c.quality.speech_s, 0)}</td><td class="num">${fmt(c.quality.snr_db, 0)}</td></tr>`).join("")}
      </tbody></table></div></section>
    </div>`}`;
}
const sentence = (t) => { const m = t.match(/^(speech|pauses|pause length) (\d+)% (.+) than baseline$/); return m ? `${m[1][0].toUpperCase() + m[1].slice(1)} ${m[1] === "pauses" ? "were" : "was"} ${m[2]}% ${m[3]} than in your baseline recordings.` : t; };
function timingTable(cks) {
  const rows = cks.filter((c) => c.timing && c.timing.speech_rate_wpm != null).slice(-10).reverse();
  if (!rows.length) return "";
  return `<table><caption class="sr-only">Speech timing per check-in</caption><thead><tr><th>Date</th><th class="num">Words / speech-min</th><th class="num">Pauses / min</th><th class="num">Mean pause (s)</th></tr></thead><tbody>
    ${rows.map((c) => `<tr><td>${fmtDate(c.created_at)}</td><td class="num">${fmt(c.timing.speech_rate_wpm, 0)}</td><td class="num">${fmt(c.timing.pause_rate_pm, 1)}</td><td class="num">${fmt(c.timing.mean_pause_s, 2)}</td></tr>`).join("")}</tbody></table>`;
}

// ------------------------------------------------------------------ About & Privacy
async function viewAbout() {
  const h = await health();
  const res = await api("/results").catch(() => ({}));
  const team = res.team || [];
  $app.innerHTML = `<h1>About &amp; Privacy</h1>
    <div class="grid half">
      <section class="card stack"><h2>Who it's for</h2>
        <p class="small">People whose speech is worth following between clinic visits, under a clinician's supervision, and the researchers studying whether that is reliable.
          Voice Drift is a research prototype; it measures speech and recording conditions and does not diagnose anything.</p>
        <h2>How it works</h2><ol class="small">
          <li>A one-minute daily voice check-in, recorded in the browser.</li>
          <li>A quality check rejects recordings with too little speech, too much noise or distortion.</li>
          <li>A frozen speech model (WavLM) turns the voice into numbers; a small trained projection aims to ignore the room and microphone while keeping how the person speaks.</li>
          <li>Each check-in is compared with the person's own first five, using a boundary set from normal day-to-day variation.</li>
        </ol></section>
      <section class="card stack"><h2>Limitations</h2><ul class="small">
          <li>Evaluated on a small research corpus of Italian interview speech, with a synthetic speech change. See <a href="#/results">Results</a> for the numbers and their uncertainty.</li>
          <li>Day-to-day variation in real homes is larger than in the evaluation data.</li>
          <li>A difference from baseline is not evidence of any health condition, and not a clinical alert.</li>
        </ul>
        <h2>What's next</h2><ul class="small">
          <li>Real (not synthetic) speech-change recordings, and more speakers and languages.</li>
          <li>Recalibration on recordings made the way the app records them.</li>
          <li>Clinician-supervised pilot with consent, reporting, and sharing controls.</li>
        </ul>
        <h2>Team</h2><p class="small">${team.length ? team.map(esc).join(", ") : "Built at HackPrinceton 2026."}</p></section>
    </div>
    <div class="grid half" style="margin-top:20px">
      <section class="card stack"><h2>${icon("shield")} How your data is handled</h2>
        <dl class="meta small">
          <dt>Audio capture</dt><dd>In your browser, only while you are recording.</dd>
          <dt>Where analysis runs</dt><dd>On this computer's Voice Drift server (127.0.0.1). Audio is uploaded only to it.</dd>
          <dt>Audio retention</dt><dd>The audio file is deleted when analysis finishes, including when it fails.</dd>
          <dt>What is kept</dt><dd>Recording-quality results, comparison scores, per-window speech features (not audio), a coarse loudness summary, and the transcript, in a local database on this computer.</dd>
          <dt>Transcripts</dt><dd>${h.llm_reachable ? "Sent to Anthropic's API to find quotes for the evidence view. Audio is never sent." : "Not sent anywhere: no language-model key is configured."}</dd>
          <dt>Who can see results</dt><dd>Anyone using this app on this computer. There are no accounts, and nothing is sent to a clinician automatically.</dd>
        </dl>
        <p class="small muted">Derived features are not claimed to be anonymous; they stay on this computer.</p></section>
      <section class="card stack"><h2>Your data</h2>
        <p class="small">Delete every check-in, score and transcript stored for the current profile (<strong>${esc(S.pid)}</strong>). This can't be undone.</p>
        <p><button id="del">Delete this profile's data</button> <span id="delmsg" role="status" class="small"></span></p>
        <hr><h2>Demo &amp; research</h2>
        <p class="small muted">Synthetic profiles are labelled "Synthetic demo data" everywhere they appear and are kept separate from real recordings.</p>
        <div class="row">${["all", "stable", "drift", "blank"].map((s) => `<button data-reset="${s}">Load demo: ${s}</button>`).join("")}</div>
        <p><a href="#/research">${icon("report", 16)} Research workspace</a> <span class="small muted">(all profiles, transcripts, evidence quotes)</span></p>
      </section>
    </div>`;
  document.getElementById("del").onclick = async () => {
    if (!confirm(`Delete all stored data for ${S.pid}? This can't be undone.`)) return;
    await api(`/patients/${encodeURIComponent(S.pid)}`, { method: "DELETE" });
    document.getElementById("delmsg").textContent = "Deleted.";
  };
  $app.querySelectorAll("[data-reset]").forEach((b) => (b.onclick = async () => {
    b.disabled = true; await post("/demo/reset", { scenario: b.dataset.reset }); S.health = null; route();
  }));
}

// ------------------------------------------------------------------ Research workspace (former clinician view)
async function viewResearch() {
  const list = await api("/patients");
  $app.innerHTML = `<h1>Research workspace</h1><p class="lead">All profiles on this computer. Neutral comparison labels only; no risk scores.</p>
    <div class="card tablewrap"><table><thead><tr><th>Profile</th><th>Data</th><th>Latest comparison</th><th>Last check-in</th><th class="num">Open notes</th></tr></thead><tbody>
      ${list.map((p) => `<tr class="click" data-id="${esc(p.id)}" tabindex="0"><td>${esc(p.display_name)}</td><td>${modeTag(p.data_mode)}</td><td>${statusPill(p.status)}</td><td>${p.last_checkin ? fmtDT(p.last_checkin) : "&ndash;"}</td><td class="num">${p.open_alerts}</td></tr>`).join("") || '<tr><td colspan="5" class="muted">No profiles yet.</td></tr>'}
    </tbody></table></div>`;
  $app.querySelectorAll("tr.click").forEach((tr) => {
    const go = () => (location.hash = "#/research/" + encodeURIComponent(tr.dataset.id));
    tr.onclick = go; tr.onkeydown = (e) => { if (e.key === "Enter") go(); };
  });
}

async function viewResearchDetail(pid) {
  const tl = await timeline(pid, 365);
  if (!tl) { $app.innerHTML = `<div class="card">Unknown profile. <a href="#/research">Back</a></div>`; return; }
  const cks = tl.checkins;
  let sel = cks.length ? cks[cks.length - 1].checkin_id : null;
  const render = () => {
    const c = cks.find((x) => x.checkin_id === sel);
    $app.innerHTML = `<a href="#/research">&larr; Research workspace</a>
      <h1>${esc(tl.patient.display_name)} ${modeTag(tl.patient.data_mode)}</h1>
      ${tl.alerts.length ? `<section class="card stack"><h2>Notes for review</h2>${[...tl.alerts].reverse().map((a) => `<div class="row small"><strong>${esc(ALERT_NAMES[a.kind] || a.kind)}</strong><span class="muted">${fmtDT(a.created_at)}</span>
        <a href="#" data-ck="${esc(a.checkin_id)}">view check-in</a>${a.acked_at ? `<span class="muted">reviewed by ${esc(a.acked_by)}${a.note ? ": " + esc(a.note) : ""}</span>` : `<input id="note_${a.id}" placeholder="Note" aria-label="Review note"><button data-ack="${a.id}">Mark reviewed</button>`}</div>`).join("")}</section>` : ""}
      <section class="card" style="margin-top:16px"><h2>Difference from baseline (session level)</h2>${sessionChart(cks)}
        <div class="row small">${cks.map((x) => `<a href="#" data-ck="${esc(x.checkin_id)}">${fmtDate(x.created_at)}</a>`).join(" ")}</div></section>
      ${c ? checkinDetail(c) : ""}`;
    $app.querySelectorAll("[data-ck]").forEach((el) => (el.onclick = (ev) => { ev.preventDefault(); sel = el.dataset.ck; render(); }));
    $app.querySelectorAll("[data-ack]").forEach((b) => (b.onclick = async () => {
      await post(`/alerts/${b.dataset.ack}/ack`, { clinician: "researcher", note: $app.querySelector(`#note_${b.dataset.ack}`).value });
      viewResearchDetail(pid);
    }));
  };
  render();
}

function highlight(text, evidence) {
  let html = esc(text);
  for (const e of evidence || []) {
    const q = esc(e.quote).replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
    html = html.replace(new RegExp(q, "i"), (m) => `<mark title="${esc(e.item)} (${esc(e.direction)})">${m}</mark>`);
  }
  return html;
}
function checkinDetail(c) {
  const safety = c.status === "SAFETY";
  return `<section class="card stack" style="margin-top:16px">
    <div class="row"><h2 style="margin:0">Check-in ${fmtDT(c.created_at)}</h2>${statusPill(c.status)}${modeTag(c.data_mode)}${c.acted ? '<span class="mode synthetic_demo">acted</span>' : ""}</div>
    <p class="small">${qualityPill(c.quality.verdict)} &middot; speech ${fmt(c.quality.speech_s, 0)} s &middot; SNR ${fmt(c.quality.snr_db, 0)} dB${c.environment ? " &middot; " + esc(c.environment) : ""}</p>
    ${safety ? `<div class="support"><p class="big">Support resources were shown</p><p>No scores are displayed for this check-in.</p></div>` : ""}
    ${!safety && c.drift ? `<p>Difference from baseline: <strong>${fmt(c.drift.d / c.drift.tau)}</strong> &times; the session boundary${c.drift.confirmed ? " (sustained over recent check-ins)" : ""}.</p>` : !safety ? '<p class="muted">Baseline check-in: not compared.</p>' : ""}
    ${(c.timing_chips || []).length ? `<p class="small">${c.timing_chips.map((x) => esc(sentence(x.text))).join(" ")}</p>` : ""}
    ${c.has_window_analysis ? `<p><a class="btn" href="#/analysis/${encodeURIComponent(c.checkin_id)}">${icon("analysis")}Window-by-window analysis</a></p>` : ""}
    ${!safety ? `<h3>Evidence quotes</h3>${(c.evidence || []).length ? c.evidence.map((e) => `<p class="small"><span class="pill gray">${esc(e.item)}</span> &ldquo;${esc(e.quote)}&rdquo;</p>`).join("") : '<p class="muted small">None found (or evidence unavailable).</p>'}
      <h3>Transcript</h3><div class="transcript small">${c.transcript ? highlight(c.transcript, c.evidence) : '<span class="muted">Transcript unavailable.</span>'}</div>` : ""}
  </section>`;
}
