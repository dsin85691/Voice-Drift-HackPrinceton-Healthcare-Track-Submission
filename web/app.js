// Voice Drift frontend (UI spec v1.0). Dependency-free ES module served by FastAPI; no build step.
// All numbers shown come from the backend. Missing values are shown as missing, never as zero.
const $app = document.getElementById("app");
const DEMO = new URLSearchParams(location.search).get("demo") === "1";
// Browser recordings come out 0.3-2 s shorter than the on-screen timer (mic start-up, last chunk on stop), so Stop
// unlocks at 23 s to land safely above the server's minimum. The guidance to the person stays "at least 20 seconds".
const MIN_S = 23, MAX_S = 60;

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
const S = { pid: stored || null, health: null, results: null };
// Results visibility (per browser): "auto" = show the evidence only if results/aggregate.json exists on the server;
// "show" / "hide" force it for rehearsal. Never changes or invents data: "show" without a file still says pending.
let resultsMode = "auto";
try { resultsMode = localStorage.getItem("vd_results_mode") || "auto"; } catch {}
const setResultsMode = (m) => { resultsMode = m; try { localStorage.setItem("vd_results_mode", m); } catch {} };
const resultsVisible = () => resultsMode === "show" || (resultsMode === "auto" && !!(S.results && S.results.aggregate));
const setPid = (p) => { S.pid = p; try { localStorage.setItem("vd_pid", p); } catch {} };
// where real recordings go when the current profile is synthetic (the topbar always offers this profile)
const LIVE_PID = "p_demo";

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
  // from the Voice Drift Healthcare UI kit
  chart: '<path d="M3 3v18h18"/><path d="m6 16 4-5 4 3 6-8"/>',
  sound: '<path d="M2 10v4m4-8v12m4-15v18m4-13v8m4-12v16m4-10v4"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  arrow: '<path d="M4 12h15m-6-6 6 6-6 6"/>',
  chevron: '<path d="m9 18 6-6-6-6"/>',
  alert: '<path d="m12 3 10 18H2L12 3Z"/><path d="M12 9v5m0 3h.01"/>',
  file: '<path d="M6 2h8l5 5v14H6V2Zm8 0v6h5M9 13h7M9 17h7"/>',
  queue: '<rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4V2h6v2M9 10h6M9 15h6M9 18h3"/>',
  play: '<path d="m8 5 11 7-11 7V5Z"/>',
  checkcircle: '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
};
const icon = (n, size = 20) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${IC[n]}</svg>`;

// Neutral wording (UI spec section 10): no risk, deterioration or diagnosis language, no alarm red.
// One wording for the task and the claims, used on Home, Check-in, the demo and Results (backlog VD-05 / VD-06)
const INSTRUCTION = "Read the passage aloud at your normal speaking pace. Voice Drift compares your vocal patterns with previous recordings of the same passage. Pauses and mistakes are okay.";
const DISCLAIMER = "Voice Drift is an experimental monitoring aid, not a diagnostic system. Changes in speech do not independently establish changes in a person's mental health.";
const UNMATCHED_CAVEAT = "Noise-robustness comparisons are preliminary because clean-speech false-alert rates have not yet been matched across methods.";
// The OVERALL voice-pattern decision (session score vs boundary). Individual measurements are shown separately.
const STATUS = {
  BASELINE: ["Building baseline", "gray", "info"],
  GREEN: ["Overall pattern within usual range", "teal", "check"],
  AMBER: ["Overall pattern above boundary", "amber", "info"],
  RED: ["Above boundary on repeated check-ins", "navy", "info"],
  SAFETY: ["Support resources shown", "navy", "shield"],
  INCOMPATIBLE: ["Not compared (different speaking task)", "gray", "info"],
};
const QUALITY = { PASS: ["Good recording", "teal", "check"], WARN: ["Usable, some background noise", "amber", "info"], REJECT: ["Not usable", "gray", "x"] };
const MODES = { real_recording: "Real recording", controlled_experiment: "Controlled experiment", synthetic_demo: "Synthetic demo data", demo_synthetic_voice: "Prerecorded sample (synthetic voice)" };
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
// Demo VIEW switch (not accounts): who the screen is speaking to. Stored per browser only. Anyone using this
// computer can open either view; the switch changes navigation and wording, never access (see About & Privacy).
let role = "patient";
try { role = localStorage.getItem("vd_role") === "clinician" ? "clinician" : "patient"; } catch {}
const isClin = () => role === "clinician";
const setRole = (r) => { role = r; try { localStorage.setItem("vd_role", r); } catch {} };
const NAV_PATIENT = [
  ["home", "Home", "#/", "home"], ["checkin", "Check-In", "#/checkin", "mic"], ["trends", "Trends", "#/trends", "trends"],
  ["about", "About & Privacy", "#/about", "shield"],
];
const NAV_CLINICIAN = [
  ["queue", "Review queue", "#/review", "queue"], ["patients", "Patients", "#/research", "report"],
  ["results", "Results", "#/results", "analysis"], ["about", "About & Privacy", "#/about", "shield"],
];
const NAV = [...NAV_PATIENT, ...NAV_CLINICIAN];               // labels for the breadcrumb
function renderNav(active) {
  const nav = (isClin() ? NAV_CLINICIAN : NAV_PATIENT).filter(([k]) => k !== "results" || resultsVisible());
  document.getElementById("sidenav").innerHTML = nav.map(([k, label, href, ic]) =>
    `<a class="navlink" href="${href}" ${k === active ? 'aria-current="page"' : ""}>${icon(ic)}${label}</a>`).join("");
  document.getElementById("bottomnav").innerHTML = nav
    .map(([k, label, href, ic]) => `<a href="${href}" ${k === active ? 'aria-current="page"' : ""}>${icon(ic, 22)}${label.replace(" & Privacy", "").replace("Review queue", "Queue")}</a>`).join("");
}
async function renderTopbar() {
  const [pts, h, notes] = await Promise.all([api("/patients").catch(() => []), health(), api("/v1/notifications").catch(() => null)]);
  const ids = pts.map((p) => p.id);
  const opts = (ids.includes("p_demo") ? pts : [{ id: "p_demo", display_name: "Demo speaker (live)", data_mode: "real_recording" }, ...pts]);
  if (!S.pid) S.pid = ids.includes("p_drift") ? "p_drift" : "p_demo";
  const cur = opts.find((p) => p.id === S.pid) || opts[0];
  if (cur && cur.id !== S.pid) setPid(cur.id);
  const pageName = (NAV.find(([k]) => k === S.page) || [, "Review queue"])[1];
  const seg = (r, label) => `<button type="button" data-role="${r}" aria-pressed="${role === r}" class="${role === r ? "primary" : ""}" style="min-height:36px;padding:6px 12px;font-size:14px">${label}</button>`;
  document.getElementById("topbar").innerHTML = `
    <span class="crumb">${isClin() ? "Clinician view" : "Your workspace"} ${icon("chevron", 15)} <strong>${esc(pageName)}</strong></span>
    <span class="spacer"></span>
    <span class="row" style="gap:6px" role="group" aria-label="Viewing as (demo switch, no accounts)"><span class="small muted">Viewing as</span>${seg("patient", "Patient")}${seg("clinician", "Clinician")}<span class="tiny muted">demo, no accounts</span></span>
    ${isClin() ? "" : '<label for="profile" class="small">You are</label>'}
    ${isClin() ? "" : `<select id="profile">${opts.map((p) => `<option value="${esc(p.id)}" ${p.id === S.pid ? "selected" : ""}>${esc(p.display_name)}</option>`).join("")}</select>${modeTag(cur && cur.data_mode)}`}
    ${isClin() ? `<a class="pill ${notes && notes.unresolved ? "navy" : "gray"}" href="#/review" title="Reviewer queue (demo: in-app only, nothing is sent externally)">${icon("report", 14)}${notes && notes.unresolved ? `${notes.unresolved} pending review${notes.unresolved === 1 ? "" : "s"}` : "No pending reviews"}</a>` : ""}
    <span class="pill gray" title="Speech encoder in use">${icon("analysis", 14)}${esc(modelLabel(h))}</span>`;
  document.getElementById("sidefoot").innerHTML = `Not a diagnostic system.<br>${esc(modelLabel(h))}`;
  const prof = document.getElementById("profile");
  if (prof) prof.onchange = (e) => { setPid(e.target.value); route(); };
  document.querySelectorAll("[data-role]").forEach((b) => (b.onclick = () => {
    if (b.dataset.role === role) return;
    setRole(b.dataset.role);
    if (location.hash === "#/" || location.hash === "") route(); else location.hash = "#/";   // land on the role's home
  }));
}
// A page that belongs to the other view: say so plainly, offer the switch (no access control is implied)
function otherView(target) {
  $app.innerHTML = `<div class="card stack"><h1>${target === "clinician" ? "This is part of the clinician view" : "Recording is in the patient view"}</h1>
    <p>${target === "clinician" ? "The review queue and the patient list are shown in the clinician view." : "Check-ins are recorded from the patient view."}
      This is a demo switch, not an account: anyone using this computer can open either view.</p>
    <p><button class="primary" id="switchview">Switch to the ${target} view</button></p></div>`;
  document.getElementById("switchview").onclick = () => { setRole(target); route(); };
}

let cleanup = () => {};
async function route() {
  cleanup(); cleanup = () => {};
  const h = location.hash.replace(/^#/, "").split("?")[0] || "/";   // "#/trends?days=7" -> "/trends"
  const seg = h.split("/").filter(Boolean);
  const top = seg[0] || "home";
  const clinPage = ["review", "research", "clinician"].includes(top);
  const active = isClin()
    ? ({ home: "queue", review: "queue", research: "patients", clinician: "patients", settings: "about", windows: "results", analysis: "results" }[top] || top)
    : ({ settings: "about", windows: "trends", analysis: "trends" }[top] || top);
  S.results = await api("/results").catch(() => null);      // refreshed every navigation: picks up a newly copied file
  S.page = active;
  renderNav(active);
  await renderTopbar();
  $app.innerHTML = '<p class="muted">Loading&hellip;</p>';
  try {
    if (isClin() && top === "home") await viewReviewQueue();               // clinician lands on the queue
    else if (!isClin() && clinPage) otherView("clinician");
    else if (isClin() && top === "checkin") otherView("patient");
    else if (top === "home") await viewHome();
    else if (top === "checkin") await viewCheckin();
    else if (top === "analysis" && seg[1] === "lab") await viewLab();
    else if (top === "analysis" || top === "windows") await viewAnalysis(seg[1] ? decodeURIComponent(seg[1]) : null);
    else if (top === "trends") await viewTrends();
    else if (top === "results") await viewResults();
    else if (top === "about" || top === "settings") await viewAbout();
    else if (top === "review" && seg[1]) await viewReviewDetail(decodeURIComponent(seg[1]));
    else if (top === "review") await viewReviewQueue();
    else if ((top === "research" || top === "clinician") && seg[1]) await viewResearchDetail(decodeURIComponent(seg[1]));
    else if (top === "research" || top === "clinician") await viewResearch();
    else $app.innerHTML = `<div class="card"><h1>Page not found</h1><a href="#/">Go to Home</a></div>`;
    // kit "prototype ribbon": the same disclaimer on every page (backlog VD-06)
    $app.insertAdjacentHTML("afterbegin", `<div class="ribbon">${icon("info", 18)}<span><strong>Research prototype:</strong> ${esc(DISCLAIMER)}</span></div>`);
  } catch (e) {
    $app.innerHTML = `<div class="callout caution" role="alert"><h3>Reconnecting to the Voice Drift server…</h3>
      <p>We couldn't load this page (${esc(e.message)}). If the server is restarting, this page reloads by itself as soon as it is back.</p>
      <p class="small muted" id="reconn">Checking every 3 seconds.</p></div>`;
    // auto-recover after a server restart: poll health, then re-render the same page (no manual reload needed)
    const iv = setInterval(async () => {
      try {
        const r = await fetch("/api/health", { cache: "no-store" });
        if (r.ok) { clearInterval(iv); S.health = null; route(); }
      } catch { const el = document.getElementById("reconn"); if (el) el.textContent = `Still waiting for the server (${new Date().toLocaleTimeString()}).`; }
    }, 3000);
    const prev = cleanup;
    cleanup = () => { clearInterval(iv); prev(); };
  }
  $app.focus({ preventScroll: true });
}
window.addEventListener("hashchange", route);

// ---- server unavailable: never fail silently. Any network failure that escapes a handler shows one banner, which
// checks /api/health every 3 s and, when the server answers, removes itself and re-renders the current page.
const OFFLINE_TEXT = "Voice Drift isn't responding. Check that the server is running, then try again.";
const isNetErr = (e) => e instanceof TypeError && /fetch|network|load failed/i.test(String(e.message || e));
const offlineMsg = (e) => (isNetErr(e) ? OFFLINE_TEXT : `That didn't work (${e && e.message ? e.message : e}).`);
let offlineTimer = null;
function showOffline() {
  if (document.getElementById("offline-banner")) return;
  document.body.insertAdjacentHTML("afterbegin", `<div id="offline-banner" role="alert" style="position:sticky;top:0;z-index:50;background:#FFF4E8;border-bottom:2px solid var(--amber);color:var(--amber-ink);padding:10px 16px;font-weight:700;text-align:center">${OFFLINE_TEXT} <span style="font-weight:500">Reconnecting automatically…</span></div>`);
  offlineTimer = setInterval(async () => {
    try {
      const r = await fetch("/api/health", { cache: "no-store" });
      if (!r.ok) return;
      clearInterval(offlineTimer); offlineTimer = null;
      document.getElementById("offline-banner")?.remove();
      S.health = null; route();
    } catch { /* still down: keep waiting */ }
  }, 3000);
}
window.addEventListener("unhandledrejection", (ev) => {
  if (isNetErr(ev.reason)) { ev.preventDefault(); showOffline(); }
});

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
  const res = S.results || {};
  const synthetic = tl && tl.patient.data_mode === "synthetic_demo";
  // ---- everything below is computed from the stored timeline; nothing is invented for display
  const scoredCks = cks.filter((c) => c.drift_score != null);
  const recent = cks.filter((c) => c.status !== "SAFETY").slice(-5);
  const recentScored = scoredCks.slice(-5);
  const nAbove = recentScored.filter((c) => c.drift && c.drift.exceeds).length;
  // Home is the PATIENT landing page: an open review is never announced to the patient (an experimental heuristic
  // flag is a clinician-side workflow item); they see their normal trend and neutral wording only.
  const pendingFlag = isClin() ? cks.map((c) => c.review_flag).filter((f) => f && f.state === "pending").pop() : null;
  const lastScored = scoredCks[scoredCks.length - 1];
  const fallbackEnc = h.encoder && h.encoder.includes("fallback");
  const signed = (v, more, less) => (v == null ? null : `${Math.abs(Math.round(v))}<small>% ${v < 0 ? less : v > 0 ? more : "change"}</small>`);
  const series = (k) => cks.map((c) => c[k]).filter((v) => v != null);
  const attn = pendingFlag ? {
      active: true, ic: "alert", h: "A pattern worth reviewing",
      p: "Several recent recordings differ from this person's baseline, on different days. That is a prompt for a person to look closer, not a diagnosis.",
      btn: `<a class="btn primary" href="#/review/${encodeURIComponent(pendingFlag.flag_id)}">Open the review item ${icon("arrow", 17)}</a>` }
    : lastScored && lastScored.drift.exceeds ? {
      active: true, ic: "info", h: "One recording different from your usual",
      p: "A single recording can differ for many reasons: the room, the microphone, a slower morning. Repeating the check-in shows whether it repeats.",
      btn: synthetic ? `<a class="btn primary" href="#/trends">See the trend ${icon("arrow", 17)}</a>` : `<a class="btn primary" href="#/checkin?verify=${encodeURIComponent(lastScored.checkin_id)}">Repeat check-in ${icon("arrow", 17)}</a>` }
    : !ready ? {
      active: false, ic: "clock", h: "Building the baseline",
      p: `The first ${need} good recordings set the personal baseline. Until then, no comparisons are shown.`,
      btn: `<a class="btn primary" href="#/checkin">${icon("mic", 17)}Start check-in</a>` }
    : {
      active: false, ic: "checkcircle", h: "Looking familiar so far",
      p: "Recent check-ins are within this person's usual range.",
      btn: `<a class="btn primary" href="#/trends">View check-in history ${icon("arrow", 17)}</a>` };
  $app.innerHTML = `
    <div class="page-heading"><div><div class="eyebrow"><span class="line"></span>Voice monitoring, in context</div>
      <h1>The story behind your voice.</h1>
      <p>Voice Drift compares short recordings of the same reading passage with a person's own baseline. A noisy kitchen or a new phone can change a recording as much as the speaker does, so it also checks recording quality and is designed to be less sensitive to the room.</p></div>
      <a class="btn" href="#/checkin">${icon("mic", 17)}New check-in</a></div>
    <section class="vd-hero" aria-label="About Voice Drift">
      <div class="vd-hero-copy"><div class="vd-kicker">PERSONALIZED VOICE TRACKING</div>
        <h2>Your voice is unique.<br><em>Your baseline should be, too.</em></h2>
        <p>${esc(INSTRUCTION)}</p>
        <div class="row"><button class="lime" id="herosample" ${res.sample && res.sample.present ? "" : "disabled"}>${icon("play", 18)}Analyze sample recording</button>
          <a class="btn ghost-light" href="#/checkin">Start check-in ${icon("arrow", 17)}</a></div></div>
      <div class="vd-visual" aria-hidden="true"><div class="vd-orbit o1"></div><div class="vd-orbit o2"></div>
        <div class="vd-visual-label"><span>VOICE FINGERPRINT</span><span class="pill-dark">PERSONAL</span></div>
        <div class="vd-orb">${icon("sound", 60)}</div>
        <div class="vd-visual-foot">${waveformHtml()}<span>compared with your own baseline</span></div></div>
    </section>
    ${tl && tl.repeat_requested ? `<div class="callout info" style="margin:16px 0"><h3>Repeat check-in requested</h3><p class="small">Your reviewer asked for another check-in using the same reading passage. This is a request for another recording, not a treatment instruction.</p><a class="btn primary" href="#/checkin">${icon("mic")}Start check-in</a></div>` : ""}
    ${synthetic ? `<div class="callout caution" style="margin:16px 0"><h3>Synthetic demo profile</h3><p class="small">Every value on this profile is generated for the demo, in a simplified feature space, to illustrate the monitoring and review workflow. The last two days are set to differ from the baseline; they are not a measurement of slower speech, and the trained model's overall score is not shown to detect slowed speech (see Results). Switch profile at the top to see real recordings.</p></div>` : ""}

    <div class="section-title"><div><div class="eyebrow">The current picture</div><h2>At a glance <span>&middot; ${last ? `latest check-in ${fmtDT(last.created_at)}` : "no check-ins yet"}</span></h2></div>
      <span class="small muted">${ready ? `Baseline set ${fmtDate(tl.baseline.frozen_at)} from ${need} recordings` : `Baseline ${Math.min(passes, need)} of ${need}`}</span></div>
    <div class="metric-grid">
      <a class="metric-card blue" href="#/trends"><div class="metric-top"><span class="metric-icon blue">${icon("chart")}</span>${icon("arrow", 17)}</div>
        <span class="metric-label">Speaking rate</span>
        <div class="metric-value">${signed(lastScored && lastScored.speaking_rate_change_pct, "faster", "slower") || '&ndash;<small>not available</small>'}</div>
        <div class="metric-bottom"><span>${lastScored && lastScored.speaking_rate_change_pct != null ? "vs. baseline" : "needs a transcript"}</span>${miniSpark(series("speaking_rate_change_pct"), "#3F72B8")}</div></a>
      <a class="metric-card peach" href="#/trends"><div class="metric-top"><span class="metric-icon coral">${icon("clock")}</span>${icon("arrow", 17)}</div>
        <span class="metric-label">${lastScored && lastScored.pause_measure === "mean_pause_s" ? "Pause length" : "Pause time"}</span>
        <div class="metric-value">${signed(lastScored && lastScored.pause_duration_change_pct, "more", "less") || '&ndash;<small>not available</small>'}</div>
        <div class="metric-bottom"><span>vs. baseline</span>${miniSpark(series("pause_duration_change_pct"), "#C9805F")}</div></a>
      <a class="metric-card lavender" href="#/trends"><div class="metric-top"><span class="metric-icon purple">${icon("queue")}</span>${icon("arrow", 17)}</div>
        <span class="metric-label">Different from your usual</span>
        <div class="metric-value">${nAbove}<small>of ${recentScored.length} compared</small></div>
        <div class="small-pills" aria-label="Last ${recent.length} check-ins">${recent.map((c, i) => { const k = c.drift && c.drift.exceeds ? "flag" : c.drift_score != null ? "good" : "none";
          return `<span class="pill-${k}" title="${fmtDate(c.created_at)}: ${k === "flag" ? "above boundary" : k === "good" ? "within boundary" : "not compared"}">${i + 1}</span>`; }).join("") || '<span class="pill-none">&ndash;</span>'}</div></a>
      <a class="metric-card green" href="#/trends"><div class="metric-top"><span class="metric-icon green">${icon("sound")}</span>${icon("arrow", 17)}</div>
        <span class="metric-label">Overall voice pattern</span>
        <div class="metric-value">${lastScored ? `${fmt(lastScored.drift_score)}<small>&times; boundary</small>` : '&ndash;<small>not compared yet</small>'}</div>
        <div class="metric-bottom"><span>${lastScored ? (lastScored.drift.exceeds ? "above 1.0" : "1.0 = edge of usual range") : ""}</span>${miniSpark(series("drift_score"), "#4F8A5D")}</div></a>
    </div>
    <p class="tiny muted" style="margin:10px 2px 0">The overall pattern and the individual measurements are separate: one measurement can change a lot while the overall pattern stays within range, and the other way round.${fallbackEnc ? " No trained speech model is loaded, so comparisons use simpler fallback features and are experimental." : ""}</p>

    <div class="overview-columns">
      <section class="card" aria-labelledby="tr-h"><div class="row"><div><h3 id="tr-h" style="margin:0">How the voice is shifting</h3><p class="small muted" style="margin:2px 0 0">Each point is one check-in, relative to the personal boundary</p></div><span class="spacer"></span><a class="small" href="#/trends">All trends &rarr;</a></div>
        ${ready ? sessionChart(cks.slice(-14), { height: 200 }) : `<p class="muted">The trend appears once the baseline is ready (${Math.min(passes, need)}/${need}).</p>`}</section>
      <section class="attention ${attn.active ? "active" : ""}" aria-labelledby="attn-h"><div class="attention-icon">${icon(attn.ic, 24)}</div>
        <div class="eyebrow">What we're noticing</div><h3 id="attn-h">${attn.h}</h3><p>${attn.p}</p>
        ${ready ? `<div class="attention-stat"><strong>${nAbove}<span> / ${recentScored.length}</span></strong><span>recent compared check-ins<br>different from your usual</span></div>`
          : `<div class="dots" aria-hidden="true" style="margin:auto 0 14px">${Array.from({ length: need }, (_, i) => `<span class="${i < Math.min(passes, need) ? "on" : ""}"></span>`).join("")}</div>`}
        ${attn.btn}</section>
    </div>

    <div class="section-title"><div><div class="eyebrow">Your track record</div><h2>Recent check-ins</h2></div><a class="small" href="#/trends">See all &rarr;</a></div>
    <section class="card history">${cks.slice().reverse().slice(0, 4).map((c) => { const flag = c.drift && c.drift.exceeds;
      return `<a class="history-row" href="#" data-hist="${esc(c.checkin_id)}"><span class="history-sym ${flag ? "flag" : ""}">${icon(flag ? "sound" : c.quality.verdict === "REJECT" ? "x" : "check", 18)}</span>
        <span class="history-info"><strong>${fmtDate(c.created_at)} check-in</strong><small>${fmtDT(c.created_at)} &middot; ${esc(MODES[c.data_mode] || "")}${c.drift_score != null ? ` &middot; ${fmt(c.drift_score)} &times; boundary` : ""}</small></span>
        <span>${c.quality.verdict === "REJECT" ? qualityPill("REJECT") : statusPill(c.status)}</span>${icon("chevron", 18)}</a>`; }).join("") || '<p class="muted" style="padding:14px 0">No check-ins yet.</p>'}</section>
    <div id="homeptdetail"></div>

    <div style="margin-top:22px">${sampleCardHtml(res)}</div>
    <section class="education"><div class="education-art">${icon("shield", 30)}</div>
      <div><strong>Built around a personal baseline, not a population average.</strong><p>Voices differ from person to person, so each check-in is compared with the same person's earlier readings of the same passage, with recording quality checked first.</p></div>
      <a class="small" href="#/about">How it works &rarr;</a></section>`;
  wireSample(res);
  if (synthetic) {   // synthetic profiles can't take recordings: every "check-in" button goes to the live profile instead
    $app.querySelectorAll('a[href="#/checkin"]').forEach((a) => {
      a.innerHTML = `${icon("mic", 17)}Record on the live profile`;
      a.title = "This is a synthetic demo profile; recordings are made on the live profile";
      a.onclick = () => setPid(LIVE_PID);
    });
  }
  const hs = document.getElementById("herosample");
  if (hs) hs.onclick = () => { const b = document.getElementById("samplebtn"); if (b) { b.scrollIntoView({ behavior: "smooth", block: "center" }); b.click(); } };
  wireChartPoints(cks, "homeptdetail");
  $app.querySelectorAll("[data-hist]").forEach((a) => (a.onclick = (e) => {
    e.preventDefault();
    const c = cks.find((x) => x.checkin_id === a.dataset.hist), el = document.getElementById("homeptdetail");
    if (c && el) { el.innerHTML = checkinDetail(c); el.scrollIntoView({ behavior: "smooth", block: "nearest" }); }
  }));
}
// kit components, drawn from real series only
function waveformHtml() {
  const lv = [12, 20, 14, 38, 22, 52, 27, 45, 15, 28, 58, 40, 68, 37, 23, 51, 30, 44, 18, 26, 47, 24, 58, 37, 20, 30, 13, 24, 42, 17, 31, 12];
  return `<div class="waveform" aria-hidden="true">${lv.map((x) => `<i style="height:${x}%"></i>`).join("")}</div>`;
}
function miniSpark(values, color) {
  const v = values.slice(-8);
  if (v.length < 2) return "";
  const lo = Math.min(...v), hi = Math.max(...v), span = hi - lo || 1;
  const pts = v.map((y, i) => `${5 + (i * 130) / (v.length - 1)},${49 - ((y - lo) / span) * 35}`);
  const [lx, ly] = pts[pts.length - 1].split(",");
  return `<svg class="mini-spark" viewBox="0 0 145 57" aria-hidden="true"><polyline points="${pts.join(" ")}" fill="none" stroke="${color}" stroke-width="2.9" stroke-linecap="round" stroke-linejoin="round"/><circle cx="${lx}" cy="${ly}" r="4.5" fill="${color}" stroke="#fff" stroke-width="2.5"/></svg>`;
}

// ------------------------------------------------------------------ headline number (from results/aggregate.json only)
const pct = (r) => (r && r.rate != null ? `${Math.round(r.rate * 100)}%` : "&ndash;");
const ci = (r) => (r && r.ci && r.ci[0] != null ? `${Math.round(r.ci[0] * 100)}–${Math.round(r.ci[1] * 100)}%` : "");
function headlineHtml(agg) {
  const h = agg && agg.headline;
  if (!h) return `<div class="card" style="margin:6px 0 20px"><p class="small muted" style="margin:0">${icon("info", 16)} Headline result pending: the aggregate experiment hasn't been run yet. <a href="#/results">How it is measured &rarr;</a></p></div>`;
  if (isV2(agg)) {
    const pd = h.primary_difference;
    return `<section class="card" style="margin:6px 0 20px" aria-label="Headline result">
      <p style="margin:0;max-width:90ch"><strong>${esc(h.sentence)}</strong></p>
      <p class="tiny muted" style="margin:8px 0 0">${pd && pd.diff != null ? `Paired difference in false alerts (projection minus raw): ${pts(pd.diff)} points, 95% interval ${pts(pd.ci[0])} to ${pts(pd.ci[1])}${pd.excludes_zero ? "" : ", which includes zero"}. ` : ""}${agg.meta.eval_speakers} held-out speakers, synthetic speech changes. <a href="#/results">Evidence and limits &rarr;</a></p>
      ${b1Short(agg) ? `<p class="small muted" style="margin:6px 0 0">${esc(b1Short(agg))}</p>` : ""}</section>`;
  }
  return `<section class="card" style="margin:6px 0 20px" aria-label="Headline result"><div class="row" style="align-items:baseline">
      <div><div class="small muted">False alerts when noise is added (unchanged speech)</div>
        <div class="kpi">${pct(h.false_alert_raw)} <small>raw model</small> &rarr; ${pct(h.false_alert_projection)} <small>with the noise-invariant projection</small></div></div>
      <span class="spacer"></span>
      <div class="small"><div class="muted">Detects a controlled change in pausing</div><strong>${pct(h.change_sensitivity_raw)} raw &rarr; ${pct(h.change_sensitivity_projection)} projection</strong></div>
    </div><p class="tiny muted" style="margin:8px 0 0">${esc(h.level)}, ${agg.meta.eval_speakers} held-out speakers, ${esc(h.noise_condition)}; synthetic speech change. <a href="#/results">Evidence and limits &rarr;</a></p>
    <p class="small callout caution" style="margin:8px 0 0">${esc(UNMATCHED_CAVEAT)} ${esc(b1Short(agg))}</p></section>`;
}

// VD-08: the B1 ablation, stated from the numbers in the results file (never assumed)
const NOISE_CONDS = ["noise_mix", "noise_20dB", "noise_10dB", "noise_5dB", "noise_0dB"];
function b1Compare(agg) {
  const R = agg && agg.results && (isV2(agg) ? agg.results.session.matched : agg.results.session);
  if (!R || !R.B0 || !R.B1 || !R.P1) return null;
  const cs = NOISE_CONDS.filter((c) => R.B0[c] && R.B1[c] && R.P1[c]);
  const lt = (a, b, c) => R[a][c].rate != null && R[b][c].rate != null && R[a][c].rate < R[b][c].rate;
  return { n: cs.length, b1BeatsB0: cs.filter((c) => lt("B1", "B0", c)).length, p1BeatsB1: cs.filter((c) => lt("P1", "B1", c)).length,
           b1BeatsP1: cs.filter((c) => lt("B1", "P1", c)).length, R };
}
function b1Short(agg) {
  const k = b1Compare(agg);
  if (!k || !k.n) return "";
  return k.p1BeatsB1 < k.n
    ? `The simpler invariance-only head (B1) had fewer false alerts than the full projection in ${k.b1BeatsP1} of ${k.n} noise conditions, so the full projection is not shown to be better.`
    : `The full projection had fewer false alerts than B1 in all ${k.n} noise conditions.`;
}
function b1Note(agg) {
  const k = b1Compare(agg);
  if (!k || !k.n) return "";
  const R = k.R, sens = (m) => pct(R[m].pause_change), clean = (m) => pct(R[m].clean);
  const lead = k.b1BeatsB0 >= Math.ceil(k.n / 2) && k.p1BeatsB1 < k.n
    ? "Our current results suggest that invariance training accounts for much of the observed noise robustness. The full projection method does not consistently outperform the simpler B1 approach under the existing evaluation. Further matched-operating-point experiments are needed to determine whether the additional projection contributes a reliable benefit."
    : k.p1BeatsB1 === k.n
      ? "In these results the full projection had fewer false alerts than the simpler B1 head in every noise condition; the intervals below show how large and how certain that difference is."
      : "In these results invariance training (B1) did not reduce false alerts relative to raw WavLM in most noise conditions.";
  return `<div class="callout info small" style="margin:12px 0"><h3 style="margin-top:0">What the B1 ablation shows</h3><p>${esc(lead)}</p>
    <ul style="margin:0">
      <li><strong>Invariance training vs raw WavLM:</strong> B1 had fewer false alerts than raw WavLM in ${k.b1BeatsB0} of ${k.n} noise conditions.</li>
      <li><strong>Full projection vs B1:</strong> P1 had fewer false alerts than B1 in ${k.p1BeatsB1} of ${k.n}; B1 had fewer in ${k.b1BeatsP1}.</li>
      <li><strong>Trade-off:</strong> clean speech flagged ${clean("B0")} (raw), ${clean("B1")} (B1), ${clean("P1")} (P1); synthetic pause change detected ${sens("B0")}, ${sens("B1")}, ${sens("P1")}.</li>
    </ul></div>`;
}

// ------------------------------------------------------------------ session-level chart (Home + Trends)
// One point per accepted check-in: session drift / session boundary. Never a window-level threshold.
function sessionChart(cks, { height = 240 } = {}) {
  if (!cks.length) return '<p class="muted">No check-ins in this period.</p>';
  const W = 760, H = height, L = 58, R = 40, T = 22, B = 46;   // R/T leave room for the last date and "acted" labels
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
    const above = !!(c.drift && c.drift.exceeds);          // the stored decision, never re-derived here
    const rev = c.review_flag && isClin() ? `<circle cx="${cx}" cy="${y(v)}" r="12" fill="none" stroke="var(--navy)" stroke-width="2"/>` : "";
    const tip = `${fmtDate(c.created_at)}: ${v.toFixed(2)} &times; boundary${above ? " (above boundary)" : ""}${c.review_flag && isClin() ? `, part of a review (${c.review_flag.state})` : ""}${c.simulated ? ", simulated" : ""}${c.acted ? ", acted" : ""}`;
    s += above
      ? `<g class="pt" data-ck="${esc(c.checkin_id)}" tabindex="0" style="cursor:pointer">${rev}<path d="M${cx} ${y(v) - 7}l7 7-7 7-7-7z" fill="var(--amber)"/><title>${tip}</title></g>`
      : `<g class="pt" data-ck="${esc(c.checkin_id)}" tabindex="0" style="cursor:pointer">${rev}<circle cx="${cx}" cy="${y(v)}" r="6" fill="var(--teal)"/><title>${tip}</title></g>`;
    if (c.acted) s += `<text x="${cx}" y="${y(v) - 12}" text-anchor="middle">acted</text>`;
  });
  s += `<path d="${path}" fill="none" stroke="var(--teal)" stroke-width="2" opacity=".55"/>`;
  s += `<text x="14" y="${(H - B) / 2}" transform="rotate(-90 14 ${(H - B) / 2})" text-anchor="middle">difference (&times; boundary)</text></svg>`;
  const acted = cks.some((c) => c.acted);
  const nAbove = cks.filter((c) => c.drift && c.drift.exceeds).length;
  const nRej = cks.filter((c) => c.quality.verdict === "REJECT").length;
  const nBase = cks.filter((c) => ratio(c) == null && c.quality.verdict !== "REJECT").length;
  return `<figure>${s}
    <div class="legend"><span>${icon("check", 12)} <span class="box" style="background:var(--teal);border-radius:50%"></span>within boundary</span>
      <span><span class="box" style="background:var(--amber);transform:rotate(45deg)"></span>above boundary</span>
      ${isClin() && cks.some((c) => c.review_flag) ? '<span><span class="box" style="border:2px solid var(--navy);border-radius:50%;background:none"></span>part of a review-queue item</span>' : ""}
      <span>&#9675; baseline / not compared</span><span>&times; not usable</span></div>
    <p class="small muted">Vertical axis: how far each check-in's voice is from your baseline, as a multiple of the usual-range boundary (1 = the edge of normal day-to-day variation). Horizontal axis: check-in date.
      ${vals.length} compared check-in${vals.length === 1 ? "" : "s"}, ${nAbove} above the boundary, ${nBase} baseline or not compared, ${nRej} not usable. Each point is one whole check-in (session level).
      ${isClin() ? "A single point above 1 is an observation; a review-queue item needs 2 of the last 3 usable check-ins above 1, on at least 2 different days within 7 days. " : "A single point above 1 can happen for many reasons, such as the room, the microphone or a slower morning. "}Select a point to see its measurements.</p>
    ${acted ? `<p class="small"><span class="mode synthetic_demo">acted</span> marks simulated days generated to differ from the baseline, to illustrate the review workflow. They are not measurements of real or acted speech, and nobody's real speech changed.</p>` : ""}</figure>`;
}

// ------------------------------------------------------------------ Check-in (patient)
async function viewCheckin() {
  const verify = new URLSearchParams(location.hash.split("?")[1] || "").get("verify");
  let promptId = "standard_reading_v1";
  if (verify) {   // a repeat must use the original's passage
    const orig = await api(`/checkins/${encodeURIComponent(verify)}`).catch(() => null);
    if (orig && orig.protocol) promptId = orig.protocol.prompt_id;
  }
  // Synthetic demo profiles can't take real recordings (different feature space and task): say so before recording
  const ptl = await timeline(S.pid, 1).catch(() => null);
  if (ptl && ptl.patient && ptl.patient.data_mode === "synthetic_demo") {
    $app.innerHTML = `<div class="card stack"><div class="eyebrow"><span class="line"></span>Acoustic check-in</div>
      <h1>${esc(ptl.patient.display_name)} is a synthetic demo profile</h1>
      <p>Its check-ins are generated for the demo, to illustrate the monitoring and review workflow, so a real recording can't be compared with them.
        To record, switch to the live profile. Its baseline is your own first five readings.</p>
      <p><button class="primary" id="tolive">${icon("mic")}Switch to Demo speaker (live) and record</button></p>
      <p class="small muted">To see ${esc(ptl.patient.display_name)}'s review, switch the view to Clinician at the top.</p></div>`;
    document.getElementById("tolive").onclick = () => { setPid(LIVE_PID); route(); };
    return;
  }
  const prompt = await api(`/prompt?prompt_id=${encodeURIComponent(promptId)}`);
  const reading = prompt.task_type === "read_speech";
  $app.innerHTML = `
    <div class="page-heading"><div><div class="eyebrow"><span class="line"></span>Acoustic check-in</div>
      <h1>${verify ? "Repeat the check-in." : "Take a voice check-in."}</h1>
      <p>${reading ? `${esc(INSTRUCTION)} It takes about 40 seconds.` : "This repeat uses the older free-speech task of the check-in it repeats, so it stays comparable."}</p></div></div>
    ${verify ? `<div class="callout info" style="margin-bottom:16px"><p style="margin:0">Read the <strong>same passage</strong> again, with the same microphone and in the same place if you can. We'll compare this recording with the one you are repeating.</p></div>` : ""}
    <div class="checkin-layout">
      <section class="reading-card" aria-labelledby="rp-h">
        <div class="reading-top"><span class="reading-icon">${icon("file", 20)}</span>01 / ${reading ? "READING PASSAGE" : "SPEAKING TASK"}</div>
        <h2 id="rp-h">${reading ? "The same passage,<br>every time." : "Speak about your day."}</h2>
        <p class="intro">${reading ? "Reading identical words each time keeps recordings comparable, so differences come from how you speak, not what you say." : "Kept for repeats of older check-ins only."}</p>
        <blockquote>${esc(prompt.text)}</blockquote>
        <p class="reading-foot">Passage ${esc(prompt.prompt_id)} &middot; version ${esc(prompt.prompt_version)} &middot; protocol ${esc(prompt.protocol_version)}</p>
      </section>
      <section class="card recorder-card stack" aria-labelledby="rec-h">
        <div><div class="eyebrow" style="justify-content:center">02 / Record your voice</div><h3 id="rec-h" style="font-size:22px;margin-top:8px">Ready when you are.</h3>
          <p class="small muted" style="margin:0">Find a quiet space and read the passage aloud.</p></div>
        <div class="ring"><svg width="190" height="190" viewBox="0 0 190 190" aria-hidden="true"><circle cx="95" cy="95" r="84" fill="none" stroke="var(--line)" stroke-width="10"/><circle id="arc" cx="95" cy="95" r="84" fill="none" stroke="var(--teal)" stroke-width="10" stroke-linecap="round" stroke-dasharray="527.8" stroke-dashoffset="527.8"/></svg><div class="t" id="clock" aria-hidden="true">1:00</div></div>
        <div class="meter" role="meter" aria-label="Microphone level" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" id="lvlwrap"><div id="lvl"></div></div>
        <p id="hint" class="small" aria-live="polite">&nbsp;</p>
        <div class="row" style="justify-content:center">
          <button class="primary big" id="rec">${icon("mic", 22)}Start recording</button>
          <button id="stop" disabled>Stop &amp; submit (after 0:${MIN_S})</button>
          <button id="cancel" disabled>Cancel</button>
        </div>
        ${DEMO ? `<div class="row"><button id="noisemark" disabled>Mark noise start</button><span class="small muted" id="noiselog">Demo: mark when background noise starts and stops.</span></div>` : ""}
        <p id="status" role="status" class="small muted">Ready. When you press Start, your browser will ask to use the microphone.</p>
      </section>
    </div>
    <section class="guidance" aria-label="Before you start">
      ${(prompt.instructions || []).slice(0, 3).map((x, i) => `<div><b>0${i + 1}</b><strong>${["Same voice", "Same distance", "Same words"][i] || "Tip"}</strong><p>${esc(x)}</p></div>`).join("")}
    </section>
    <div class="callout info small" style="margin-top:16px"><h3>${icon("shield", 16)} Your recording</h3>
      Audio is sent only to this computer's Voice Drift server, analysed there, and the audio file is deleted afterwards.
      ${reading ? "Reading-passage recordings are never sent to a language-model service." : "A written transcript may be sent to a language-model service to find quotes."}
      Please read the whole passage (about 40 seconds); the Stop button unlocks after ${MIN_S} seconds. See <a href="#/about">About &amp; Privacy</a>.</div>`;
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
  const finish = () => {
    cancelAnimationFrame(raf); clearInterval(timer); stream?.getTracks().forEach((t) => t.stop());
    // finish() runs on stop AND on navigation; closing an already-closed AudioContext throws InvalidStateError
    if (ctx && ctx.state !== "closed") ctx.close().catch(() => {});
    ctx = null;
  };
  // safe any number of times, and a no-op if recording never started (navigation away from an idle recorder)
  cleanup = () => {
    stopped = true; finish();
    const m = mr; mr = null;
    if (m && m.state !== "inactive") m.stop();
  };

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
    const mime = mr.mimeType || "audio/webm";        // captured now: cleanup() may null `mr` before onstop fires
    mr.onstop = () => { finish(); if (!stopped) upload(new Blob(chunks, { type: mime })); };
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
      if (el >= MAX_S && mr && mr.state !== "inactive") mr.stop();   // stop() on an inactive recorder throws
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
    fd.append("audio", blob, "checkin.webm"); fd.append("prompt_id", prompt.prompt_id);
    if (verify) fd.append("verification_of", verify);
    const end = +elapsed().toFixed(2);
    marks.forEach((m) => { if (m.end_seconds == null) m.end_seconds = end; });
    if (marks.length) fd.append("interventions", JSON.stringify(marks));
    $app.innerHTML = `<h1>Checking your recording</h1><div class="card"><ul class="steps" id="steps"></ul><p id="pstatus" role="status" class="small muted">Uploading&hellip;</p></div>`;
    try {
      const r = await fetch(`/api/patients/${encodeURIComponent(S.pid)}/checkins`, { method: "POST", body: fd });
      if (r.status === 409 || r.status === 404) {
        const j = await r.json().catch(() => ({}));
        const synthetic = String(j.detail || "").includes("synthetic demo profile");
        renderOutcome({ outcome: "failed", switchLive: synthetic,
          message: synthetic ? "This is a synthetic demo profile, so a real recording can't be compared with it. Switch to the live profile to record."
            : `${verify ? "This repeat" : "This recording"} couldn't be saved: ${j.detail || r.status}.` });
        return;
      }
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
    const again = r.verification_of ? `#/checkin?verify=${encodeURIComponent(r.verification_of)}` : "#/checkin";
    $app.innerHTML = `<h1>${r.switchLive ? "This profile can't take recordings" : r.outcome === "failed" ? "This check-in couldn't be analysed" : "Recording quality was insufficient. Please try again."}</h1>
      <div class="callout caution" role="alert"><h3>${r.switchLive ? "Synthetic demo profile" : r.outcome === "failed" ? "Something went wrong" : "We couldn't use this recording"}</h3><p>${esc(r.message)}</p>
        ${r.quality && (r.quality.reason_codes || []).length ? `<p class="tiny muted" style="margin:0">Reason: ${r.quality.reason_codes.map(esc).join(", ")}. This recording was not used for your baseline or any comparison.</p>` : ""}</div>
      <p style="margin-top:16px"><button class="primary big" id="again">${icon("mic", 22)}${r.switchLive ? "Switch to Demo speaker (live) and record" : "Record again"}</button></p>`;
    // a real button: works even when the address is already #/checkin (a same-hash link does nothing)
    document.getElementById("again").onclick = () => {
      if (r.switchLive) setPid(LIVE_PID);
      if (location.hash === again) route(); else location.hash = again;
    };
    return;
  }
  const prog = r.baseline_progress;
  $app.innerHTML = `<h1>${icon("check", 26)} Check-in recorded</h1>
    <div class="card stack">
      <p class="small">${icon("check", 16)} ${esc((r.quality && r.quality.message) || "Recording quality passed.")}</p>
      <p>${esc(r.message || "")}</p>
      ${prog ? `<div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="${prog[1]}" aria-valuenow="${prog[0]}" aria-label="Baseline progress"><div style="width:${(prog[0] / prog[1]) * 100}%"></div></div><p class="small muted">Baseline ${prog[0]} of ${prog[1]}</p>` : ""}
      <div id="verif"></div>
      <div class="row">
        ${r.has_window_analysis ? `<a class="btn" href="#/analysis/${encodeURIComponent(r.checkin_id)}">${icon("analysis")}View analysis</a>` : ""}
        <a class="btn" href="#/trends">${icon("trends")}View trends</a><a class="btn" href="#/">Home</a>
      </div>
      ${r.simulated ? '<span class="mode synthetic_demo">Synthetic demo data</span>' : ""}
    </div>
    ${r.acoustic_features ? `<section class="card stack" style="margin-top:20px">${acousticHtml(r.acoustic_features, r.acoustic_summary)}</section>` : ""}
    <section class="card stack" style="margin-top:20px">${contextFormHtml(r.context)}</section>`;
  wireContext(r.checkin_id);
  api(`/checkins/${encodeURIComponent(r.checkin_id)}/verification`).then((v) => {
    const el = document.getElementById("verif");
    if (el) el.innerHTML = verificationHtml(v, r.checkin_id, r.has_window_analysis);
  }).catch(() => {});
}

// ------------------------------------------------------------------ VD-01 one-click sample (shared by Home + Results)
function sampleCardHtml(res) {
  const sp = res && res.sample;
  const ok = sp && sp.present;
  const warmMsg = demoWarmMsg(res, res && res.sample_snapshot);
  return `<section class="card stack" aria-labelledby="smp-h">
    <div class="row"><h2 id="smp-h" style="margin:0">Try it without recording</h2><span class="spacer"></span>
      <button class="primary" id="samplebtn" ${ok ? "" : "disabled"}>${icon("analysis")}Analyze sample recording</button></div>
    <p class="small">${esc(INSTRUCTION)}</p>
    <p class="small muted">The sample is the standard passage read by a <strong>synthetic text-to-speech voice</strong>, not a person. It is analysed with the same pipeline as a real check-in, against a baseline built from five other readings by the same synthetic voice. The analysis is measured once in advance, so the result appears straight away.</p>
    ${ok ? "" : '<p class="small callout caution">The bundled sample audio is missing from this deployment, so the demo is unavailable. Everything else still works.</p>'}
    <p id="samplemsg" role="status" class="small">${warmMsg}</p><div id="sampleout"></div></section>`;
}
// While the server measures the demos in the background (first minute or two after start-up), say so plainly.
function demoWarmMsg(res, have) {
  const w = (res && res.demo_warm) || {};
  if (have) return "";
  if (w.state === "running") return "Preparing the demo in the background (usually under two minutes after the server starts). This page updates when it is ready.";
  if (w.state === "error") return `The demo couldn't be prepared (${esc(w.error || "error")}). Clicking the button will try again live.`;
  return "";
}
function provenance(j) {
  if (!j || !j.ran_at) return "";
  return `<p class="tiny muted" style="margin:6px 0 0">${j.cached ? "Measured" : "Measured just now"} ${j.cached ? `on ${esc(fmtDT(j.ran_at))} ` : ""}by this server with ${esc(j.model || "")}${j.compute_seconds != null ? `; the analysis took ${j.compute_seconds} s` : ""}. ${j.cached ? "Shown from that measurement, which is reused only while the code, model and audio are unchanged." : ""}
    <button type="button" class="linklike" data-fresh="${j.label ? "sample" : "replay"}">Re-run live</button></p>`;
}
function sampleResultHtml(j) {
  const c = j.checkin;
  const overall = c.drift_status === "unavailable" ? '<span class="muted">not available (no baseline or unusable recording)</span>'
    : `<strong>${fmt(c.drift_score)} &times; boundary</strong> ${statusPill(c.status)}`;
  return `<div class="callout info small" style="margin:0">${modeTag("demo_synthetic_voice")} ${esc(j.label)}${provenance(j)}</div>
    <dl class="meta">
      <dt>Recording quality</dt><dd>${qualityPill(c.quality.verdict)} ${fmt(c.speech_duration_seconds, 0)} s of speech</dd>
      <dt>Overall voice pattern</dt><dd>${overall}<br><span class="tiny muted">Session score relative to this speaker's boundary. Above 1.0 means elevated; exactly 1.0 counts as within range.</span></dd>
      ${isClin() ? `<dt>Review queue</dt><dd>${c.review_required ? "A review item exists for this pattern." : "No review item. A single recording can't create one: the rule needs 2 of the last 3 usable check-ins above the boundary, on 2 different days."}</dd>` : ""}
    </dl>
    ${acousticHtml(c.acoustic_features, c.acoustic_summary, { heading: "Individual measurements" })}
    <p class="tiny muted">Individual measurements and the overall pattern are separate: one measurement can change a lot while the overall pattern stays within range, and the other way round.</p>
    <p class="small"><a href="#/trends" id="sampletrends">See this sample speaker's trend &rarr;</a></p>`;
}
function wireSample(res) {
  const b = document.getElementById("samplebtn");
  if (!b) return;
  const msg = document.getElementById("samplemsg");
  const show = (j) => {
    document.getElementById("sampleout").innerHTML = sampleResultHtml(j);
    const t = document.getElementById("sampletrends");
    if (t) t.onclick = () => setPid(j.profile);
    const fr = document.querySelector('#sampleout [data-fresh="sample"]');
    if (fr) fr.onclick = () => run(true);
  };
  const run = async (fresh) => {
    b.disabled = true;                                    // no duplicate submissions while it runs
    msg.textContent = fresh ? "Re-running the analysis live. This takes about a minute with the full speech model." : "Loading the measured analysis…";
    try {
      const r = await fetch(`/api/demo/sample${fresh ? "?fresh=1" : ""}`, { method: "POST" });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || `server error ${r.status}`);
      msg.textContent = "";
      show(j);
    } catch (e) { msg.textContent = "The sample analysis didn't complete: " + e.message + ". You can try again."; }
    b.disabled = !(res && res.sample && res.sample.present);
  };
  b.onclick = () => run(false);
  // still being measured in the background: check every 5 s and say when it's ready (stops on navigation)
  if (!(res && res.sample_snapshot) && res && res.demo_warm && res.demo_warm.state === "running") {
    const iv = setInterval(async () => {
      const r2 = await api("/results").catch(() => null);
      if (!r2) return;
      if (r2.sample_snapshot || (r2.demo_warm && r2.demo_warm.state !== "running")) {
        clearInterval(iv);
        S.results = r2;
        msg.textContent = r2.sample_snapshot ? "The demo is ready. Click \"Analyze sample recording\"." : demoWarmMsg(r2, false);
      }
    }, 5000);
    const prev = cleanup;
    cleanup = () => { clearInterval(iv); prev(); };
  }
}

// ------------------------------------------------------------------ F4 acoustic evidence (shared)
const FEATURE_FMT = { pause_fraction: (v) => `${Math.round(v * 1000) / 10}%`, speaking_rate_wpm: (v) => `${Math.round(v)} wpm`, pitch_variability: (v) => `${fmt(v, 1)} st` };
const DELTA_FMT = { pause_fraction: (v) => `${v > 0 ? "+" : ""}${Math.round(v * 1000) / 10} points`, speaking_rate_wpm: (v) => `${v > 0 ? "+" : ""}${Math.round(v)} wpm`, pitch_variability: (v) => `${v > 0 ? "+" : ""}${fmt(v, 1)} st` };
function acousticHtml(F, summary, { heading = "What Changed in Your Speech?" } = {}) {
  if (!F) return `<h2>${heading}</h2><p class="muted small">No measurements for this check-in.</p>`;
  const row = (k) => {
    const f = F[k], ok = f.status === "VALID";
    const val = (v) => (v == null ? '<span class="muted">unavailable</span>' : FEATURE_FMT[k](v));
    return `<tr><td>${esc(f.label)}<br><span class="tiny muted">${esc(f.unit)}</span></td><td class="num">${val(f.baseline)}${f.baseline_n ? `<br><span class="tiny muted">mean of ${f.baseline_n}</span>` : ""}</td><td class="num">${val(f.current)}</td>
      <td class="num">${ok ? `<strong>${DELTA_FMT[k](f.delta)}</strong>${f.relative != null ? `<br><span class="tiny muted">${f.relative > 0 ? "+" : ""}${Math.round(f.relative * 100)}%</span>` : ""}` : `<span class="muted small">${esc((f.reason || f.status).toString())}</span>`}</td></tr>`;
  };
  return `<h2>${heading}</h2>
    ${(summary || []).length ? summary.map((x) => `<p>${esc(x)}</p>`).join("") : '<p class="muted small">No comparable measurement for this check-in.</p>'}
    <div class="tablewrap"><table><caption class="sr-only">Speech measurements, baseline vs this recording</caption><thead><tr><th>Measurement</th><th class="num">Your baseline</th><th class="num">This recording</th><th class="num">Difference</th></tr></thead>
      <tbody>${Object.keys(F).map(row).join("")}</tbody></table></div>
    <p class="tiny muted">Same measurement method for baseline and this recording. Pause time counts gaps of 200 ms or more between stretches of speech. These describe the recording; they are not symptoms and are not shown to explain the overall difference from baseline.</p>`;
}

// ------------------------------------------------------------------ F3 verification (shared)
const VRESULT = {
  REPRODUCED: ["Repeated: both recordings differ from your baseline", "navy"],
  NOT_REPRODUCED: ["Not repeated: only the first recording differed", "teal"],
  INCONCLUSIVE: ["Inconclusive", "gray"],
  UNCALIBRATED: ["Not classified (no calibrated boundary)", "gray"],
};
function verificationHtml(v, cid, hasAnalysis) {
  if (!v) return "";
  const side = (x, label) => `<tr><td>${label}<br><span class="tiny muted">${fmtDT(x.recorded_at)}</span></td><td>${qualityPill(x.quality)}</td><td class="num">${x.normalized_drift == null ? '<span class="muted">not scored</span>' : `${fmt(x.normalized_drift)} &times;`}</td><td>${x.exceeded == null ? "&ndash;" : x.exceeded ? "Above boundary" : "Within boundary"}</td></tr>`;
  if (v.repeats && v.repeats.length) {
    const last = v.repeats[v.repeats.length - 1], res = last.verification || {};
    const [label, tone] = VRESULT[res.result] || [res.result || "Pending", "gray"];
    return `<div class="callout info"><h3>Repeat check-in result <span class="pill ${tone}">${esc(label)}</span></h3>
      <div class="tablewrap"><table><thead><tr><th>Recording</th><th>Quality</th><th class="num">Difference (&times; boundary)</th><th>Session boundary</th></tr></thead><tbody>
        ${side(v.original, "Original")}${side(last, "Repeat")}</tbody></table></div>
      <p class="tiny muted" style="margin:6px 0 0">${res.reason ? `Reason: ${esc(res.reason)}. ` : ""}This says whether the measurement repeated under the same conditions. It is not a health assessment, and a same-day repeat counts as one episode, not two.</p></div>`;
  }
  const o = v.offer;
  if (!o || !o.available) return "";
  return `<div class="callout ${o.kind === "VERIFY" ? "info" : "caution"}"><h3>${esc(o.message)}</h3>
    ${o.kind === "VERIFY" ? '<p class="small">You can repeat the check-in now with the same passage to see whether the difference repeats.</p>' : ""}
    <div class="row"><a class="btn primary" href="#/checkin?verify=${encodeURIComponent(cid)}">${icon("mic")}Repeat Check-In</a>
    <a class="btn" href="${hasAnalysis ? `#/analysis/${encodeURIComponent(cid)}` : "#/trends"}">View Details</a></div></div>`;
}

// ------------------------------------------------------------------ F5 patient-reported context
const CONTEXT = [["TIRED", "Feeling tired"], ["COLD_OR_SORE_THROAT", "Cold or sore throat"], ["DIFFERENT_ROOM_OR_MIC", "Different microphone or room"], ["NOTICED_SPEECH_CHANGE", "Noticed a change in how I speak"], ["NONE", "No additional context"]];
const contextText = (c) => (c && ((c.tags || []).length || c.note) ? [...(c.tags || []).map((t) => (CONTEXT.find(([k]) => k === t) || [t, t])[1]), c.note ? `“${c.note}”` : ""].filter(Boolean).join(", ") : "");
function contextFormHtml(ctx) {
  const have = new Set((ctx && ctx.tags) || []);
  return `<h2>Is there anything that might help explain today's recording?</h2><p class="small muted">Optional. This is saved with the check-in for your reviewer. It doesn't change any measurement.</p>
    <fieldset style="border:0;padding:0;margin:0"><legend class="sr-only">Context</legend>${CONTEXT.map(([k, l]) => `<label class="row small" style="gap:8px"><input type="checkbox" name="ctx" value="${k}" ${have.has(k) ? "checked" : ""}> ${l}</label>`).join("")}</fieldset>
    <label for="ctxnote" class="small">Short note (optional)</label><input id="ctxnote" maxlength="300" value="${esc((ctx && ctx.note) || "")}">
    <div class="row"><button id="ctxsave">Save context</button><span id="ctxmsg" role="status" class="small muted"></span></div>`;
}
function wireContext(cid) {
  const b = document.getElementById("ctxsave");
  if (!b) return;
  b.onclick = async () => {
    const tags = [...document.querySelectorAll('input[name="ctx"]:checked')].map((x) => x.value);
    const msg = document.getElementById("ctxmsg");
    try {
      const j = await api(`/checkins/${encodeURIComponent(cid)}/context`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tags, note: document.getElementById("ctxnote").value }) });
      msg.textContent = "Saved.";
      if (j.safety && j.safety.triggered) {      // typed crisis language: same support message as the spoken path
        b.closest("section").insertAdjacentHTML("afterend", `<div class="support" role="alert" style="margin-top:20px"><p class="big">You are not alone.</p><p>${esc(j.safety.message)}</p><p class="big"><a href="tel:988">Call or text 988</a></p></div>`);
      }
    } catch (e) { msg.textContent = "Couldn't save (" + e.message + ")."; }
  };
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
  // Sanity check: by construction ~10% of CLEAN sessions should cross. If a model is far above that, its boundary
  // doesn't fit these speakers, so compare how much the noise ADDS on top of each model's own clean rate.
  const cr = h.clean_false_alert_raw, cp = h.clean_false_alert_projection;
  if (cr && cp && cr.rate != null && cp.rate != null && Math.max(cr.rate, cp.rate) > 0.2) {
    const add = (n, c) => Math.round((n.rate - c.rate) * 100);
    s += ` Caveat: on clean, unchanged speech the boundaries already flag ${pct(cr)} (raw) and ${pct(cp)} (projection), against a design target of about 10%, so at least one boundary doesn't fit these speakers.` +
         ` Measured against each model's own clean rate, the added noise raised false alerts by ${add(fr, cr)} points for raw and ${add(fp, cp)} points for the projection.`;
  }
  return s;
}
// schema 2 (matched operating point) and schema 1 (train-speaker boundary only) are both readable
const isV2 = (agg) => !!(agg && agg.schema >= 2);
const levelRates = (agg, level, op = "matched") => (isV2(agg) ? agg.results[level][op] : agg.results[level]);
const MODEL_NAME = { B0: "Raw WavLM (B0)", P1: "Projection (P1)", B1: "Projection, invariance only (B1)" };
const shouldAlert = (agg, c) => (agg.should_alert || ["pause_change"]).includes(c);
const pts = (x) => (x == null ? "&ndash;" : `${x > 0 ? "+" : ""}${Math.round(x * 100)}`);
function diffHtml(d) {
  if (!d || d.diff == null) return "&ndash;";
  return `<strong>${pts(d.diff)} pts</strong><br><span class="tiny muted">${pts(d.ci[0])} to ${pts(d.ci[1])} &middot; ${d.excludes_zero ? "interval excludes 0" : "interval includes 0"}</span>`;
}
function resultsTable(agg, level, { conds = null, op = "matched" } = {}) {
  const R = levelRates(agg, level, op), models = ["B0", "B1", "P1"].filter((m) => R[m]);
  const cs = (conds || Object.keys(R[models[0]])).filter((c) => R[models[0]][c]);
  const PD = isV2(agg) && level === "session" && op === "matched" ? agg.paired_differences["P1-B0"] : null;
  return `<table><caption class="sr-only">${level}-level alert rates</caption><thead><tr><th scope="col">Condition</th>${models.map((m) => `<th scope="col" class="num">${MODEL_NAME[m]}</th>`).join("")}${PD ? '<th scope="col" class="num">P1 minus raw (paired)</th>' : ""}</tr></thead><tbody>
    ${cs.map((c) => `<tr><td>${esc(COND_LABEL(agg, c))}<br><span class="tiny muted">${shouldAlert(agg, c) ? "share detected, higher is better" : c === "clean" ? "share flagged, target about 10%" : "share flagged (false alerts), lower is better"}</span></td>
      ${models.map((m) => `<td class="num"><strong>${pct(R[m][c])}</strong><br><span class="tiny muted">${ci(R[m][c])} &middot; n=${R[m][c].n}</span></td>`).join("")}
      ${PD ? `<td class="num">${diffHtml(PD.rate[c])}</td>` : ""}</tr>`).join("")}</tbody></table>`;
}
function aucTable(agg) {
  const A = agg.auc || {}, models = ["B0", "B1", "P1"].filter((m) => A[m]);
  const cs = models.length ? Object.keys(A[models[0]]) : [];
  if (!cs.length) return "";
  const PD = agg.paired_differences["P1-B0"];
  return `<h3>Threshold-free check: does changed speech rank above noise?</h3>
    <p class="small muted">AUC = the chance that a session with changed speech is further from baseline than the same kind of session with only noise added at 10 dB. 0.5 = no better than chance, 1 = always. No boundary is involved.</p>
    <div class="tablewrap"><table><thead><tr><th scope="col">Changed speech vs noise only</th>${models.map((m) => `<th scope="col" class="num">${MODEL_NAME[m]}</th>`).join("")}<th scope="col" class="num">P1 minus raw (paired)</th></tr></thead><tbody>
    ${cs.map((c) => { const cell = (r) => (r || {}).value; const iv = (r) => (r && r.ci && r.ci[0] != null && r.ci[1] != null ? `${r.ci[0].toFixed(2)}–${r.ci[1].toFixed(2)}` : "");
      const d = PD && PD.auc && PD.auc[c], dOk = d && d.diff != null, dIv = dOk && d.ci && d.ci[0] != null && d.ci[1] != null;
      return `<tr><td>${esc(COND_LABEL(agg, c))}</td>${models.map((m) => `<td class="num"><strong>${cell(A[m][c]) == null ? "&ndash;" : cell(A[m][c]).toFixed(2)}</strong><br><span class="tiny muted">${iv(A[m][c])}</span></td>`).join("")}
      <td class="num">${dOk ? `<strong>${d.diff > 0 ? "+" : ""}${d.diff.toFixed(2)}</strong><br><span class="tiny muted">${dIv ? `${d.ci[0].toFixed(2)} to ${d.ci[1].toFixed(2)} &middot; ${d.excludes_zero ? "excludes 0" : "includes 0"}` : "no interval"}</span>` : "&ndash;"}</td></tr>`; }).join("")}
    </tbody></table></div>`;
}
function sweepChart(agg) {
  const S = agg.noise_sweep;
  if (!S || !S.levels_db || !S.levels_db.length) return "";
  const W = 640, H = 260, L = 52, R = 150, T = 14, B = 44, lv = S.levels_db;
  const x = (i) => L + (lv.length === 1 ? 0.5 : i / (lv.length - 1)) * (W - L - R);
  const y = (v) => T + (1 - v) * (H - T - B);
  const style = { B0: "stroke:var(--coral)", B1: "stroke:var(--gray);stroke-dasharray:6 4", P1: "stroke:var(--teal)" };
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="False-alert rate on unchanged speech as added noise gets louder, per method, at the matched operating point">`;
  for (let v = 0; v <= 1.001; v += 0.25) s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
  lv.forEach((d, i) => { s += `<text x="${x(i)}" y="${H - B + 18}" text-anchor="middle">${d} dB</text>`; });
  s += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">signal-to-noise ratio (lower = louder noise)</text>`;
  s += `<text transform="translate(12 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">false alerts</text>`;
  Object.entries(S.by_method).forEach(([m, arr]) => {
    const p = arr.map((r, i) => (r && r.rate != null ? `${x(i)},${y(r.rate)}` : null)).filter(Boolean);
    if (!p.length) return;
    s += `<polyline fill="none" stroke-width="2.5" style="${style[m] || ""}" points="${p.join(" ")}"/>`;
    arr.forEach((r, i) => { if (r && r.rate != null) s += `<circle cx="${x(i)}" cy="${y(r.rate)}" r="3.5" style="fill:currentColor;${(style[m] || "").replace("stroke:", "color:")}"><title>${MODEL_NAME[m]}, ${lv[i]} dB: ${pct(r)} (${ci(r)})</title></circle>`; });
    const last = arr[arr.length - 1];
    if (last && last.rate != null) s += `<text x="${W - R + 8}" y="${y(last.rate) + 4}">${MODEL_NAME[m]}</text>`;
  });
  return `<h3>Noise sweep</h3><p class="small muted">Unchanged speech with held-out noise or babble added at each level. Each model at its matched boundary. Hover a point for its interval.</p>${s}</svg>`;
}
function matchedNote(agg) {
  if (!isV2(agg)) return "";
  const M = agg.matched_operating_point.by_method;
  const cells = Object.entries(M).map(([m, v]) => `${MODEL_NAME[m]} ${pct(v.session.clean_alert_rate)}${v.session.within_tolerance ? "" : " (outside 5–15%)"}`);
  return `<p class="small"><strong>Matched operating point.</strong> ${esc(agg.matched_operating_point.procedure)} Achieved clean alert rates: ${cells.join(", ")}.</p>`;
}
function verdictV2(agg) {
  const h = agg.headline, pd = h.primary_difference, cd = h.change_difference;
  if (!pd || pd.diff == null) return "";
  const M = agg.matched_operating_point.by_method;
  const offTarget = Object.keys(M).filter((m) => !M[m].session.within_tolerance);
  let s = pd.excludes_zero
    ? (pd.diff < 0 ? `At the matched operating point the projection produced fewer false alerts under 10 dB noise (${pts(pd.diff)} points, interval ${pts(pd.ci[0])} to ${pts(pd.ci[1])}).`
                   : `At the matched operating point the projection produced MORE false alerts under 10 dB noise (${pts(pd.diff)} points).`)
    : `At the matched operating point the difference in false alerts under 10 dB noise is ${pts(pd.diff)} points, but its interval (${pts(pd.ci[0])} to ${pts(pd.ci[1])}) includes zero, so it is not established with ${agg.meta.eval_speakers} speakers.`;
  void cd;
  // each speech change, with its own rates and its own threshold-free check (never mixing conditions)
  const R = agg.results.session.matched, PD = agg.paired_differences["P1-B0"] || { rate: {}, auc: {} };
  const NAME = { pause_change: "synthetic pauses", delivery_change: "20% slower delivery" };
  const f2 = (x) => x.toFixed(2);
  for (const c of ["pause_change", "delivery_change"]) {
    if (!R.B0[c] || !R.P1[c]) continue;
    const rB = R.B0[c].rate, rP = R.P1[c].rate, near = rB - R.B0.clean.rate <= 0.10 && rP - R.P1.clean.rate <= 0.10;
    const aB = agg.auc.B0 && agg.auc.B0[c], aP = agg.auc.P1 && agg.auc.P1[c], ad = PD.auc[c], dr = PD.rate[c];
    if (near) {
      s += ` For ${NAME[c]}, neither model alerted more than on unchanged speech (${pct(R.B0[c])} raw, ${pct(R.P1[c])} projection)`;
      if (aB && aP) s += `, and neither separated it from noise-only sessions better than chance (AUC ${f2(aB.value)} and ${f2(aP.value)}, where 0.5 is chance). Averaging each 8 s window largely removes speaking rate, which is why speaking rate is reported as a separate measurement`;
      s += ".";
      continue;
    }
    s += ` For ${NAME[c]}, the projection alerted ${dr && dr.diff < -0.05 ? `less often (${pct(R.P1[c])} vs ${pct(R.B0[c])}${dr.excludes_zero ? "" : ", interval includes zero"})` : `about as often (${pct(R.P1[c])} vs ${pct(R.B0[c])})`}`;
    if (aB && aP && ad && ad.diff != null) {
      s += Math.max(aB.value, aP.value) < 0.6
        ? `, and neither model separated it from noise-only sessions better than chance (AUC ${f2(aB.value)} and ${f2(aP.value)}).`
        : ad.diff < 0
          ? `; without any boundary, raw WavLM separated it from noise-only sessions at least as well (AUC ${f2(aB.value)} vs ${f2(aP.value)}, difference interval ${f2(ad.ci[0])} to ${f2(ad.ci[1])}${ad.excludes_zero ? "" : ", includes zero"}). The fewer false alerts come largely with lower sensitivity: a trade-off, not a clear improvement.`
          : `; without any boundary, the projection separated it from noise-only sessions better (AUC ${f2(aP.value)} vs ${f2(aB.value)}, difference interval ${f2(ad.ci[0])} to ${f2(ad.ci[1])}${ad.excludes_zero ? "" : ", includes zero"}).`;
    } else s += ".";
  }
  if (offTarget.length) s += ` Note: the clean alert rate for ${offTarget.join(", ")} fell outside 5–15%, so the operating points are only approximately matched.`;
  return s;
}
async function viewResults() {
  if (!resultsVisible()) {
    $app.innerHTML = `<h1>Results</h1><div class="card"><p>The evidence section is hidden in this view${resultsMode === "hide" ? " (switched off on this browser)" : " because no results file is present on this server"}.</p>
      <p class="small muted">Change it under <a href="#/about">About &amp; Privacy &rarr; Presentation</a>.</p></div>`;
    return;
  }
  const res = S.results || await api("/results");
  const agg = res.aggregate, h = agg && agg.headline, snap = res.demo_snapshot, d = res.demo;
  const canRun = d.clip_present && d.profile_baseline_ready;
  const why = !d.clip_present ? "The bundled demo audio is missing from this deployment, so this demo is unavailable. Everything else still works."
    : !d.profile_baseline_ready ? `The clip is compared with the baseline of <strong>${esc(d.profile)}</strong>, which needs 5 good check-ins from the same speaker first.` : "";
  const v2 = isV2(agg);
  const ca = res.calibration_artifacts || {};
  $app.innerHTML = `<div class="page-heading"><div><div class="eyebrow"><span class="line"></span>Transparency first</div>
      <h1>Does the projection ignore the room but not the speaker?</h1>
      <p>These results measure robustness to recording noise and sensitivity to synthetic speech changes. They are not measurements of how well depression or any other condition is monitored.</p></div></div>
    ${h && v2 ? `<p class="lead" style="max-width:80ch"><strong>${esc(h.sentence)}</strong></p>
        <div class="callout info" style="margin-bottom:20px"><p style="margin:0">${esc(verdictV2(agg))}</p></div>`
    : h ? `<p class="lead" style="max-width:80ch"><strong>Under ${esc(h.noise_condition)}, raw WavLM flagged ${pct(h.false_alert_raw)} of unchanged sessions as different; the noise-invariant projection flagged ${pct(h.false_alert_projection)}.
        For a controlled change in pausing, raw WavLM detected ${pct(h.change_sensitivity_raw)} and the projection ${pct(h.change_sensitivity_projection)}.</strong></p>
        <div class="callout info" style="margin-bottom:20px"><p style="margin:0">${esc(verdict(h))}</p>
        <p class="small" style="margin:8px 0 0"><strong>${esc(UNMATCHED_CAVEAT)}</strong> These figures use the original boundary rule, under which the models' clean alert rates differ.</p></div>`
      : `<div class="callout caution" style="margin-bottom:20px"><h3>Results pending</h3><p class="small">The aggregate experiment hasn't been run on this machine's data yet. On Colab: <code>python -m research.lite.aggregate</code>, then copy <code>results/aggregate.json</code> into the project. Nothing on this page is filled in until then.</p></div>`}

    <div style="margin-bottom:20px">${sampleCardHtml(res)}</div>
    <section class="card stack"><div class="row"><h2 style="margin:0">Same recording, with and without noise</h2><span class="spacer"></span>
      <button class="primary" id="rundemo" ${canRun ? "" : "disabled"}>${icon("analysis")}Run the noise demo</button></div>
      <p class="small muted">A bundled reading${d.baseline_auto ? " (synthetic text-to-speech voice)" : ""} is analysed twice, as recorded and with ${esc(d.noise.replace("_", " "))} noise added from ${d.start_s} to ${d.end_s} s at ${d.snr_db} dB, against the same voice's baseline. One example: the numbers below are the evidence. ${why}</p>
      <p id="demomsg" role="status" class="small"></p>
      <div id="demoout">${snap ? demoSnapshotHtml(snap) : '<p class="muted small">No demo has been run yet.</p>'}</div>
    </section>

    ${agg && v2 ? `<section class="card stack" style="margin-top:20px"><h2>The numbers</h2>
      <p class="small muted">Session level (3 windows, about 24 s, the way the app scores a check-in). 95% intervals: bootstrap over ${agg.meta.eval_speakers} held-out speakers, the same resample for every model, so differences are paired.
        Primary comparison, fixed on ${esc(agg.primary_comparison.fixed_on)} before the rerun: ${agg.primary_comparison.methods.join(" vs ")}, ${esc(agg.primary_comparison.level)} level, ${esc(COND_LABEL(agg, agg.primary_comparison.condition))}, matched operating point.</p>
      ${matchedNote(agg)}
      <p class="tiny muted">Calibration files: ${ca.present ? (ca.consistent ? `present and from the same run (commit ${esc((ca.source_commit || "unknown").slice(0, 10))})` : "<strong>present but from a different run than these results: not used</strong>") : "not present for this run"}.</p>
      <div class="tablewrap">${resultsTable(agg, "session", { conds: ["clean", "noise_10dB", "pause_change", "delivery_change"] })}</div>
      ${b1Note(agg)}
      ${aucTable(agg)}
      ${sweepChart(agg)}
      <details><summary>All conditions, session level (noise mix, change plus noise, every noise level)</summary><div class="tablewrap">${resultsTable(agg, "session")}</div></details>
      <details><summary>Window level (single 8 s windows)</summary><div class="tablewrap">${resultsTable(agg, "window")}</div></details>
      <details><summary>Original boundary rule (training speakers only; clean rates not matched)</summary><div class="tablewrap">${resultsTable(agg, "session", { op: "train_boundary" })}</div></details>
      <p class="tiny muted">Generated ${esc(agg.generated_at)} &middot; WavLM layer ${agg.meta.wavlm_layer} &middot; ${esc(agg.meta.distance)} distance &middot; boundary = ${Math.round(agg.meta.boundary_quantile * 100)}th percentile of normal variation.</p></section>`
    : agg ? `<section class="card" style="margin-top:20px"><h2>The numbers</h2>
      <p class="small muted">Session level (3 windows, about 24 s, the way the app scores a check-in). Each model is judged against its own boundary, set from <em>training</em> speakers' clean audio only. 95% intervals: bootstrap over ${agg.meta.eval_speakers} held-out speakers.</p>
      <p class="small callout caution">${esc(UNMATCHED_CAVEAT)}</p>
      <div class="tablewrap">${resultsTable(agg, "session")}</div>
      ${b1Note(agg)}
      <details><summary>Window level (single 8 s windows)</summary><div class="tablewrap">${resultsTable(agg, "window")}</div></details>
      <p class="tiny muted">Generated ${esc(agg.generated_at)} &middot; WavLM layer ${agg.meta.wavlm_layer} &middot; ${esc(agg.meta.distance)} distance &middot; boundary = ${Math.round(agg.meta.boundary_quantile * 100)}th percentile of normal variation.</p></section>` : ""}

    <div class="grid half" style="margin-top:20px">
      <section class="card stack"><h2>Method</h2><ul class="small">
        <li>Frozen <strong>WavLM-base</strong> speech encoder; a small <strong>projection</strong> (768&rarr;128) trained, without diagnosis labels, to give the same output for clean and noisy versions of the same speech (invariance, variance and covariance terms) while still predicting pause fraction, pitch variability and speech timing.</li>
        <li>Training noise: MUSAN noise, music and babble, simulated rooms and microphone coloring. Evaluation noise comes from <strong>different noise files</strong> and <strong>different speakers</strong>.</li>
        <li>Encoder layer: WavLM layer ${agg ? agg.meta.wavlm_layer : 6}, mean-pooled over each 8 s window.${agg ? ` ${agg.meta.eval_speakers} held-out evaluation speakers${agg.meta.train_speakers || agg.meta.train_speakers_for_boundary ? `; ${agg.meta.train_speakers || agg.meta.train_speakers_for_boundary} training speakers` : ""}.` : ""}</li>
        <li><strong>Three models.</strong> ${agg && agg.models ? Object.entries(agg.models).map(([m, t]) => `<strong>${m}</strong>: ${esc(t)}`).join(" ") : "<strong>B0</strong>: raw WavLM. <strong>B1</strong>: projection trained for noise invariance only. <strong>P1</strong>: B1 plus acoustic preservation (it must still predict pause fraction, pitch variability and speech-run rate)."}
          B1 is the simpler head. P1's extra term is meant to stop real delivery changes from being projected away along with the noise; the tables show whether that matters here.</li>
        <li>A session's difference from baseline is the cosine distance from the mean of that speaker's other sessions. Crossing the model's boundary counts as an alert.${agg && isV2(agg) ? " Each model's boundary is set so that it flags about 10% of clean, unchanged sessions (leave-one-speaker-out), so all models are compared at the same clean alert rate." : ""}</li>
        <li>Controlled speech changes: short pauses of the recording's own room tone inserted after each phrase${agg && isV2(agg) && agg.results.session.matched.B0.delivery_change ? ", and delivery slowed by 20% without changing pitch" : ""}. The words and recording conditions stay the same. Both are synthetic.</li>
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
      <li>Cummins, Scherer, Krajewski, Schnieder, Epps &amp; Quatieri. A review of depression and suicide risk assessment using speech analysis. Speech Communication 71, 2015.</li>
      <li>Low, Bentley &amp; Ghosh. Automated assessment of psychiatric disorders using speech: a systematic review. Laryngoscope Investigative Otolaryngology 5(1), 2020.</li>
      <li>Fagherazzi, Fischer, Ismael &amp; Despotovic. Voice for health: the use of vocal biomarkers from research to clinical practice. Digital Biomarkers 5(1), 2021.</li>
    </ol></section>

    <details class="card" style="margin-top:20px"><summary>Try your own recording (controlled experiment)</summary>${await labFormHtml()}</details>`;
  wireLab();
  wireSample(res);
  const runDemo = async (fresh) => {
    const btn = document.getElementById("rundemo");
    btn.disabled = true;
    const msg = document.getElementById("demomsg");
    msg.textContent = fresh ? "Re-running live: the clip is analysed clean and with noise (about a minute with the full speech model)…" : "Loading the measured analysis…";
    try {
      const r = await fetch(`/api/demo/run${fresh ? "?fresh=1" : ""}`, { method: "POST" });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.status);
      msg.textContent = "";
      document.getElementById("demoout").innerHTML = demoSnapshotHtml(j);
      wireSnapshot(j);
    } catch (err) { msg.textContent = "The demo failed: " + err.message + ". You can try again."; }
    btn.disabled = false;
  };
  document.getElementById("rundemo").onclick = () => runDemo(false);
  wireReplayFresh = () => runDemo(true);
  if (snap) wireSnapshot(snap);
  if (!snap) { const m = document.getElementById("demomsg"); if (m && !m.textContent) m.textContent = demoWarmMsg(res, false); }
}
let wireReplayFresh = () => {};
function demoSnapshotHtml(j) {
  const n = j.noise;
  return `${provenance({ ...j, cached: j.cached !== false })}
    ${analysisBlock(j.noisy, { key: "dnoisy", title: `With ${n.kind.replace("_", " ")} noise, ${n.start_seconds}–${n.end_seconds} s at ${n.snr_db} dB`, recordingLabel: "Bundled demo clip + noise" })}
    <details class="card"><summary>Same clip with no noise added</summary>${analysisBlock(j.clean, { key: "dclean", recordingLabel: "Bundled demo clip" })}</details>`;
}
function wireSnapshot(j) {
  wireAnalysis(j.noisy, "dnoisy"); wireAnalysis(j.clean, "dclean");
  const fr = document.querySelector('#demoout [data-fresh="replay"]');
  if (fr) fr.onclick = () => wireReplayFresh();
}

// ------------------------------------------------------------------ Trends (session level)
async function viewTrends() {
  const days = Number(new URLSearchParams(location.hash.split("?")[1] || "").get("days")) || 30;
  const tl = await timeline(S.pid, Math.min(days, 365));
  const cks = tl ? tl.checkins : [];
  const sp = (d, label) => `<a href="#/trends?days=${d}" ${d === days ? 'aria-current="page"' : ""}>${label}</a>`;
  const chips = cks.filter((c) => (c.timing_chips || []).length).slice(-6).reverse();
  $app.innerHTML = `<div class="eyebrow"><span class="line"></span>Trends &amp; signals</div><h1>Voice trends</h1>
    <p class="lead">One point per check-in, compared with your baseline. These describe your speech and recordings; they are not health measurements.</p>
    <nav class="tabs" aria-label="Time range">${sp(7, "7 days")}${sp(30, "30 days")}${sp(90, "90 days")}${sp(365, "All")}</nav>
    ${!tl ? '<div class="card muted">No check-ins on this profile yet.</div>' : `
    <section class="card"><h2>Difference from baseline</h2>${tl.baseline ? sessionChart(cks) : `<p class="muted">Shown once your baseline is ready.</p>`}</section>
    <div id="ptdetail"></div>
    <div class="grid half" style="margin-top:20px">
      <section class="card"><h2>Speech timing</h2>
        ${chips.length ? chips.map((c) => `<p class="small"><strong>${fmtDate(c.created_at)}:</strong> ${c.timing_chips.map((x) => esc(sentence(x.text))).join(" ")}</p>`).join("") : '<p class="muted small">No timing differences of note, or timing not available (needs transcription).</p>'}
        <div class="tablewrap">${timingTable(cks)}</div></section>
      <section class="card"><h2>Recording quality</h2><div class="tablewrap"><table><thead><tr><th>Date</th><th>Quality</th><th class="num">Speech (s)</th><th class="num">SNR (dB)</th></tr></thead><tbody>
        ${cks.slice().reverse().map((c) => `<tr><td>${fmtDT(c.created_at)}</td><td>${qualityPill(c.quality.verdict)}</td><td class="num">${fmt(c.quality.speech_s, 0)}</td><td class="num">${fmt(c.quality.snr_db, 0)}</td></tr>`).join("")}
      </tbody></table></div></section>
    </div>`}`;
  wireChartPoints(cks, "ptdetail");
}
// VD-09: selecting a chart point shows that check-in's own stored measurements
function wireChartPoints(cks, targetId) {
  $app.querySelectorAll("g.pt[data-ck]").forEach((g) => {
    const show = () => {
      const c = cks.find((x) => x.checkin_id === g.dataset.ck);
      const el = document.getElementById(targetId);
      if (c && el) { el.innerHTML = checkinDetail(c); el.scrollIntoView({ behavior: "smooth", block: "nearest" }); }
    };
    g.onclick = show; g.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); show(); } };
  });
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
  const res = S.results || {};
  const team = res.team || [];
  $app.innerHTML = `<div class="eyebrow"><span class="line"></span>How it works</div><h1>About &amp; Privacy</h1>
    <section class="card stack" style="margin-bottom:20px" aria-labelledby="why-h">
      <h2 id="why-h">The problem</h2>
      <p>People go weeks between appointments, and changes in how they are doing can go unseen until the next visit. Questionnaires are infrequent and depend on someone noticing and reporting the change themselves.
        Voice is a low-effort signal that could be checked in between, but two things get in the way: a noisy kitchen or a new phone can change a recording as much as the speaker does, and population "detectors" trained to label a voice are a different question from whether this person has changed.</p>
      <h2>What Voice Drift does</h2>
      <p><strong>It tells a care team when someone's voice has changed from their own normal, persistently, and not because of the room, so a person can check in sooner.</strong></p>
      <ol class="small">
        <li><strong>Measure:</strong> how far today's reading is from the person's own first five good readings, in units of their own variability, after checking recording quality and suppressing room noise.</li>
        <li><strong>Decide:</strong> whether the difference persists (2 of the last 3 usable check-ins, on separate days), rather than a single unusual day.</li>
        <li><strong>Route:</strong> a persistent difference becomes a nonurgent item for a human to review. Nothing is diagnosed and no risk label is shown to the person.</li>
      </ol>
      <h2>What it is not</h2>
      <p class="small">It does not decide whether anyone has depression or any other condition. A depression classifier was trained as an experiment and is deliberately not used: a population classifier answers a different question, and on this research corpus its accuracy may reflect recording differences rather than illness.
        A change from someone's baseline has <strong>not yet been shown to track a change in their mental health</strong>; that needs a longitudinal study with clinician-rated outcomes.</p>
      <h2>What the evidence shows so far</h2>
      <ul class="small">
        <li>Room noise is largely ignored: at about the same clean-speech alert rate, added noise triggered far fewer false alerts with the projection than with raw WavLM (see <a href="#/results">Results</a>).</li>
        <li>Some speech changes register (inserted pauses), at the cost of lower sensitivity than raw WavLM.</li>
        <li>Slowed speech does not register in the overall score; speaking rate is reported as a separate measurement.</li>
      </ul>
    </section>
    <div class="grid half">
      <section class="card stack"><h2>Who it's for</h2>
        <p class="small">People whose speech is worth following between clinic visits, under a clinician's supervision, and the researchers studying whether that is reliable.
          Voice Drift is a research prototype; it measures speech and recording conditions and does not diagnose anything.</p>
        <h2>How it works</h2><ol class="small">
          <li>A short check-in: reading the same passage aloud (about 40 seconds), recorded in the browser.</li>
          <li>A quality check rejects recordings with too little speech, too much noise or distortion.</li>
          <li>A frozen speech model (WavLM) turns the voice into numbers; a small projection, trained without any diagnosis labels, aims to ignore the room and microphone while keeping how the person speaks.</li>
          <li>Each check-in is compared with the person's own first five good ones, using a boundary set in advance from other people's normal day-to-day variation (about 1 in 10 ordinary days crosses it).</li>
        </ol></section>
      <section class="card stack"><h2>Limitations</h2><ul class="small">
          <li>Evaluated on a small research corpus of Italian interview speech, with synthetic speech changes. See <a href="#/results">Results</a> for the numbers and their uncertainty.</li>
          <li>Day-to-day variation in real homes is larger than in the evaluation data.</li>
          <li>The overall score does not detect slowed speech; speaking rate is shown separately.</li>
          <li>A difference from baseline is not evidence of any health condition, and not a clinical alert.</li>
        </ul>
        <h2>What's next</h2><ul class="small">
          <li>Real (not synthetic) speech-change recordings, and more speakers and languages.</li>
          <li>Recalibration on recordings made the way the app records them.</li>
          <li>Clinician-supervised pilot with consent, reporting, and sharing controls.</li>
        </ul>
        <h2>Team</h2>${team.length ? team.map((m) => (typeof m === "string" ? `<p class="small"><strong>${esc(m)}</strong></p>`
          : `<div class="small" style="margin-bottom:12px"><p style="margin:0"><strong>${esc(m.name)}</strong>${m.affiliation ? `<br><span class="muted">${esc(m.affiliation)}</span>` : ""}</p>${m.bio ? `<p style="margin:4px 0 0">${esc(m.bio)}</p>` : ""}</div>`)).join("")
          : ""}<p class="tiny muted">Built at HackPrinceton 2026.</p></section>
    </div>
    <div class="grid half" style="margin-top:20px">
      <section class="card stack"><h2>${icon("shield")} How your data is handled</h2>
        <dl class="meta small">
          <dt>Audio capture</dt><dd>In your browser, only while you are recording.</dd>
          <dt>Where analysis runs</dt><dd>On this computer's Voice Drift server (127.0.0.1). Audio is uploaded only to it.</dd>
          <dt>Audio retention</dt><dd>The audio file is deleted when analysis finishes, including when it fails.</dd>
          <dt>What is kept</dt><dd>Recording-quality results, comparison scores, per-window speech features (not audio), a coarse loudness summary, and the transcript, in a local database on this computer.</dd>
          <dt>Transcripts</dt><dd>${h.llm_reachable ? "Sent to Anthropic's API to find quotes for the evidence view. Audio is never sent." : "Not sent anywhere: no language-model key is configured."}</dd>
          <dt>Who can see results</dt><dd>Anyone using this app on this computer. There are no accounts, and nothing is sent to a clinician automatically.
            The "Viewing as: Patient / Clinician" switch at the top only changes which pages and wording are shown, for the demo; it is not a login and does not restrict access.</dd>
        </dl>
        <p class="small muted">Derived features are not claimed to be anonymous; they stay on this computer.</p></section>
      <section class="card stack"><h2>Your data</h2>
        <p class="small">Delete every check-in, score and transcript stored for the current profile (<strong>${esc(S.pid)}</strong>). This can't be undone.</p>
        <p><button id="del">Delete this profile's data</button> <span id="delmsg" role="status" class="small"></span></p>
        <hr><h2>Demo &amp; research</h2>
        <p class="small muted">Synthetic profiles are labelled "Synthetic demo data" everywhere they appear and are kept separate from real recordings.</p>
        <div class="row">${["all", "stable", "drift", "blank"].map((s) => `<button data-reset="${s}">Load demo: ${s}</button>`).join("")}</div>
        <fieldset class="stack" style="border:1px solid var(--line);border-radius:10px;padding:12px 14px"><legend><strong>Presentation: results section</strong></legend>
          <p class="small muted" style="margin:0">Results file on this server: <strong>${S.results && S.results.aggregate ? `found (generated ${esc(S.results.aggregate.generated_at)})` : "not found"}</strong>. This switch only changes what this browser shows.</p>
          ${[["auto", "Automatic: show only if a results file is present"], ["show", "Always show the Results tab and headline"], ["hide", "Hide results (plain product view)"]]
            .map(([v, l]) => `<label class="check"><input type="radio" name="rmode" value="${v}" ${resultsMode === v ? "checked" : ""}> ${l}</label>`).join("")}
        </fieldset>
        <p><a href="#/research">${icon("report", 16)} Research workspace</a> <span class="small muted">(all profiles, transcripts, evidence quotes)</span></p>
        <p><a href="#/review">${icon("report", 16)} Review queue</a> <span class="small muted">(speech-change reviews; demo, in-app only)</span></p>
      </section>
    </div>`;
  document.getElementById("del").onclick = async () => {
    if (!confirm(`Delete all stored data for ${S.pid}? This can't be undone.`)) return;
    try {
      await api(`/patients/${encodeURIComponent(S.pid)}`, { method: "DELETE" });
      document.getElementById("delmsg").textContent = "Deleted.";
    } catch (e) { document.getElementById("delmsg").textContent = offlineMsg(e); }
  };
  $app.querySelectorAll('input[name="rmode"]').forEach((r) => (r.onchange = () => { setResultsMode(r.value); route(); }));
  $app.querySelectorAll("[data-reset]").forEach((b) => (b.onclick = async () => {
    b.disabled = true;
    try {
      await post("/demo/reset", { scenario: b.dataset.reset });
      S.health = null; route();
    } catch (e) {
      b.disabled = false;
      b.parentElement.querySelector(".resetmsg")?.remove();
      b.insertAdjacentHTML("afterend", `<span class="resetmsg small" role="alert" style="margin-left:8px;color:var(--amber-ink)">${esc(offlineMsg(e))}</span>`);
    }
  }));
}

// ------------------------------------------------------------------ Review queue (alerting spec v1.0, reviewer-facing)
// A flag is a nonurgent, experimental request for human review: no risk levels, sorted by time, never by score.
const FLAG_STATUS = { OPEN: ["Open", "navy", "info"], ACKNOWLEDGED: ["Acknowledged", "teal", "check"], REVIEWED: ["Reviewed", "teal", "check"],
  CLOSED: ["Closed", "gray", "check"], DISMISSED: ["Dismissed", "gray", "x"] };
const flagPill = (s) => pill(FLAG_STATUS[s] || [s, "gray", "info"]);
async function viewReviewQueue() {
  const show = new URLSearchParams(location.hash.split("?")[1] || "").get("status") || "pending";
  const q = { pending: "?status=OPEN,ACKNOWLEDGED", reviewed: "?status=REVIEWED,CLOSED,DISMISSED", all: "" }[show] ?? "";
  const flags = await api(`/v1/review-flags${q}`);
  const tab = (k, l) => `<a href="#/review?status=${k}" ${show === k ? 'aria-current="page"' : ""}>${l}</a>`;
  const empty = show === "pending"
    ? "No pending reviews. An item appears only when 2 of a person's last 3 usable check-ins are above their boundary on 2 different days within 7 days; single unusual recordings and rejected recordings never create one. The seeded \"drift\" demo profile shows an example (About &amp; Privacy &rarr; reset demo data)."
    : "Nothing here yet.";
  $app.innerHTML = `<div class="eyebrow"><span class="line"></span>Review with context</div><h1>Review queue</h1>
    <p class="lead">Speech-change reviews raised by a persistence rule: at least 2 of the last 3 quality-passing check-ins above the person's own boundary, on separate days, within 7 days. This is an experimental heuristic, not a clinical decision rule. Items come from the same stored decisions shown on Home and Trends.</p>
    <div class="callout info small" style="margin-bottom:16px">${icon("shield", 16)} Demo mode: this queue is internal to the app. Nothing is sent to anyone outside it, and no real clinician is contacted.</div>
    <nav class="tabs" aria-label="Filter">${tab("pending", "Pending")}${tab("reviewed", "Reviewed")}${tab("all", "All")}</nav>
    <div class="card tablewrap"><table><caption class="sr-only">Review items, newest first</caption><thead><tr><th>Person</th><th>Triggering check-in</th><th class="num">Score / boundary</th><th class="num">Check-ins above boundary</th><th>Reason</th><th>Supporting changes</th><th>State</th><th>Data</th><th></th></tr></thead><tbody>
      ${flags.map((f) => { const t = f.trigger || {}; const sup = [...(t.acoustic_summary || []), ...(t.timing_chips || []).map((x) => sentence(x.text))]; return `<tr><td>${esc(f.patient_display_name)}</td>
        <td class="small">${t.recorded_at ? fmtDT(t.recorded_at) : "&ndash;"}<br><span class="tiny muted">${esc(t.checkin_id || "")}</span></td>
        <td class="num"><strong>${fmt(t.normalized_score)}</strong> / ${fmt(t.threshold, 1)}</td>
        <td class="num"><strong>${f.qualifying_checkins ?? "&ndash;"}</strong> of ${f.considered_checkins ?? f.supporting_checkins} considered<br><span class="tiny muted">the rule looks at the last ${f.summary.config ? f.summary.config.window_sessions : 3} usable check-ins</span></td>
        <td class="small">${esc(t.reason || f.summary.explanation)}</td>
        <td class="small">${sup.length ? sup.map(esc).join(" ") : '<span class="muted">none recorded</span>'}</td>
        <td>${f.review_state === "pending" ? '<span class="pill amber">Pending</span>' : '<span class="pill teal">Reviewed</span>'}<br>${flagPill(f.status)}</td><td>${modeTag(f.data_mode)}</td>
        <td><a class="btn" href="#/review/${encodeURIComponent(f.flag_id)}">Open</a></td></tr>`; }).join("") || `<tr><td colspan="9" class="muted small">${empty}</td></tr>`}
    </tbody></table></div>`;
}

async function viewReviewDetail(fid) {
  const f = await api(`/v1/review-flags/${encodeURIComponent(fid)}`);
  const tl = await timeline(f.patient_id, 365);
  const sm = f.summary, ACTIONS = { OPEN: ["acknowledge", "request_repeat", "dismiss", "close"], ACKNOWLEDGED: ["review", "request_repeat", "dismiss", "close"], REVIEWED: ["request_repeat", "close"] };
  const LABEL = { acknowledge: "Acknowledge", review: "Mark reviewed", close: "Close review", dismiss: "Dismiss", request_repeat: "Request repeat" };
  const acts = ACTIONS[f.status] || [];
  $app.innerHTML = `<a href="#/review">&larr; Review queue</a>
    <h1>Speech Change Review ${flagPill(f.status)}</h1>
    <p class="row small">${esc(f.patient_display_name)} ${modeTag(f.data_mode)} <span class="muted">Detected ${fmtDT(f.created_at)} &middot; assigned to ${esc(f.assigned_reviewer_id || "nobody")}</span></p>
    <div class="grid two">
      <section class="card stack"><h2>Summary</h2>
        <p><strong>${esc(sm.explanation)}</strong></p>
        <dl class="meta small"><dt>Rule</dt><dd>${esc(f.rule_id)} v${esc(f.rule_version)} (experimental heuristic)</dd>
          <dt>Pattern</dt><dd>${fmtDT(sm.first_elevated)} to ${fmtDT(sm.last_elevated)}</dd>
          <dt>Qualifying recordings</dt><dd>${sm.qualifying} of ${sm.window}, on ${sm.distinct_days} days</dd></dl>
        <p class="small muted">A crossing of the boundary describes measured speech, not a health condition. This review does not prescribe any follow-up.</p></section>
      <section class="card stack"><h2>Review</h2>
        ${acts.length ? `<label for="rnote">Note (required to close or dismiss)</label><textarea id="rnote" rows="3" maxlength="500"></textarea>
          <p class="tiny muted" style="margin:0">Request repeat asks the person for another check-in in the app. It is not a treatment instruction, and nothing is sent outside the app.</p>
          <div class="row">${acts.map((a) => `<button data-act="${a}" class="${a === acts[0] ? "primary" : ""}">${LABEL[a]}</button>`).join("")}<button data-act="note">Add note only</button></div>`
          : `<p class="muted small">This review is ${esc(f.status.toLowerCase())}. Dismissing a review does not mean the speech change was insignificant.</p>`}
        <p id="ractmsg" role="status" class="small"></p></section>
    </div>
    <section class="card" style="margin-top:20px"><h2>Evidence</h2>
      ${tl ? sessionChart(tl.checkins.slice(-14)) : ""}
      <div class="tablewrap"><table><caption class="sr-only">Check-ins behind this review</caption><thead><tr><th>Check-in</th><th>Role</th><th>Quality</th><th class="num">Difference (&times; boundary)</th><th class="num">Pause time vs baseline</th><th>Speech timing</th><th>Reported context</th><th></th></tr></thead><tbody>
        ${f.evidence.map((e) => { const pf = e.acoustic_features && e.acoustic_features.pause_fraction; return `<tr><td>${fmtDT(e.recorded_at)}</td><td>${e.role === "qualifying" ? "Above boundary" : "Within boundary (context)"}</td><td>${qualityPill(e.quality)}</td>
          <td class="num">${fmt(e.normalized_score)}</td><td class="num">${pf && pf.status === "VALID" ? `${FEATURE_FMT.pause_fraction(pf.current)}<br><span class="tiny muted">${DELTA_FMT.pause_fraction(pf.delta)}</span>` : '<span class="muted small">unavailable</span>'}</td>
          <td class="small">${(e.timing_chips || []).map((x) => esc(sentence(x.text))).join(" ") || "&ndash;"}</td><td class="small">${esc(contextText(e.context)) || "&ndash;"}</td>
          <td>${e.has_window_analysis ? `<a href="#/analysis/${encodeURIComponent(e.checkin_id)}">Window analysis</a>` : ""}</td></tr>`; }).join("")}
      </tbody></table></div>
      ${(f.verifications || []).length ? `<p class="small"><strong>Repeat check-ins</strong> (shown for context, never counted as extra evidence): ${f.verifications.map((v) => `${fmtDT(v.recorded_at)}: ${esc((VRESULT[v.result] || [v.result || "pending"])[0])}`).join("; ")}</p>` : ""}
      <p class="small"><a href="#/research/${encodeURIComponent(f.patient_id)}" id="totrends">Open this person's check-ins &rarr;</a></p></section>
    <div class="grid half" style="margin-top:20px">
      <section class="card"><h2>Model context</h2><dl class="meta small">
        <dt>Encoder</dt><dd>${esc(sm.encoder)}</dd><dt>Projection</dt><dd>${esc(sm.projection)}</dd>
        <dt>Baseline</dt><dd>${esc(sm.baseline_id || "&ndash;")}</dd><dt>Boundary</dt><dd>${esc(sm.calibration_id || "&ndash;")} (${esc(sm.calibration)})</dd>
        <dt>Settings</dt><dd>${sm.config.min_elevated} of ${sm.config.window_sessions}, &ge;${sm.config.min_distinct_days} days, ${sm.config.lookback_days}-day window, ${sm.config.cooldown_hours} h cooldown</dd></dl>
        <ul class="small">${f.limitations.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></section>
      <section class="card"><h2>History</h2><ol class="small">${f.audit.map((a) => `<li>${fmtDT(a.timestamp)}: <strong>${esc(a.action)}</strong> by ${esc(a.actor_id)}${a.reason_code ? ` (${esc(a.reason_code)})` : ""}${a.detail && a.detail.note ? `: &ldquo;${esc(a.detail.note)}&rdquo;` : ""}</li>`).join("")}</ol>
        <p class="tiny muted">Notifications: ${f.notifications.map((n) => `${esc(n.channel)}, ${esc(n.delivery_status)}, external delivery ${n.external_delivery ? "yes" : "no"}`).join("; ")}</p></section>
    </div>`;
  document.getElementById("totrends").onclick = () => setPid(f.patient_id);
  $app.querySelectorAll("[data-act]").forEach((b) => (b.onclick = async () => {
    const msg = document.getElementById("ractmsg");
    if (["close", "dismiss"].includes(b.dataset.act) && !document.getElementById("rnote").value.trim()) {
      msg.textContent = "Add a note with the reason before closing or dismissing."; return;
    }
    try {
      await post(`/v1/review-flags/${encodeURIComponent(fid)}/actions`, { action: b.dataset.act, actor: "reviewer_demo", note: document.getElementById("rnote").value });
      await viewReviewDetail(fid); renderTopbar();
      if (b.dataset.act === "request_repeat") document.getElementById("ractmsg").textContent = "Repeat requested. The person will see it on their Home page.";
    } catch (e) { msg.textContent = "That action isn't allowed now (" + e.message + ")."; }
  }));
}

// ------------------------------------------------------------------ Research workspace (former clinician view)
async function viewResearch() {
  const list = await api("/patients");
  $app.innerHTML = `<div class="eyebrow"><span class="line"></span>Clinician view</div><h1>Patients</h1><p class="lead">All profiles on this computer. Neutral comparison labels only; no risk scores. Pending reviews are in the <a href="#/review">review queue</a>.</p>
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
    $app.innerHTML = `<a href="#/research">&larr; Patients</a>
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
    <p class="tiny muted">Task: ${esc((c.protocol || {}).prompt_id || "default")} (${esc((c.protocol || {}).task_type || "free_speech")})${c.checkin_type === "verification" ? ` &middot; repeat of an earlier check-in${c.verification ? `: ${esc((VRESULT[c.verification.result] || [c.verification.result])[0])}` : ""}` : ""}</p>
    ${c.comparison ? `<p class="small callout caution">${esc(c.comparison.message)}</p>` : ""}
    ${!safety && c.acoustic_features ? acousticHtml(c.acoustic_features, c.acoustic_summary, { heading: "Speech measurements" }) : ""}
    ${contextText(c.context) ? `<p class="small"><strong>Reported context:</strong> ${esc(contextText(c.context))}</p>` : ""}
    ${c.has_window_analysis ? `<p><a class="btn" href="#/analysis/${encodeURIComponent(c.checkin_id)}">${icon("analysis")}Window-by-window analysis</a></p>` : ""}
    ${!safety && isClin() ? `<h3>Evidence quotes</h3>${(c.evidence || []).length ? c.evidence.map((e) => `<p class="small"><span class="pill gray">${esc(e.item)}</span> &ldquo;${esc(e.quote)}&rdquo;</p>`).join("") : '<p class="muted small">None found (or evidence unavailable).</p>'}
      <h3>Transcript</h3><div class="transcript small">${c.transcript ? highlight(c.transcript, c.evidence) : '<span class="muted">Transcript unavailable.</span>'}</div>` : ""}
  </section>`;
}
