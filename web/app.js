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

  const finish = () => { cancelAnimationFrame(raf); clearInterval(timer); stream?.getTracks().forEach((t) => t.stop()); ctx?.close(); };
  cleanup = () => { stopped = true; finish(); if (mr && mr.state !== "inactive") mr.stop(); };

  rec.onclick = async () => {
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } }); }
    catch (e) { hint.textContent = "Microphone unavailable: " + e.message; return; }
    rec.disabled = true;
    cancel.disabled = false;
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
    ${["stable", "drift", "all", "blank"].map((s) => `<button data-reset="${s}">Reset: ${s}</button>`).join("")}</div>`;
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
  let sel = cks.length ? cks[cks.length - 1].checkin_id : null;
  const render = () => {
    const c = cks.find((x) => x.checkin_id === sel);
    $app.innerHTML = `<a href="#/clinician">&larr; Patients</a>
      <h1>${esc(tl.patient.display_name)} ${tl.patient.simulated ? '<span class="sim">simulated</span>' : ""}</h1>
      ${alertsHtml(tl)}
      <div class="card"><h2>Drift vs. personal baseline</h2>${chart(tl, sel)}
        <div class="small muted">Dashed line = threshold &tau;. First ${tl.baseline ? tl.baseline.n : 5} passing check-ins form the baseline (grey). &#9679; scored &nbsp; &#9675; low-weight (WARN) &nbsp; &times; rejected. Acted days are labeled simulated.</div></div>
      ${c ? checkinHtml(c) : '<div class="card muted">No check-ins yet.</div>'}
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
    ${c.summary ? `<p><em>${esc(c.summary)}</em></p>` : ""}
    ${hasScores ? `<h3>Evidence${c.evidence.length ? "" : " (none)"}</h3>${c.evidence.map((e) => `<blockquote><span class="chip">${esc(e.item)} &middot; ${esc(e.direction)}</span><br>&ldquo;${esc(e.quote)}&rdquo;</blockquote>`).join("")}
    <h3>Transcript</h3><div class="transcript">${c.transcript ? highlight(c.transcript, c.evidence) : '<span class="muted">Transcript unavailable.</span>'}</div>` : ""}
  </div>`;
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
