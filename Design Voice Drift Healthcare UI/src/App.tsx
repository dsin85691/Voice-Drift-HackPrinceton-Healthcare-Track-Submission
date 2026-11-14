import { useEffect, useState } from "react";

type IconName =
  | "home"
  | "mic"
  | "trend"
  | "users"
  | "flask"
  | "settings"
  | "bell"
  | "calendar"
  | "chevron"
  | "arrow"
  | "check"
  | "shield"
  | "play"
  | "close"
  | "menu";

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, React.ReactNode> = {
    home: (
      <>
        <path d="m3 10 9-7 9 7" />
        <path d="M5 9v11h14V9" />
        <path d="M9 20v-6h6v6" />
      </>
    ),
    mic: (
      <>
        <rect x="8" y="3" width="8" height="12" rx="4" />
        <path d="M5 11a7 7 0 0 0 14 0M12 18v3M9 21h6" />
      </>
    ),
    trend: (
      <>
        <path d="M3 18 9 12l4 4 8-10" />
        <path d="M15 6h6v6" />
      </>
    ),
    users: (
      <>
        <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M17 11a4 4 0 0 0 0-8M22 21v-2a4 4 0 0 0-3-3.87" />
      </>
    ),
    flask: (
      <>
        <path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3" />
        <path d="M8 15h8" />
      </>
    ),
    settings: (
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.7 1.7 0 0 0 .34 1.88l.06.06-2.83 2.83-.06-.06a1.7 1.7 0 0 0-1.88-.34 1.7 1.7 0 0 0-1.03 1.56V21h-4v-.08A1.7 1.7 0 0 0 9 19.37a1.7 1.7 0 0 0-1.88.34l-.06.06-2.83-2.83.06-.06A1.7 1.7 0 0 0 4.63 15 1.7 1.7 0 0 0 3.08 14H3v-4h.08A1.7 1.7 0 0 0 4.63 9a1.7 1.7 0 0 0-.34-1.88l-.06-.06 2.83-2.83.06.06A1.7 1.7 0 0 0 9 4.63 1.7 1.7 0 0 0 10 3.08V3h4v.08A1.7 1.7 0 0 0 15 4.63a1.7 1.7 0 0 0 1.88-.34l.06-.06 2.83 2.83-.06.06A1.7 1.7 0 0 0 19.37 9 1.7 1.7 0 0 0 20.92 10H21v4h-.08A1.7 1.7 0 0 0 19.4 15Z" />
      </>
    ),
    bell: (
      <>
        <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4" />
      </>
    ),
    calendar: (
      <>
        <rect x="3" y="5" width="18" height="16" rx="2" />
        <path d="M16 3v4M8 3v4M3 10h18" />
      </>
    ),
    chevron: <path d="m9 18 6-6-6-6" />,
    arrow: (
      <>
        <path d="M5 12h14M13 6l6 6-6 6" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    shield: (
      <>
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />
        <path d="m9 12 2 2 4-4" />
      </>
    ),
    play: <path d="m8 5 11 7-11 7Z" />,
    close: (
      <>
        <path d="m18 6-12 12M6 6l12 12" />
      </>
    ),
    menu: (
      <>
        <path d="M4 7h16M4 12h16M4 17h16" />
      </>
    ),
  };
  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name]}
    </svg>
  );
}

function Brand() {
  return (
    <div className="brand">
      <svg className="brand-mark" viewBox="0 0 38 38" aria-hidden="true">
        <rect width="38" height="38" rx="12" fill="#087F73" />
        <path
          d="M10 20v-3m4 8V12m5 16V9m5 16V13m5 7v-4"
          fill="none"
          stroke="white"
          strokeWidth="2.4"
          strokeLinecap="round"
        />
        <path
          d="M10 24c4.2-1.4 6.6 2.5 10.5.5 3.2-1.7 4.6-5.2 8.5-4.5"
          fill="none"
          stroke="#A7E0D8"
          strokeWidth="1.7"
          strokeLinecap="round"
        />
      </svg>
      <div>
        <strong>Voice Drift</strong>
        <span>Clinical monitoring</span>
      </div>
    </div>
  );
}

const navItems: { label: string; icon: IconName }[] = [
  { label: "Overview", icon: "home" },
  { label: "Check-ins", icon: "mic" },
  { label: "Voice timeline", icon: "trend" },
  { label: "Review queue", icon: "users" },
  { label: "Validation lab", icon: "flask" },
];

const waveform = [
  12, 20, 8, 27, 17, 33, 19, 40, 24, 13, 29, 36, 16, 22, 11, 31, 38, 21, 14, 27, 20, 35, 16, 24, 11,
  30, 18, 25, 14, 20,
];

function MiniWave({ alert = false }: { alert?: boolean }) {
  return (
    <div className={`mini-wave ${alert ? "alert" : ""}`} aria-hidden="true">
      {waveform.slice(0, 18).map((h, i) => (
        <i key={i} style={{ height: `${h * 0.55}px` }} />
      ))}
    </div>
  );
}

function App() {
  const [active, setActive] = useState("Overview");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [demoLoaded, setDemoLoaded] = useState(false);
  const [notice, setNotice] = useState("");

  useEffect(() => {
    if (!recording) return;
    const timer = window.setInterval(() => setSeconds((s) => Math.min(60, s + 1)), 1000);
    return () => window.clearInterval(timer);
  }, [recording]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(""), 2600);
    return () => window.clearTimeout(timer);
  }, [notice]);

  function openRecording() {
    setSeconds(0);
    setRecording(true);
  }

  function finishRecording() {
    setRecording(false);
    setNotice("Check-in saved and queued for analysis");
  }

  function chooseNav(label: string) {
    setActive(label);
    setSidebarOpen(false);
    if (label !== "Overview") setNotice(`${label} workspace selected`);
  }

  return (
    <div className="app-shell">
      {sidebarOpen && (
        <button className="scrim" aria-label="Close menu" onClick={() => setSidebarOpen(false)} />
      )}
      <aside className={sidebarOpen ? "sidebar open" : "sidebar"}>
        <Brand />
        <nav aria-label="Primary navigation">
          <p className="nav-label">Workspace</p>
          {navItems.map((item) => (
            <button
              key={item.label}
              className={active === item.label ? "nav-item active" : "nav-item"}
              onClick={() => chooseNav(item.label)}
            >
              <Icon name={item.icon} />
              <span>{item.label}</span>
              {item.label === "Review queue" && <b>3</b>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="privacy-card">
            <span>
              <Icon name="shield" size={17} />
            </span>
            <div>
              <strong>Data protected</strong>
              <p>Audio encrypted at rest</p>
            </div>
          </div>
          <button className="nav-item">
            <Icon name="settings" />
            <span>Settings</span>
          </button>
          <div className="user-row">
            <div className="avatar">AM</div>
            <div>
              <strong>Dr. Amara Miles</strong>
              <span>Clinical researcher</span>
            </div>
            <Icon name="chevron" size={15} />
          </div>
        </div>
      </aside>

      <main>
        <header className="topbar">
          <button
            className="mobile-menu"
            aria-label="Open menu"
            onClick={() => setSidebarOpen(true)}
          >
            <Icon name="menu" />
          </button>
          <div>
            <p>Tuesday, 20 August</p>
            <h1>{active}</h1>
          </div>
          <div className="top-actions">
            <button className="icon-button" aria-label="Notifications">
              <Icon name="bell" />
              <i />
            </button>
            <button className="primary-button compact" onClick={openRecording}>
              <Icon name="mic" size={16} />
              New check-in
            </button>
          </div>
        </header>

        <div className="content">
          <section className="welcome">
            <div className="welcome-copy">
              <p className="eyebrow">LONGITUDINAL VOICE HEALTH</p>
              <h2>Good morning, Amara.</h2>
              <p>Here’s what’s changed across your monitored profiles this week.</p>
              <div className="welcome-note">
                <span className="status-dot" />
                All monitoring systems are healthy
              </div>
            </div>
            <div className="welcome-aside">
              <div className="care-team">
                <div className="team-avatar amber">MO</div>
                <div className="team-avatar rose">EL</div>
                <div className="team-avatar blue">NK</div>
                <div className="team-avatar more">+8</div>
              </div>
              <span>Care team online</span>
              <div className="date-control">
                <Icon name="calendar" size={16} />
                Last 30 days
                <Icon name="chevron" size={14} />
              </div>
            </div>
          </section>

          <section className="metrics-grid" aria-label="Summary metrics">
            <article className="metric-card metric-featured">
              <div className="metric-icon teal">
                <Icon name="users" />
              </div>
              <div>
                <p>Active profiles</p>
                <strong>24</strong>
                <span className="up">+3 this month</span>
              </div>
            </article>
            <article className="metric-card metric-sky">
              <div className="metric-icon blue">
                <Icon name="mic" />
              </div>
              <div>
                <p>Check-ins this week</p>
                <strong>67</strong>
                <span>82% completion</span>
              </div>
            </article>
            <article className="metric-card metric-rose">
              <div className="metric-icon coral">
                <Icon name="trend" />
              </div>
              <div>
                <p>Flagged for review</p>
                <strong>3</strong>
                <span className="alert-text">2 newly detected</span>
              </div>
            </article>
            <article className="metric-card metric-lilac">
              <div className="metric-icon violet">
                <Icon name="shield" />
              </div>
              <div>
                <p>Noise resilience</p>
                <strong>91%</strong>
                <span className="up">+6% vs baseline</span>
              </div>
            </article>
          </section>

          <section className="main-grid">
            <article className="panel drift-panel">
              <div className="panel-head">
                <div>
                  <h3>Voice drift overview</h3>
                  <p>Average personalized drift across all active profiles</p>
                </div>
                <button className="text-button" onClick={() => chooseNav("Voice timeline")}>
                  View timeline <Icon name="arrow" size={15} />
                </button>
              </div>
              <div className="chart-wrap">
                <div className="y-axis">
                  <span>High</span>
                  <span>0.6</span>
                  <span>0.4</span>
                  <span>0.2</span>
                  <span>Stable</span>
                </div>
                <div className="chart">
                  <div className="threshold">
                    <span>Review threshold</span>
                  </div>
                  <svg
                    viewBox="0 0 600 190"
                    preserveAspectRatio="none"
                    role="img"
                    aria-label="Voice drift trend from May to August"
                  >
                    <defs>
                      <linearGradient id="area" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0" stopColor="#138F82" stopOpacity=".22" />
                        <stop offset="1" stopColor="#138F82" stopOpacity="0" />
                      </linearGradient>
                    </defs>
                    <path
                      d="M0 137 C42 130 56 143 98 128 S151 101 194 116 S247 141 288 119 S343 65 386 85 S440 119 480 95 S533 55 600 70 L600 190 L0 190Z"
                      fill="url(#area)"
                    />
                    <path
                      d="M0 137 C42 130 56 143 98 128 S151 101 194 116 S247 141 288 119 S343 65 386 85 S440 119 480 95 S533 55 600 70"
                      fill="none"
                      stroke="#087F73"
                      strokeWidth="3"
                      strokeLinecap="round"
                    />
                    <circle cx="600" cy="70" r="5" fill="#fff" stroke="#087F73" strokeWidth="3" />
                  </svg>
                  <div className="x-axis">
                    <span>May 20</span>
                    <span>Jun 10</span>
                    <span>Jul 01</span>
                    <span>Jul 22</span>
                    <span>Aug 19</span>
                  </div>
                </div>
              </div>
              <div className="chart-summary">
                <span className="summary-icon">
                  <Icon name="trend" size={17} />
                </span>
                <p>
                  <strong>Drift remains within the expected range.</strong> Two profiles crossed the
                  review threshold this week.
                </p>
              </div>
            </article>

            <article className="panel queue-panel">
              <div className="panel-head">
                <div>
                  <h3>Review queue</h3>
                  <p>Prioritized by drift score</p>
                </div>
                <span className="count-pill">3 pending</span>
              </div>
              <div className="patient-list">
                {[
                  ["JL", "Jordan Lee", "Today, 9:42 AM", "0.78", true, "peach"],
                  ["RT", "Riley Thompson", "Yesterday, 4:18 PM", "0.65", true, "blue"],
                  ["SK", "Samir Khan", "Mon, 11:06 AM", "0.54", false, "gold"],
                ].map(([initials, name, time, score, urgent, tone]) => (
                  <button
                    className="patient-row"
                    key={name as string}
                    onClick={() => setNotice(`${name} review opened`)}
                  >
                    <div className={`patient-avatar ${tone} ${urgent ? "urgent" : ""}`}>
                      {initials}
                    </div>
                    <div className="patient-info">
                      <strong>{name}</strong>
                      <span>{time}</span>
                      <MiniWave alert={urgent as boolean} />
                    </div>
                    <div className="score">
                      <strong>{score}</strong>
                      <span>drift</span>
                    </div>
                    <Icon name="chevron" size={16} />
                  </button>
                ))}
              </div>
              <button className="secondary-button full" onClick={() => chooseNav("Review queue")}>
                Open review queue <Icon name="arrow" size={15} />
              </button>
            </article>
          </section>

          <section className="bottom-grid">
            <article className="demo-card">
              <div className="demo-visual">
                <div className="orb">
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                </div>
              </div>
              <div className="demo-copy">
                <span className="tag">INTERACTIVE DEMO</span>
                <h3>{demoLoaded ? "Sample profile ready" : "See Voice Drift in action"}</h3>
                <p>
                  {demoLoaded
                    ? "A 12-week longitudinal profile has been added to your workspace."
                    : "Load a seeded profile and follow one recording from capture through noise-robust analysis and clinical review."}
                </p>
                <button
                  className="dark-button"
                  onClick={() => {
                    setDemoLoaded(true);
                    setNotice("Demo profile loaded successfully");
                  }}
                >
                  {demoLoaded ? (
                    <>
                      <Icon name="check" size={16} />
                      Profile loaded
                    </>
                  ) : (
                    <>
                      <Icon name="play" size={15} />
                      Launch sample profile
                    </>
                  )}
                </button>
              </div>
            </article>
            <article className="panel checkin-card">
              <div className="checkin-icon">
                <Icon name="mic" size={23} />
              </div>
              <div>
                <p>YOUR NEXT CHECK-IN</p>
                <h3>Friday, 23 August</h3>
                <span>60-second standardized reading</span>
              </div>
              <button onClick={openRecording}>
                Start early <Icon name="arrow" size={15} />
              </button>
            </article>
          </section>
        </div>
      </main>

      {recording && (
        <div className="modal-backdrop" role="presentation">
          <section
            className="record-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="record-title"
          >
            <button
              className="modal-close"
              aria-label="Close recording"
              onClick={() => setRecording(false)}
            >
              <Icon name="close" />
            </button>
            <p className="eyebrow">VOICE CHECK-IN</p>
            <h2 id="record-title">{seconds ? "Recording in progress" : "Ready when you are"}</h2>
            <p className="passage">
              “The morning light moved slowly across the room, and outside, the city began another
              ordinary day.”
            </p>
            <div className={`record-wave ${seconds ? "live" : ""}`}>
              {waveform.map((h, i) => (
                <i key={i} style={{ height: `${h}px`, animationDelay: `${i * -45}ms` }} />
              ))}
            </div>
            <div className="record-time">
              <strong>0:{String(seconds).padStart(2, "0")}</strong>
              <span>/ 1:00</span>
            </div>
            <div className="record-progress">
              <i style={{ width: `${Math.max(2, (seconds / 60) * 100)}%` }} />
            </div>
            <button
              className="primary-button record-action"
              onClick={seconds ? finishRecording : () => setSeconds(1)}
            >
              {seconds ? (
                "Finish recording"
              ) : (
                <>
                  <Icon name="mic" />
                  Begin recording
                </>
              )}
            </button>
            <span className="secure-note">
              <Icon name="shield" size={14} />
              Encrypted and processed securely
            </span>
          </section>
        </div>
      )}
      {notice && (
        <div className="toast">
          <span>
            <Icon name="check" size={15} />
          </span>
          {notice}
        </div>
      )}
    </div>
  );
}

export default App;
