import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, ReactNode, CSSProperties } from 'react'

type Page = 'overview' | 'checkin' | 'insights' | 'review' | 'results'
type Metric = 'pace' | 'pauses' | 'consistency'
type CheckIn = {
  id: string
  date: string
  day: string
  time: string
  pace: number
  pauses: number
  consistency: number
  flagged: boolean
}

const normalReadings: CheckIn[] = [
  { id: 'VD-101', date: 'Nov 06', day: 'Fri', time: '9:18 AM', pace: 102, pauses: 98, consistency: 95, flagged: false },
  { id: 'VD-102', date: 'Nov 08', day: 'Sun', time: '9:36 AM', pace: 99, pauses: 103, consistency: 94, flagged: false },
  { id: 'VD-103', date: 'Nov 10', day: 'Tue', time: '8:57 AM', pace: 103, pauses: 101, consistency: 96, flagged: false },
  { id: 'VD-104', date: 'Nov 12', day: 'Thu', time: '9:12 AM', pace: 97, pauses: 106, consistency: 92, flagged: false },
  { id: 'VD-105', date: 'Nov 14', day: 'Sat', time: '9:42 AM', pace: 100, pauses: 102, consistency: 95, flagged: false },
]
const demoReadings: CheckIn[] = [
  normalReadings[0],
  normalReadings[1],
  { id: 'VD-103', date: 'Nov 10', day: 'Tue', time: '8:57 AM', pace: 88, pauses: 132, consistency: 77, flagged: true },
  { id: 'VD-104', date: 'Nov 12', day: 'Thu', time: '9:12 AM', pace: 76, pauses: 149, consistency: 65, flagged: true },
  { id: 'VD-105', date: 'Nov 14', day: 'Sat', time: '9:42 AM', pace: 73, pauses: 156, consistency: 60, flagged: true },
]
const pageNames: Record<Page, string> = {
  overview: 'Overview', checkin: 'Voice check-in', insights: 'Drift insights', review: 'Review queue', results: 'Results & methods',
}
const metricInfo: Record<Metric, { label: string; short: string; unit: string; description: string; color: string; min: number; max: number; range: [number, number] }> = {
  pace: { label: 'Speech pace', short: 'Relative pace', unit: '% of usual pace', description: 'How quickly words are spoken compared with the personal baseline.', color: '#4a85d8', min: 60, max: 115, range: [90, 110] },
  pauses: { label: 'Pause duration', short: 'Relative pauses', unit: '% of usual duration', description: 'The duration of silent intervals compared with a person’s usual pattern.', color: '#db8965', min: 75, max: 175, range: [85, 120] },
  consistency: { label: 'Signal consistency', short: 'Consistency', unit: 'illustrative score', description: 'A simplified demo-only indicator of how similar repeated check-ins appear.', color: '#779b7a', min: 45, max: 105, range: [85, 100] },
}

function Icon({ name, size = 19, strokeWidth = 1.8 }: { name: string; size?: number; strokeWidth?: number }) {
  const common = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true as const }
  const shapes: Record<string, ReactNode> = {
    dashboard: <><rect x="3" y="3" width="7" height="7" rx="1.7"/><rect x="14" y="3" width="7" height="7" rx="1.7"/><rect x="3" y="14" width="7" height="7" rx="1.7"/><rect x="14" y="14" width="7" height="7" rx="1.7"/></>,
    mic: <><rect x="9" y="2" width="6" height="13" rx="3"/><path d="M5 10a7 7 0 0 0 14 0M12 17v5m-4 0h8"/></>,
    chart: <><path d="M3 3v18h18"/><path d="m6 16 4-5 4 3 6-8"/></>,
    queue: <><rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4V2h6v2M9 10h6M9 15h6M9 18h3"/></>,
    layers: <><path d="m12 2 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 17l9 5 9-5"/></>,
    arrow: <><path d="M4 12h15m-6-6 6 6-6 6"/></>,
    chevron: <><path d="m9 18 6-6-6-6"/></>,
    bell: <><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4"/></>,
    sound: <><path d="M2 10v4m4-8v12m4-15v18m4-13v8m4-12v16m4-10v4"/></>,
    shield: <><path d="m12 22-1.3-.5C5 19 3.5 15.6 3.5 11V4.6L12 2l8.5 2.6V11c0 4.6-1.5 8-7.2 10.5L12 22Z"/><path d="m9 12 2 2 4-4"/></>,
    calendar: <><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 3v4M17 3v4M3 10h18"/></>,
    download: <><path d="M12 3v12m-4-4 4 4 4-4M4 17v4h16v-4"/></>,
    info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></>,
    check: <path d="m5 12 5 5L20 7"/>,
    checkcircle: <><circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/></>,
    alert: <><path d="m12 3 10 18H2L12 3Z"/><path d="M12 9v5m0 3h.01"/></>,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    refresh: <><path d="M20 7V3m0 4h-4M4 17v4m0-4h4"/><path d="M20 7a8 8 0 0 0-14-2M4 17a8 8 0 0 0 14 2"/></>,
    play: <path d="m8 5 11 7-11 7V5Z"/>,
    pause: <><path d="M8 5v14M16 5v14"/></>,
    stop: <rect x="6" y="6" width="12" height="12" rx="2"/>,
    upload: <><path d="M12 16V3m-5 5 5-5 5 5M4 17v4h16v-4"/></>,
    lock: <><rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></>,
    menu: <path d="M4 6h16M4 12h16M4 18h16"/>,
    close: <path d="M5 5 19 19M19 5 5 19"/>,
    spark: <><path d="m12 2 1.8 7.2L21 11l-7.2 1.8L12 20l-1.8-7.2L3 11l7.2-1.8L12 2Z"/></>,
    file: <><path d="M6 2h8l5 5v14H6V2Zm8 0v6h5M9 13h7M9 17h7"/></>,
  }
  return <svg {...common}>{shapes[name] ?? shapes.info}</svg>
}

function Mark({ light = false }: { light?: boolean }) {
  return <span className={`brand-mark ${light ? 'brand-mark-light' : ''}`} aria-hidden="true"><i/><i/><i/><i/><i/></span>
}
function Waveform({ compact = false }: { compact?: boolean }) {
  const levels = [12,20,14,38,22,52,27,45,15,28,58,40,68,37,23,51,30,44,18,26,47,24,58,37,20,30,13,24,42,17,31,12,22,11]
  return <div className={`waveform ${compact ? 'waveform-compact' : ''}`} aria-hidden="true">{levels.map((x,i) => <i key={i} style={{ height: `${x}%` }}/>)}</div>
}
function Status({ flagged, label }: { flagged: boolean; label?: string }) {
  return <span className={`status ${flagged ? 'status-flag' : 'status-good'}`}><span className="status-dot"/>{label ?? (flagged ? 'Outside range' : 'Within range')}</span>
}
function PageHeading({ eyebrow, title, description, actions }: { eyebrow: string; title: string; description: string; actions?: ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow"><span className="eyebrow-line"/>{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{actions && <div className="page-heading-actions">{actions}</div>}</div>
}
function MiniSpark({ values, color, inverted = false }: { values: number[]; color: string; inverted?: boolean }) {
  const lo = Math.min(...values) - 3, hi = Math.max(...values) + 3
  const points = values.map((v, i) => `${5 + i*33},${49 - (v-lo)/(hi-lo)*35}`).join(' ')
  return <svg className="mini-spark" viewBox="0 0 145 57" aria-hidden="true"><polyline points={points} fill="none" stroke={color} strokeWidth="2.9" strokeLinecap="round" strokeLinejoin="round"/><circle cx="137" cy={49-(values[4]-lo)/(hi-lo)*35} r="4.5" fill={color} stroke={inverted ? '#1e403a' : '#fff'} strokeWidth="2.5"/></svg>
}
function ChartView({ data, metric }: { data: CheckIn[]; metric: Metric }) {
  const info = metricInfo[metric]
  const w = 790, h = 254, left = 52, right = 25, top = 25, bottom = 40
  const chartH = h - top - bottom, chartW = w - left - right
  const y = (n:number) => top + (info.max - n)/(info.max - info.min)*chartH
  const x = (i:number) => left + i*chartW/(data.length - 1)
  const vals = data.map(d=>d[metric])
  const points = vals.map((v,i)=>`${x(i)},${y(v)}`).join(' ')
  const gradientPoints = `${left},${h-bottom} ${points} ${w-right},${h-bottom}`
  const levels = Array.from({length:5}, (_,i) => info.min + i*(info.max-info.min)/4)
  return <div className="chart-container"><svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`${info.label} across five example check-ins. Latest value ${vals[4]} percent.`}>
    <defs><linearGradient id={`fill-${metric}`} x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor={info.color} stopOpacity="0.19"/><stop offset="100%" stopColor={info.color} stopOpacity="0"/></linearGradient></defs>
    <rect x={left} y={y(info.range[1])} width={chartW} height={Math.max(0,y(info.range[0])-y(info.range[1]))} fill="#8eb49b" opacity=".10" rx="6"/>
    {levels.map((n,i)=><g key={i}><line x1={left} x2={w-right} y1={y(n)} y2={y(n)} stroke="#dfe9e5" strokeDasharray="4 5"/><text x={left-12} y={y(n)+4} textAnchor="end" className="chart-tick">{Math.round(n)}</text></g>)}
    <polygon points={gradientPoints} fill={`url(#fill-${metric})`}/>
    <polyline points={points} fill="none" stroke={info.color} strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round"/>
    {vals.map((v,i)=><g key={i}><circle cx={x(i)} cy={y(v)} r="6.4" fill="#fff" stroke={info.color} strokeWidth="3"/><text x={x(i)} y={h-12} textAnchor="middle" className="chart-tick">{data[i].date}</text></g>)}
  </svg><div className="chart-legend"><span><i style={{ background: info.color }}/>{info.label}</span><span><i className="legend-range"/>Illustrative usual range</span></div></div>
}
function DownloadButton({ readings }: { readings: CheckIn[] }) {
  const download = () => {
    const rows = ['id,date,time,pace_pct,pauses_pct,consistency_score,flagged',...readings.map(r=>[r.id,r.date,r.time,r.pace,r.pauses,r.consistency,r.flagged].join(','))].join('\n')
    const url = URL.createObjectURL(new Blob([rows],{type:'text/csv;charset=utf-8'}))
    const a = document.createElement('a'); a.href = url; a.download='voice-drift-illustrative-checkins.csv'; a.click()
    window.setTimeout(()=>URL.revokeObjectURL(url),1000)
  }
  return <button className="btn btn-outline btn-sm" onClick={download}><Icon name="download" size={16}/>Export demo CSV</button>
}

function App() {
  const [page, setPage] = useState<Page>('overview')
  const [metric, setMetric] = useState<Metric>('pace')
  const [demoActive, setDemoActive] = useState(false)
  const [reviewFilter, setReviewFilter] = useState<'flagged'|'all'>('flagged')
  const [selectedId, setSelectedId] = useState('VD-105')
  const [mobileNav, setMobileNav] = useState(false)
  const [notice, setNotice] = useState('')
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [audioUrl, setAudioUrl] = useState<string|null>(null)
  const [audioName, setAudioName] = useState('')
  const recorder = useRef<MediaRecorder|null>(null)
  const chunks = useRef<Blob[]>([])
  const ownedAudioUrl = useRef<string|null>(null)
  const fileInput = useRef<HTMLInputElement|null>(null)
  const readings = demoActive ? demoReadings : normalReadings
  const latest = readings[readings.length - 1]
  const flagged = readings.filter(d=>d.flagged)
  const selected = readings.find(d=>d.id===selectedId) ?? latest
  const queue = reviewFilter==='flagged' ? flagged.slice().reverse() : readings.slice().reverse()
  const summary = demoActive ? '3 of 5 example check-ins are outside the illustrative boundary' : 'All 5 example check-ins are within the illustrative boundary'
  const go = (next: Page) => {setPage(next);setMobileNav(false);window.scrollTo({top:0,behavior:'smooth'})}
  const launchDemo = () => {setDemoActive(true);setSelectedId('VD-105');setNotice('Illustrative drift scenario loaded. Review queue and charts are updated.');go('overview')}
  const resetDemo = () => {setDemoActive(false);setNotice('Baseline example restored.');go('overview')}

  useEffect(()=>{if(!notice)return;const t=window.setTimeout(()=>setNotice(''),5500);return()=>window.clearTimeout(t)},[notice])
  useEffect(()=>{if(!recording)return;const i=window.setInterval(()=>setSeconds(v=>v+1),1000);return()=>window.clearInterval(i)},[recording])
  useEffect(()=>()=>{if(ownedAudioUrl.current)URL.revokeObjectURL(ownedAudioUrl.current);if(recorder.current?.state==='recording')recorder.current.stop()},[])
  const storeAudio = (blob:Blob,name:string) => {if(ownedAudioUrl.current)URL.revokeObjectURL(ownedAudioUrl.current);const url=URL.createObjectURL(blob);ownedAudioUrl.current=url;setAudioUrl(url);setAudioName(name)}
  const startRecording=async()=>{
    if(!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder==='undefined'){setNotice('Recording is not supported in this browser. Try uploading an audio file.');return}
    try{
      const stream=await navigator.mediaDevices.getUserMedia({audio:true})
      const mr=new MediaRecorder(stream)
      chunks.current=[]
      mr.ondataavailable=e=>{if(e.data.size)chunks.current.push(e.data)}
      mr.onstop=()=>{const blob=new Blob(chunks.current,{type:mr.mimeType||'audio/webm'});if(blob.size)storeAudio(blob,'New voice check-in');stream.getTracks().forEach(t=>t.stop())}
      mr.onerror=()=>{stream.getTracks().forEach(t=>t.stop());setNotice('An audio recording error occurred.')}
      recorder.current=mr;setSeconds(0);mr.start();setRecording(true)
    }catch{setNotice('Microphone permission was not granted. You can upload an audio file instead.')}
  }
  const stopRecording=()=>{if(recorder.current?.state==='recording'){recorder.current.stop();setRecording(false);setNotice('Audio captured locally. Automated voice analysis is not connected in this UI prototype.')}}
  const onFile=(ev:ChangeEvent<HTMLInputElement>)=>{const file=ev.target.files?.[0];if(!file)return; if(!file.type.startsWith('audio/')){setNotice('Please choose an audio file.');return}storeAudio(file,file.name);setNotice('Audio ready for playback. Analysis pipeline integration is still required.');ev.target.value=''}
  const changeMetric=(next:Metric)=>setMetric(next)
  const compare = useMemo(()=>({pace:latest.pace-100,pauses:latest.pauses-100,consistency:latest.consistency}),[latest])

  return <div className="app-shell">
    {mobileNav && <button aria-label="Close navigation" className="mobile-backdrop" onClick={()=>setMobileNav(false)}/>}
    <aside className={`sidebar ${mobileNav?'sidebar-open':''}`}>
      <div className="sidebar-brand"><Mark light/><div className="brand-type">voice<span>drift</span><small>VOICE WELLNESS</small></div></div>
      <div className="sidebar-content"><div className="nav-eyebrow">WORKSPACE</div><nav aria-label="Main navigation">
        {([['overview','dashboard'],['checkin','mic'],['insights','chart'],['review','queue'],['results','layers']] as [Page,string][]).map(([key,icon])=><button key={key} className={`nav-link ${page===key?'active':''}`} onClick={()=>go(key)}><Icon name={icon}/><span>{pageNames[key]}</span>{key==='review'&&flagged.length>0&&<b className="nav-count">{flagged.length}</b>}{page===key&&<span className="nav-active-marker"/>}</button>)}
      </nav></div>
      <div className="sidebar-bottom"><div className="sidebar-note"><Icon name="spark" size={18}/><strong>A little more insight.</strong><p>Understand change relative to what’s normal for you.</p></div><div className="sidebar-account"><span className="account-avatar">V</span><span><strong>Visitor profile</strong><small>Illustrative workspace</small></span><span className="account-more">•••</span></div></div>
    </aside>
    <div className="app-main">
      <header className="topbar"><div className="topbar-left"><button className="icon-button hamburger" onClick={()=>setMobileNav(true)} aria-label="Open navigation"><Icon name="menu"/></button><span className="breadcrumb-root">Your workspace</span><Icon name="chevron" size={15}/><strong>{pageNames[page]}</strong></div><div className="topbar-right"><span className="sample-label"><span/>DEMO ENVIRONMENT</span><button className="top-action" onClick={()=>setNotice('This is an illustrative UI. No health notifications are connected.')} aria-label="Notifications"><Icon name="bell" size={19}/></button><span className="top-avatar">V</span></div></header>
      <main className="main-content">
        <div className="demo-ribbon"><Icon name="info" size={16}/><span><strong>Prototype preview:</strong> All trends, thresholds and flags are illustrative, not patient data or validated medical findings.</span>{demoActive && <button onClick={resetDemo}>Reset scenario <Icon name="refresh" size={14}/></button>}</div>
        {page==='overview' && <>
          <PageHeading eyebrow="VOICE HEALTH, REIMAGINED" title="The story behind your voice." description="Small changes feel more meaningful when you can see them in the context of your own usual patterns." actions={<button className="btn btn-outline" onClick={()=>go('checkin')}><Icon name="mic" size={17}/>New check-in</button>}/>
          <section className="hero">
            <div className="hero-content"><div className="hero-kicker"><span className="hero-kicker-dot"/>PERSONALIZED VOICE TRACKING</div><h2>Your voice is unique.<br/><em>Your baseline should be, too.</em></h2><p>Spot meaningful patterns over time — without confusing a different room or a slower morning for a diagnosis.</p><div className="hero-actions"><button className="btn btn-lime" onClick={launchDemo}><Icon name="play" size={18}/>Run guided demo<Icon name="arrow" size={18}/></button><button className="btn hero-text-button" onClick={()=>go('insights')}>Explore insights <Icon name="arrow" size={17}/></button></div></div>
            <div className="hero-visual" aria-hidden="true"><div className="orbit orbit-1"/><div className="orbit orbit-2"/><div className="hero-visual-label"><span>VOICE FINGERPRINT</span><span className="hero-visual-pill">PERSONAL</span></div><div className="voice-orb"><span className="voice-orb-core"><Icon name="sound" size={61} strokeWidth={1.15}/></span></div><div className="hero-visual-bottom"><Waveform compact/><span>steady · familiar · yours</span></div></div>
          </section>
          <div className="section-title-row"><div><div className="eyebrow eyebrow-plain">THE CURRENT PICTURE</div><h2>At a glance <span>· latest check-in</span></h2></div><span className="section-note"><Icon name="calendar" size={16}/>14 Nov, 2026</span></div>
          <div className="metric-grid">
            <button className="metric-card metric-card-blue" onClick={()=>{changeMetric('pace');go('insights')}}><div className="metric-top"><span className="metric-icon blue"><Icon name="chart"/></span><Icon name="arrow" size={17}/></div><span className="metric-label">Speech pace</span><div className="metric-value">{Math.abs(compare.pace)}<small>% {compare.pace<0?'slower':compare.pace>0?'faster':'change'}</small></div><div className="metric-bottom"><span>vs. usual pace</span><MiniSpark values={readings.map(r=>r.pace)} color="#4a85d8"/></div></button>
            <button className="metric-card metric-card-peach" onClick={()=>{changeMetric('pauses');go('insights')}}><div className="metric-top"><span className="metric-icon coral"><Icon name="clock"/></span><Icon name="arrow" size={17}/></div><span className="metric-label">Pause duration</span><div className="metric-value">{Math.abs(compare.pauses)}<small>% {compare.pauses>0?'longer':compare.pauses<0?'shorter':'change'}</small></div><div className="metric-bottom"><span>vs. usual pause</span><MiniSpark values={readings.map(r=>r.pauses)} color="#df896c"/></div></button>
            <button className="metric-card metric-card-lavender" onClick={()=>go('review')}><div className="metric-top"><span className="metric-icon purple"><Icon name="queue"/></span><Icon name="arrow" size={17}/></div><span className="metric-label">Check-ins to review</span><div className="metric-value">{flagged.length}<small>of {readings.length} check-ins</small></div><div className="small-pills">{readings.map((r,i)=><span key={r.id} className={r.flagged?'pill-flag':'pill-good'} title={`${r.date}: ${r.flagged?'Outside range':'Within range'}`}>{i+1}</span>)}</div></button>
            <button className="metric-card metric-card-green" onClick={()=>{changeMetric('consistency');go('insights')}}><div className="metric-top"><span className="metric-icon green"><Icon name="sound"/></span><Icon name="arrow" size={17}/></div><span className="metric-label">Consistency score</span><div className="metric-gauge"><div className="gauge" style={{'--progress':`${latest.consistency}%`} as CSSProperties}><span>{latest.consistency}<small>/100</small></span></div><div className="gauge-copy">Demo-only<br/>pattern score</div></div></button>
          </div>
          <div className="overview-columns"><section className="panel chart-panel"><div className="panel-heading"><div><h3>How your voice is shifting</h3><p>Patterns across five example check-ins</p></div><button className="plain-link" onClick={()=>go('insights')}>View all insights <Icon name="arrow" size={16}/></button></div><div className="metric-tabs" role="tablist" aria-label="Trend metric">{(['pace','pauses','consistency'] as Metric[]).map(k=><button role="tab" aria-selected={metric===k} key={k} className={metric===k?'selected':''} onClick={()=>changeMetric(k)}>{metricInfo[k].label}</button>)}</div><ChartView data={readings} metric={metric}/></section>
            <section className={`panel attention-panel ${demoActive?'attention-active':''}`}><div className="attention-icon"><Icon name={demoActive?'alert':'checkcircle'} size={24}/></div><div className="eyebrow eyebrow-plain">WHAT WE'RE NOTICING</div><h3>{demoActive?'A pattern worth reviewing':'Looking familiar so far'}</h3><p>{demoActive?'Several consecutive example recordings differ from the baseline. That is a prompt to look closer, not a diagnosis.':'Your example check-ins are tracking close to the personal reference range.'}</p><div className="attention-stat"><strong>{flagged.length}<span> / 5</span></strong><span>check-ins outside<br/>the example range</span></div><button className="btn btn-ink btn-full" onClick={()=>go('review')}>{demoActive?'Open review queue':'View check-in history'}<Icon name="arrow" size={17}/></button></section></div>
          <div className="section-title-row history-title"><div><div className="eyebrow eyebrow-plain">YOUR TRACK RECORD</div><h2>Recent check-ins</h2></div><button className="plain-link" onClick={()=>go('review')}>See all <Icon name="arrow" size={16}/></button></div>
          <section className="panel history-panel"><div className="history-rows">{readings.slice().reverse().slice(0,4).map((r)=><button className="history-row" key={r.id} onClick={()=>{setSelectedId(r.id);setReviewFilter('all');go('review')}}><span className={`history-item-symbol ${r.flagged?'history-symbol-flag':''}`}><Icon name={r.flagged?'sound':'check'} size={18}/></span><span className="history-info"><strong>{r.date} check-in</strong><small>{r.time} · demo audio profile</small></span><span className="history-status"><Status flagged={r.flagged}/></span><Icon name="chevron" size={18}/></button>)}</div></section>
          <section className="education-strip"><div className="education-art"><Icon name="shield" size={35}/></div><div><strong>Built around your baseline. Not a population average.</strong><p>Voice patterns vary from person to person. This prototype illustrates longitudinal comparison while keeping uncertainty visible.</p></div><button className="plain-link" onClick={()=>go('results')}>How it works <Icon name="arrow" size={16}/></button></section>
        </>}
        {page==='checkin' && <>
          <PageHeading eyebrow="YOUR VOICE ROUTINE" title="Take a voice check-in." description="A quiet minute, a familiar reading, and a way to keep your recordings comparable over time."/>
          <div className="checkin-layout"><section className="reading-card"><div className="reading-card-top"><span className="reading-icon"><Icon name="file" size={21}/></span><span>01 / READING PASSAGE</span></div><h2>A short moment<br/>in your day.</h2><p className="reading-intro">For consistency, read this same passage at a comfortable, natural pace each time.</p><blockquote>“Every morning, the little bakery on the corner opens its doors just as the neighborhood begins to wake. The smell of fresh bread drifts into the street, and people stop for a moment before beginning their day.”</blockquote><div className="reading-foot"><Icon name="info" size={17}/>Speak naturally; there are no right or wrong answers.</div></section>
            <section className="panel recorder-card"><div className="recorder-heading"><div><div className="eyebrow eyebrow-plain">02 / RECORD YOUR VOICE</div><h3>Ready when you are.</h3><p>Find a relatively quiet space and read the passage aloud.</p></div></div><div className={`recording-visual ${recording?'is-recording':''}`}><div className="recording-circle"><Icon name="mic" size={32}/></div><Waveform/><span className="recording-time">{recording?`${Math.floor(seconds/60).toString().padStart(2,'0')}:${(seconds%60).toString().padStart(2,'0')}`:'00:00'}</span><span className="recording-state">{recording?'Recording in progress':'Microphone ready'}</span></div><button className={`btn ${recording?'btn-stop':'btn-ink'} btn-full record-button`} onClick={recording?stopRecording:startRecording}><Icon name={recording?'stop':'mic'} size={18}/>{recording?'Stop recording':'Start recording'}</button><div className="upload-divider"><span>or</span></div><input ref={fileInput} type="file" accept="audio/*" className="visually-hidden" onChange={onFile}/><button className="btn btn-outline btn-full" onClick={()=>fileInput.current?.click()}><Icon name="upload" size={17}/>Upload an audio file</button>{audioUrl&&<div className="playback-card"><strong><Icon name="checkcircle" size={17}/>Audio ready: {audioName}</strong><audio controls src={audioUrl}/><small>Stored in this browser session for playback. This prototype does not analyze or upload the recording.</small></div>}</section></div>
          <section className="checkin-guidance"><div><span className="guidance-number">01</span><strong>Same words</strong><p>Use the same passage to make your check-ins easier to compare.</p></div><div><span className="guidance-number">02</span><strong>Similar setting</strong><p>Reduce avoidable noise and distance changes where possible.</p></div><div><span className="guidance-number">03</span><strong>No diagnosis</strong><p>Any change should be interpreted with context and appropriate clinical advice.</p></div></section>
        </>}
        {page==='insights' && <>
          <PageHeading eyebrow="TRENDS & SIGNALS" title="The details of change." description="Switch between signal families to see what shifted, how much it shifted, and when it happened." actions={<DownloadButton readings={readings}/>}/>
          <div className="insights-grid"><section className="panel deep-chart-panel"><div className="panel-heading"><div><h3>Relative to your personal baseline</h3><p>Five illustrative check-ins · baseline indexed to 100</p></div><span className="chart-annotation"><span/>Usual range shown in green</span></div><div className="metric-tabs metric-tabs-wide" role="tablist" aria-label="Insights metric">{(['pace','pauses','consistency'] as Metric[]).map(k=><button role="tab" aria-selected={metric===k} key={k} className={metric===k?'selected':''} onClick={()=>setMetric(k)}>{metricInfo[k].label}</button>)}</div><ChartView metric={metric} data={readings}/><div className="insight-summary"><span className="insight-visual"><Icon name="info" size={20}/></span><p><strong>{metricInfo[metric].label}:</strong> {metricInfo[metric].description} This chart shows example values only; the shaded band is not a medically validated cutoff.</p></div></section><div className="insight-side"><div className="insight-feature-card"><div className="eyebrow eyebrow-plain">LATEST RELATIVE VALUE</div><strong>{latest[metric]}<span>{metric==='consistency'?'/100':'%'}</span></strong><p>{metricInfo[metric].unit}</p><div className="feature-rule"/><Status flagged={latest.flagged} label={latest.flagged?'Latest example flagged':'Latest example in range'}/></div><div className="insight-explainer"><span className="explainer-asterisk">✳</span><h3>Why a personal baseline?</h3><p>People speak at different speeds and with different rhythms. Looking at each person’s own history can provide richer context than relying on a single universal number.</p></div></div></div>
          <div className="section-title-row"><div><div className="eyebrow eyebrow-plain">FEATURE BREAKDOWN</div><h2>Three ways to look at change</h2></div></div><div className="feature-grid">{(['pace','pauses','consistency'] as Metric[]).map((k,i)=><button key={k} className={`feature-tile feature-${i}`} onClick={()=>{setMetric(k);window.scrollTo({top:0,behavior:'smooth'})}}><span className="feature-tile-index">0{i+1}</span><span className="feature-tile-icon"><Icon name={i===0?'chart':i===1?'clock':'sound'} size={25}/></span><h3>{metricInfo[k].label}</h3><p>{metricInfo[k].description}</p><span className="feature-tile-footer">Explore signal <Icon name="arrow" size={17}/></span></button>)}</div>
        </>}
        {page==='review' && <>
          <PageHeading eyebrow="REVIEW WITH CONTEXT" title="Not every change is a concern." description="A review queue helps surface consecutive deviations while keeping human judgment in the loop." actions={<button className="btn btn-ink" onClick={launchDemo}><Icon name="play" size={16}/>Load drift example</button>}/>
          <div className="review-banner"><div className="review-banner-symbol"><Icon name={flagged.length?'alert':'checkcircle'} size={26}/></div><div><strong>{flagged.length?`${flagged.length} example check-ins need a closer look`:'No check-ins are flagged'}</strong><p>{summary}. These flags are illustrative and are not clinical alerts.</p></div><span>{flagged.length ? 'REVIEW SUGGESTED' : 'ALL CLEAR IN DEMO'}</span></div>
          <div className="review-layout"><section className="panel queue-panel"><div className="queue-header"><h3>Check-in history</h3><div className="queue-filter" role="tablist" aria-label="Review queue filter"><button role="tab" aria-selected={reviewFilter==='flagged'} className={reviewFilter==='flagged'?'active':''} onClick={()=>setReviewFilter('flagged')}>Flagged ({flagged.length})</button><button role="tab" aria-selected={reviewFilter==='all'} className={reviewFilter==='all'?'active':''} onClick={()=>setReviewFilter('all')}>All ({readings.length})</button></div></div>{queue.length===0?<div className="queue-empty"><span><Icon name="checkcircle" size={30}/></span><h4>Nothing waiting in the queue</h4><p>Run the guided demo to see how a multi-check-in drift pattern might be reviewed.</p><button className="btn btn-ink btn-sm" onClick={launchDemo}>Run drift example <Icon name="arrow" size={15}/></button></div>:<div className="queue-list">{queue.map(r=><button key={r.id} className={`queue-item ${selected.id===r.id?'selected':''}`} onClick={()=>setSelectedId(r.id)}><span className={`queue-item-icon ${r.flagged?'queue-item-flag':''}`}><Icon name={r.flagged?'alert':'sound'} size={20}/></span><span className="queue-item-content"><strong>{r.date}, 2026</strong><small>{r.time} · {r.id}</small><Status flagged={r.flagged}/></span><Icon name="chevron" size={18}/></button>)}</div>}</section><section className="panel review-detail"><div className="detail-header"><div><div className="eyebrow eyebrow-plain">SELECTED CHECK-IN</div><h3>{selected.date}, 2026</h3><p>{selected.time} · {selected.id}</p></div><Status flagged={selected.flagged}/></div><div className="detail-wave"><Waveform/><span>Illustrative acoustic pattern</span></div><h4>Relative to personal baseline</h4><div className="detail-measures">{([['Speech pace',selected.pace,'pace'],['Pause duration',selected.pauses,'pauses'],['Consistency',selected.consistency,'consistency']] as [string,number,Metric][]).map(([name,value,key])=><div className="detail-measure" key={key}><div><strong>{name}</strong><span>{value}{key==='consistency'?'/100':'%'}</span></div><div className="measure-track"><span style={{width:`${Math.min(100,Math.max(4,value/(key==='pauses'?175:key==='pace'?115:100)*100))}%`,background:metricInfo[key].color}}/></div></div>)}</div><div className={`detail-note ${selected.flagged?'detail-note-warn':''}`}><Icon name={selected.flagged?'alert':'info'} size={19}/><p>{selected.flagged?'This check-in is outside an illustrative range. Repeated changes may deserve a conversation with a qualified professional, particularly if symptoms are present.':'This example falls within the illustrative range. A single reading is never enough to establish health status.'}</p></div></section></div>
        </>}
        {page==='results' && <>
          <PageHeading eyebrow="TRANSPARENCY FIRST" title="What this prototype can — and can’t — say." description="The experience is a design demonstration. Reliable health insights require independent technical and clinical validation." actions={<DownloadButton readings={readings}/>}/>
          <section className="methods-hero"><div><span className="methods-overline">HOW VOICE DRIFT IS DESIGNED</span><h2>From a small recording to a clearer question.</h2><p>One familiar passage. Repeated over time. Compared with a personal reference — not with an arbitrary person.</p></div><div className="methods-shape"><Icon name="sound" size={72} strokeWidth={1.1}/></div></section>
          <div className="pipeline-grid">{[['01','Capture','Record a consistent passage in a comparable setting.','mic'],['02','Measure','Extract pace, pauses and other acoustic features once an analysis pipeline is integrated.','chart'],['03','Compare','Evaluate changes relative to prior readings with quality checks.','layers'],['04','Review','Surface possible drift with uncertainty and human oversight.','queue']].map(([step,title,body,icon])=><div className="pipeline-card" key={step}><div className="pipeline-head"><span>{step}</span><Icon name={icon} size={22}/></div><strong>{title}</strong><p>{body}</p></div>)}</div>
          <div className="methods-columns"><section className="panel validation-panel"><div className="eyebrow eyebrow-plain">CURRENT DEMO STATUS</div><h3>Evidence, not embellishment.</h3><p>These checks are intentionally visible rather than implying finished experimental results.</p><div className="validation-rows"><div><span className="validation-icon complete"><Icon name="check" size={17}/></span><span><strong>Interactive user interface</strong><small>Navigation, sample scenario, recording, review and CSV export</small></span><span className="validation-tag built">Built</span></div><div><span className="validation-icon pending"><Icon name="clock" size={17}/></span><span><strong>Live feature extraction</strong><small>Requires an audio processing and inference backend</small></span><span className="validation-tag">Pending</span></div><div><span className="validation-icon pending"><Icon name="clock" size={17}/></span><span><strong>Calibrated drift thresholds</strong><small>Clinical utility and error rates are not established here</small></span><span className="validation-tag">Pending</span></div><div><span className="validation-icon pending"><Icon name="clock" size={17}/></span><span><strong>Noise robustness validation</strong><small>Controlled clean-speech comparisons are needed</small></span><span className="validation-tag">Pending</span></div></div></section><section className="disclosure-panel"><Icon name="shield" size={31}/><h3>Designed for reflection, not diagnosis.</h3><p>Changes in voice can reflect many factors, including fatigue, emotion, illness, recording conditions and natural variation. This prototype should not be used to screen, diagnose, or monitor a medical condition.</p><span>DEMO DATA ONLY · NO CLINICAL CLAIMS</span></section></div>
        </>}
        <footer className="site-footer"><span><Mark/> Voice Drift <span className="footer-divider">/</span> Find meaning in the little shifts.</span><span>Illustrative prototype · 2026</span></footer>
      </main>
    </div>
    {notice && <div className="toast" role="status"><Icon name="info" size={19}/><span>{notice}</span><button aria-label="Dismiss notification" onClick={()=>setNotice('')}><Icon name="close" size={16}/></button></div>}
  </div>
}

export default App
