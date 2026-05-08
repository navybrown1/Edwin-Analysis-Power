"""One-shot generator: embed MBA Grads.xlsx + pwc-logo.svg into Candidate_Dashboard.html."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
XLSX = BASE / "MBA Grads.xlsx"
SVG_PATH = BASE / "pwc-logo.svg"
OUT = BASE / "Candidate_Dashboard.html"


def load_svg_inline() -> str:
    raw = SVG_PATH.read_text(encoding="utf-8")
    if raw.strip().startswith("<?xml"):
        raw = raw.split(">", 1)[1].strip()
    return raw


def main() -> None:
    df = pd.read_excel(XLSX, "Offers")
    rows = df.replace({float("nan"): None}).to_dict(orient="records")
    payload = json.dumps(rows, default=str, separators=(",", ":"))
    svg = load_svg_inline()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>PwC · Candidate Profiler · MBA Offers</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600&family=Inter+Tight:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@400;500;600&display=swap" rel="stylesheet" />
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --app-chrome: 11.5rem;
    --chart-h: clamp(110px, 16vh, 200px);
    --chart-short: clamp(88px, 12vh, 140px);
    --ion-radius: 14px;
    --bc-navy: #2b2b2b;
    --bc-blue: #d04a02;
    --bc-blue-light: #e8e8e8;
    --bc-red: #c0392b;
    --ink: #f7f5f4;
    --line: rgba(43, 43, 43, 0.12);
    --line-2: rgba(43, 43, 43, 0.2);
    --cream: #2b2b2b;
    --muted: #707070;
    --muted-2: #4a5378;
    --gold: #d04a02;
    --green: #1a7f5c;
    --red: #c0392b;
    --header-text: #f7f5f4;
    --on-accent: #ffffff;
    --ion-glow: 0 0 0 1px rgba(43, 43, 43, 0.06), 0 10px 32px rgba(43, 43, 43, 0.08);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    background: var(--ink);
    color: var(--cream);
    font-family: 'Inter Tight', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
    height: 100%;
    overflow: hidden;
  }}
  body {{
    background:
      radial-gradient(1000px 520px at 92% -8%, rgba(208, 74, 2, 0.14), transparent 55%),
      radial-gradient(800px 480px at -5% 105%, rgba(217, 217, 217, 0.45), transparent 50%),
      var(--ink);
  }}
  .shell {{
    max-width: 1600px;
    margin: 0 auto;
    padding: 10px 20px 8px;
    height: 100vh;
    max-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}
  .dashboard-body {{ flex: 1; min-height: 0; position: relative; }}
  header.top {{
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    padding: 10px 14px 12px;
    margin: 0 -4px 6px;
    border-bottom: 1px solid rgba(217, 217, 217, 0.35);
    background: linear-gradient(165deg, #242a66 0%, #2b2b2b 48%, #161a4a 100%);
    backdrop-filter: blur(14px);
    border-radius: 0 0 var(--ion-radius) var(--ion-radius);
    flex-shrink: 0;
    color: var(--header-text);
  }}
  .brand-row {{ display: flex; align-items: flex-start; gap: 14px; margin-bottom: 4px; }}
  .bc-logo-link {{ flex-shrink: 0; line-height: 0; border-radius: 10px; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2); }}
  .bc-logo-wrap {{
    width: 44px; height: 44px;
    background: #fff;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 6px;
  }}
  .bc-logo-wrap svg {{ width: 100%; height: auto; max-height: 32px; }}
  .crumb {{
    font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
    color: rgba(247, 245, 244, 0.68);
    font-weight: 500;
  }}
  h1.title {{
    font-family: 'Fraunces', serif;
    font-weight: 400;
    font-size: clamp(1.25rem, 2.2vw, 1.75rem);
    line-height: 1.1;
    letter-spacing: -0.02em;
    color: #ffffff;
  }}
  h1.title em {{ font-style: italic; color: var(--bc-blue); font-weight: 400; }}
  .subtitle {{
    margin-top: 4px;
    font-size: 11px;
    color: rgba(247, 245, 244, 0.82);
    max-width: 44rem;
    line-height: 1.4;
  }}
  .ctrl-cluster {{ display: flex; flex-direction: column; align-items: flex-end; gap: 10px; }}
  .filter-row {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: flex-end; }}
  .sel-label {{ font-size: 10px; color: rgba(247, 245, 244, 0.72); letter-spacing: 0.1em; text-transform: uppercase; }}
  .sel {{
    background: #f7f5f4;
    color: #2b2b2b;
    border: 1px solid rgba(43, 43, 43, 0.22);
    padding: 6px 28px 6px 12px;
    font-size: 12px;
    border-radius: 4px;
    font-family: inherit;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpath fill='%232B2B2B' d='M5 7L1 3h8z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
    cursor: pointer;
  }}
  .sel:hover {{ border-color: rgba(208, 74, 2, 0.65); }}
  .btn-reset {{
    background: var(--bc-red);
    color: #fff;
    border: 0;
    padding: 6px 14px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-radius: 6px;
    cursor: pointer;
    font-family: inherit;
  }}
  .btn-reset:hover {{ filter: brightness(1.08); }}
  nav.tabs {{
    display: flex;
    gap: 4px;
    padding: 4px;
    margin: 0 0 6px;
    border: 1px solid var(--line-2);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.92);
    overflow-x: auto;
    flex-shrink: 0;
    scrollbar-width: thin;
    box-shadow: 0 1px 0 rgba(43, 43, 43, 0.04);
  }}
  nav.tabs button {{
    background: transparent;
    border: 0;
    color: var(--muted);
    padding: 10px 16px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.04em;
    cursor: pointer;
    white-space: nowrap;
    font-family: inherit;
    transition: color 0.18s;
  }}
  nav.tabs button:hover {{ color: var(--bc-navy); }}
  nav.tabs button.active {{ color: var(--bc-navy); position: relative; }}
  nav.tabs button.active::after {{
    content: '';
    position: absolute;
    left: 10px; right: 10px; bottom: 4px;
    height: 2px;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--bc-blue), var(--bc-red));
  }}
  .tab-panel {{
    display: none;
    position: absolute;
    inset: 0;
    overflow: hidden;
    flex-direction: column;
  }}
  .tab-panel.active {{ display: flex; animation: fade 0.22s ease; }}
  .tab-scroll {{
    flex: 1;
    min-height: 0;
    overflow: auto;
    padding: 0 4px 8px;
    scrollbar-width: thin;
    scrollbar-color: rgba(208, 74, 2, 0.4) transparent;
  }}
  @keyframes fade {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: none; }} }}
  .kpi-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    border: 1px solid rgba(217, 217, 217, 0.35);
    border-radius: var(--ion-radius);
    overflow: hidden;
    margin-bottom: 10px;
    background: linear-gradient(135deg, #22275e 0%, #2b2b2b 55%, #161a4a 100%);
    box-shadow: var(--ion-glow);
    color: var(--header-text);
  }}
  @media (max-width: 900px) {{
    .kpi-strip {{ grid-template-columns: repeat(2, 1fr); }}
  }}
  .kpi {{
    padding: 10px 14px;
    border-right: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
  }}
  .kpi:last-child {{ border-right: 0; }}
  .kpi .klabel {{
    font-size: 10px;
    color: rgba(247, 245, 244, 0.7);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 10px;
  }}
  .kpi .kval {{
    font-family: 'Fraunces', serif;
    font-weight: 500;
    font-size: clamp(1.2rem, 2.2vw, 1.65rem);
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1;
  }}
  .kpi .ksub {{
    font-size: 11px;
    color: rgba(247, 245, 244, 0.72);
    margin-top: 8px;
  }}
  .panel-grid {{
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 10px;
    margin-bottom: 10px;
  }}
  .panel {{
    background: linear-gradient(180deg, #ffffff 0%, #fafcfd 100%);
    border: 1px solid var(--line-2);
    border-radius: var(--ion-radius);
    padding: 10px 12px;
    box-shadow: var(--ion-glow);
  }}
  .panel.accent-left {{
    border-left: 3px solid var(--bc-red);
    box-shadow: inset 0 0 0 1px rgba(208, 74, 2, 0.08), var(--ion-glow);
  }}
  .panel-head {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--line);
  }}
  .panel-title {{
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: clamp(0.95rem, 1.2vw, 1.05rem);
    color: var(--cream);
  }}
  .panel-sub {{
    font-size: 10px;
    color: var(--muted);
    margin-top: 3px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }}
  .panel-tag {{
    font-size: 9px;
    padding: 3px 8px;
    border-radius: 3px;
    background: rgba(208, 74, 2, 0.12);
    color: var(--bc-navy);
    border: 1px solid rgba(208, 74, 2, 0.35);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
  }}
  .col-6 {{ grid-column: span 6; }}
  .col-12 {{ grid-column: span 12; }}
  @media (max-width: 960px) {{
    .col-6 {{ grid-column: span 12; }}
  }}
  .chart-box {{ position: relative; height: var(--chart-h); min-height: 0; }}
  .chart-box.short {{ height: var(--chart-short); max-height: var(--chart-short); }}
  .search-bar {{
    margin-bottom: 10px;
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
  }}
  .search-bar input {{
    flex: 1;
    min-width: 200px;
    padding: 8px 12px;
    border: 1px solid var(--line-2);
    border-radius: 8px;
    font-family: inherit;
    font-size: 13px;
  }}
  .table-wrap {{
    border: 1px solid var(--line-2);
    border-radius: var(--ion-radius);
    overflow: auto;
    max-height: min(56vh, 520px);
    background: #fff;
    box-shadow: var(--ion-glow);
  }}
  table.data {{
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
  }}
  table.data th {{
    position: sticky;
    top: 0;
    background: linear-gradient(180deg, #fafcfd 0%, #f0f2f8 100%);
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--line-2);
    font-weight: 600;
    color: var(--bc-navy);
    white-space: nowrap;
    z-index: 1;
  }}
  table.data td {{
    padding: 7px 10px;
    border-bottom: 1px solid rgba(43,43,43,0.08);
    vertical-align: top;
  }}
  table.data tr:hover td {{ background: rgba(208, 74, 2, 0.06); }}
  .pager {{
    display: flex;
    gap: 10px;
    align-items: center;
    justify-content: flex-end;
    margin-top: 10px;
    font-size: 12px;
    color: var(--muted);
  }}
  .pager button {{
    background: #fff;
    border: 1px solid var(--line-2);
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-family: inherit;
  }}
  .pager button:disabled {{ opacity: 0.45; cursor: not-allowed; }}
  .snapshot-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }}
  @media (max-width: 700px) {{ .snapshot-grid {{ grid-template-columns: 1fr; }} }}
  .snap-item {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 12px 14px;
    background: #e8eef9;
    border-radius: 8px;
    border-left: 3px solid var(--bc-blue);
  }}
  .snap-label {{ font-size: 11px; color: var(--muted); }}
  .snap-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    color: var(--cream);
    font-weight: 500;
  }}
</style>
</head>
<body>
<div class="shell">
  <header class="top">
    <div>
      <div class="brand-row">
        <div class="bc-logo-link" aria-hidden="true">
          <div class="bc-logo-wrap">{svg}</div>
        </div>
        <div>
          <div class="crumb">People analytics · <b>MBA pipeline</b></div>
          <h1 class="title">Candidate Profiler · <em>MBA Offers</em></h1>
          <p class="subtitle">Interactive view of simulated MBA graduate outcomes—school mix, post-MBA roles, compensation, and offer flags. Adjust filters to explore cohorts.</p>
        </div>
      </div>
    </div>
    <div class="ctrl-cluster">
      <div class="filter-row">
        <span class="sel-label">Year</span>
        <select id="f-year" class="sel" aria-label="Graduation year"></select>
        <span class="sel-label">School</span>
        <select id="f-school" class="sel" aria-label="School"></select>
        <span class="sel-label">Industry</span>
        <select id="f-industry" class="sel" aria-label="Post-MBA industry"></select>
        <span class="sel-label">Program</span>
        <select id="f-program" class="sel" aria-label="Program type"></select>
        <button type="button" class="btn-reset" id="btn-reset">Reset</button>
      </div>
    </div>
  </header>
  <nav class="tabs" role="tablist" aria-label="Dashboard sections">
    <button type="button" class="active" role="tab" aria-selected="true" data-tab="t1">Overview</button>
    <button type="button" role="tab" aria-selected="false" data-tab="t2">Offer mix</button>
    <button type="button" role="tab" aria-selected="false" data-tab="t3">School bench</button>
    <button type="button" role="tab" aria-selected="false" data-tab="t4">Directory</button>
  </nav>
  <div class="dashboard-body">
    <section id="t1" class="tab-panel active" role="tabpanel">
      <div class="tab-scroll">
        <div class="kpi-strip" id="kpi-strip"></div>
        <div class="panel-grid">
          <div class="panel accent-left col-6">
            <div class="panel-head">
              <div>
                <div class="panel-title">Post-MBA industry</div>
                <div class="panel-sub">Where graduates land</div>
              </div>
              <span class="panel-tag">mix</span>
            </div>
            <div class="chart-box"><canvas id="chart-industry"></canvas></div>
          </div>
          <div class="panel col-6">
            <div class="panel-head">
              <div>
                <div class="panel-title">School distribution</div>
                <div class="panel-sub">Share of filtered cohort</div>
              </div>
              <span class="panel-tag">%</span>
            </div>
            <div class="chart-box short"><canvas id="chart-school"></canvas></div>
          </div>
          <div class="panel col-6">
            <div class="panel-head">
              <div>
                <div class="panel-title">Graduation year</div>
                <div class="panel-sub">Headcount by year</div>
              </div>
              <span class="panel-tag">time</span>
            </div>
            <div class="chart-box"><canvas id="chart-year"></canvas></div>
          </div>
          <div class="panel accent-left col-6">
            <div class="panel-head">
              <div>
                <div class="panel-title">GPA vs post-MBA salary</div>
                <div class="panel-sub">Filtered sample (max 400 pts)</div>
              </div>
              <span class="panel-tag">scatter</span>
            </div>
            <div class="chart-box"><canvas id="chart-scatter"></canvas></div>
          </div>
        </div>
      </div>
    </section>
    <section id="t2" class="tab-panel" role="tabpanel">
      <div class="tab-scroll">
        <div class="panel-grid">
          <div class="panel col-12">
            <div class="panel-head">
              <div>
                <div class="panel-title">Flagged offers</div>
                <div class="panel-sub">Share answering “Yes” for Big Tech, Consulting, and Big Banks</div>
              </div>
              <span class="panel-tag">rates</span>
            </div>
            <div class="chart-box"><canvas id="chart-offers"></canvas></div>
          </div>
          <div class="panel col-12">
            <div class="panel-head">
              <div>
                <div class="panel-title">Internship pathway</div>
                <div class="panel-sub">Internship completed vs not (filtered)</div>
              </div>
              <span class="panel-tag">intern</span>
            </div>
            <div class="chart-box short"><canvas id="chart-intern"></canvas></div>
          </div>
        </div>
      </div>
    </section>
    <section id="t3" class="tab-panel" role="tabpanel">
      <div class="tab-scroll">
        <div class="panel-grid">
          <div class="panel accent-left col-6">
            <div class="panel-head">
              <div>
                <div class="panel-title">Avg post-MBA salary by school</div>
                <div class="panel-sub">Mean compensation</div>
              </div>
              <span class="panel-tag">$</span>
            </div>
            <div class="chart-box"><canvas id="chart-sal-school"></canvas></div>
          </div>
          <div class="panel col-6">
            <div class="panel-head">
              <div>
                <div class="panel-title">Avg GMAT by school</div>
                <div class="panel-sub">Admissions signal</div>
              </div>
              <span class="panel-tag">exam</span>
            </div>
            <div class="chart-box"><canvas id="chart-gmat-school"></canvas></div>
          </div>
          <div class="panel col-12">
            <div class="panel-head">
              <div>
                <div class="panel-title">Snapshot</div>
                <div class="panel-sub">Filtered cohort (same filters as header)</div>
              </div>
            </div>
            <div class="snapshot-grid" id="bench-snap"></div>
          </div>
        </div>
      </div>
    </section>
    <section id="t4" class="tab-panel" role="tabpanel">
      <div class="tab-scroll">
        <div class="panel col-12" style="margin-bottom:10px">
          <div class="search-bar">
            <input type="search" id="table-search" placeholder="Search name, school, industry, location, function…" autocomplete="off" />
            <span class="panel-tag" id="table-count"></span>
          </div>
          <div class="table-wrap">
            <table class="data" id="data-table">
              <thead><tr id="data-thead"></tr></thead>
              <tbody id="data-tbody"></tbody>
            </table>
          </div>
          <div class="pager">
            <span id="page-info"></span>
            <button type="button" id="page-prev">Prev</button>
            <button type="button" id="page-next">Next</button>
          </div>
        </div>
      </div>
    </section>
  </div>
</div>
<script>
window.__MBA_DATA = {payload};
(function () {{
  const DATA = window.__MBA_DATA;
  const COLORS = {{
    primary: '#d04a02',
    secondary: '#c0392b',
    navy: '#2b2b2b',
    teal: '#1580b0',
    green: '#1a7f5c',
    muted: ['#d04a02','#c0392b','#1580b0','#2b2b2b','#5c6bc0','#1a7f5c','#707070'],
  }};
  let charts = {{}};
  const PAGE_SIZE = 40;
  let page = 0;
  let tableRows = [];

  function uniq(col) {{
    const s = new Set();
    DATA.forEach(r => {{ if (r[col] != null && r[col] !== '') s.add(String(r[col])); }});
    return Array.from(s).sort();
  }}

  function fillSelect(el, values, labels) {{
    el.innerHTML = '<option value=\"all\">All</option>' +
      values.map((v, i) => '<option value=\"' + String(v).replace(/\"/g,'&quot;') + '\">' + (labels ? labels[i] : v) + '</option>').join('');
  }}

  function initFilters() {{
    const years = uniq('Graduation Year').sort((a,b) => Number(a)-Number(b));
    const schools = uniq('School');
    const ind = uniq('Post-MBA Industry');
    const prog = uniq('Program Type');
    fillSelect(document.getElementById('f-year'), years);
    fillSelect(document.getElementById('f-school'), schools);
    fillSelect(document.getElementById('f-industry'), ind);
    fillSelect(document.getElementById('f-program'), prog);
  }}

  function filtered() {{
    const y = document.getElementById('f-year').value;
    const s = document.getElementById('f-school').value;
    const i = document.getElementById('f-industry').value;
    const p = document.getElementById('f-program').value;
    return DATA.filter(r => {{
      if (y !== 'all' && String(r['Graduation Year']) !== y) return false;
      if (s !== 'all' && r['School'] !== s) return false;
      if (i !== 'all' && r['Post-MBA Industry'] !== i) return false;
      if (p !== 'all' && r['Program Type'] !== p) return false;
      return true;
    }});
  }}

  function fmtMoney(n) {{
    if (n == null || n === '') return '—';
    const x = Number(n);
    if (!isFinite(x)) return '—';
    return '$' + Math.round(x).toLocaleString();
  }}

  function pctYes(rows, col) {{
    if (!rows.length) return 0;
    const yes = rows.filter(r => String(r[col]).toLowerCase() === 'yes').length;
    return Math.round((yes / rows.length) * 1000) / 10;
  }}

  function mean(nums) {{
    const a = nums.filter(x => x != null && isFinite(Number(x))).map(Number);
    if (!a.length) return null;
    return a.reduce((s,x)=>s+x,0)/a.length;
  }}

  function renderKPI(rows) {{
    const n = rows.length;
    const avgSal = mean(rows.map(r => r['Post-MBA Salary']));
    const avgG = mean(rows.map(r => r['GMAT']));
    const pTech = pctYes(rows, 'Offer in Big Tech');
    const strip = document.getElementById('kpi-strip');
    strip.innerHTML =
      '<div class=\"kpi\"><div class=\"klabel\">Profiles</div><div class=\"kval\">' + n.toLocaleString() + '</div><div class=\"ksub\">in filtered cohort</div></div>' +
      '<div class=\"kpi\"><div class=\"klabel\">Avg post-MBA salary</div><div class=\"kval\">' + (avgSal != null ? fmtMoney(avgSal) : '—') + '</div><div class=\"ksub\">mean compensation</div></div>' +
      '<div class=\"kpi\"><div class=\"klabel\">Avg GMAT</div><div class=\"kval\">' + (avgG != null ? Math.round(avgG) : '—') + '</div><div class=\"ksub\">exam score</div></div>' +
      '<div class=\"kpi\"><div class=\"klabel\">Big Tech offer</div><div class=\"kval\">' + pTech + '%</div><div class=\"ksub\">share answering Yes</div></div>';
  }}

  function destroyChart(id) {{
    if (charts[id]) {{ charts[id].destroy(); delete charts[id]; }}
  }}

  function doughnutPalette(n) {{
    const base = COLORS.muted;
    const out = [];
    for (let i = 0; i < n; i++) out.push(base[i % base.length]);
    return out;
  }}

  function countBy(rows, col) {{
    const m = {{}};
    rows.forEach(r => {{
      const k = r[col] == null ? 'Unknown' : String(r[col]);
      m[k] = (m[k] || 0) + 1;
    }});
    return m;
  }}

  function renderOverviewCharts(rows) {{
    destroyChart('industry');
    destroyChart('school');
    destroyChart('year');
    destroyChart('scatter');
    const ci = document.getElementById('chart-industry');
    const cs = document.getElementById('chart-school');
    const cy = document.getElementById('chart-year');
    const cz = document.getElementById('chart-scatter');
    if (!rows.length) {{
      charts.industry = new Chart(ci, {{
        type: 'bar',
        data: {{ labels: ['No matching rows'], datasets: [{{ data: [0], backgroundColor: '#e8e8e8' }}] }},
        options: {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }},
          scales: {{ x: {{ display: false }}, y: {{ ticks: {{ color: '#707070' }} }} }} }}
      }});
      charts.school = new Chart(cs, {{
        type: 'doughnut',
        data: {{ labels: ['—'], datasets: [{{ data: [1], backgroundColor: ['#e8e8e8'] }}] }},
        options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }} }}
      }});
      charts.year = new Chart(cy, {{
        type: 'bar',
        data: {{ labels: ['—'], datasets: [{{ data: [0], backgroundColor: '#e8e8e8' }}] }},
        options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ display: false }}, x: {{ display: false }} }} }}
      }});
      charts.scatter = new Chart(cz, {{
        type: 'scatter',
        data: {{ datasets: [{{ label: '—', data: [], backgroundColor: '#ccc' }}] }},
        options: {{ responsive: true, maintainAspectRatio: false }}
      }});
      return;
    }}

    const byInd = countBy(rows, 'Post-MBA Industry');
    const labelsI = Object.keys(byInd).sort((a,b) => byInd[b]-byInd[a]);
    charts.industry = new Chart(ci, {{
      type: 'bar',
      data: {{
        labels: labelsI,
        datasets: [{{ label: 'Count', data: labelsI.map(k => byInd[k]), backgroundColor: labelsI.map((_,i) => doughnutPalette(labelsI.length)[i]), borderWidth: 0 }}]
      }},
      options: {{
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ color: '#707070' }}, grid: {{ color: 'rgba(43,43,43,0.08)' }} }},
          y: {{ ticks: {{ color: '#2b2b2b' }}, grid: {{ display: false }} }}
        }}
      }}
    }});

    const bySch = countBy(rows, 'School');
    const labelsS = Object.keys(bySch);
    charts.school = new Chart(cs, {{
      type: 'doughnut',
      data: {{
        labels: labelsS,
        datasets: [{{ data: labelsS.map(k => bySch[k]), backgroundColor: doughnutPalette(labelsS.length), borderWidth: 1, borderColor: '#fff' }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'right', labels: {{ color: '#2b2b2b', boxWidth: 12 }} }} }}
      }}
    }});

    const byY = countBy(rows, 'Graduation Year');
    const labelsY = Object.keys(byY).sort((a,b)=>Number(a)-Number(b));
    charts.year = new Chart(cy, {{
      type: 'bar',
      data: {{
        labels: labelsY,
        datasets: [{{ label: 'Headcount', data: labelsY.map(k => byY[k]), backgroundColor: 'rgba(208,74,2,0.55)', borderColor: '#d04a02', borderWidth: 1 }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          y: {{ beginAtZero: true, ticks: {{ color: '#707070' }}, grid: {{ color: 'rgba(43,43,43,0.08)' }} }},
          x: {{ ticks: {{ color: '#2b2b2b' }}, grid: {{ display: false }} }}
        }}
      }}
    }});

    const sample = rows.filter(r => r['GPA'] != null && r['Post-MBA Salary'] != null).slice(0, 400);
    charts.scatter = new Chart(cz, {{
      type: 'scatter',
      data: {{
        datasets: [{{
          label: 'Candidates',
          data: sample.map(r => ({{ x: Number(r['GPA']), y: Number(r['Post-MBA Salary']) }})),
          backgroundColor: 'rgba(192,57,43,0.35)',
          borderColor: '#c0392b',
          pointRadius: 3,
          pointHoverRadius: 5
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ title: {{ display: true, text: 'GPA', color: '#707070' }}, ticks: {{ color: '#707070' }}, grid: {{ color: 'rgba(43,43,43,0.08)' }} }},
          y: {{ title: {{ display: true, text: 'Salary', color: '#707070' }}, ticks: {{ callback: v => '$' + Number(v)/1000 + 'k', color: '#707070' }}, grid: {{ color: 'rgba(43,43,43,0.08)' }} }}
        }}
      }}
    }});
  }}

  function renderOfferCharts(rows) {{
    destroyChart('offers');
    destroyChart('intern');
    if (!rows.length) {{
      charts.offers = new Chart(document.getElementById('chart-offers'), {{
        type: 'bar',
        data: {{ labels: ['—'], datasets: [{{ data: [0], backgroundColor: '#e8e8e8' }}] }},
        options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ display: false }}, x: {{ display: false }} }} }}
      }});
      charts.intern = new Chart(document.getElementById('chart-intern'), {{
        type: 'pie',
        data: {{ labels: ['—'], datasets: [{{ data: [1], backgroundColor: ['#e8e8e8'] }}] }},
        options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }} }}
      }});
      return;
    }}
    const yes = (c) => pctYes(rows, c);
    const labels = ['Big Tech', 'Consulting', 'Big Banks'];
    const vals = [
      yes('Offer in Big Tech'),
      yes('Offer in Consulting'),
      yes('Offer in Big Banks')
    ];
    charts.offers = new Chart(document.getElementById('chart-offers'), {{
      type: 'bar',
      data: {{
        labels,
        datasets: [{{ label: '% Yes', data: vals, backgroundColor: ['rgba(208,74,2,0.75)','rgba(192,57,43,0.75)','rgba(21,128,176,0.75)'], borderWidth: 0 }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          y: {{ max: 100, beginAtZero: true, ticks: {{ callback: v => v + '%', color: '#707070' }}, grid: {{ color: 'rgba(43,43,43,0.08)' }} }},
          x: {{ ticks: {{ color: '#2b2b2b' }}, grid: {{ display: false }} }}
        }}
      }}
    }});

    const intern = countBy(rows, 'Internship Completed');
    const ik = Object.keys(intern);
    charts.intern = new Chart(document.getElementById('chart-intern'), {{
      type: 'pie',
      data: {{
        labels: ik,
        datasets: [{{ data: ik.map(k => intern[k]), backgroundColor: ['#1580b0','#d04a02','#707070'], borderWidth: 1, borderColor: '#fff' }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'right', labels: {{ color: '#2b2b2b' }} }} }}
      }}
    }});
  }}

  function renderBench(rows) {{
    destroyChart('sal-school');
    destroyChart('gmat-school');
    const snap = document.getElementById('bench-snap');
    if (!rows.length) {{
      snap.innerHTML = '<div class="snap-item"><span class="snap-label">Cohort</span><span class="snap-val">No rows</span></div>';
      charts['sal-school'] = new Chart(document.getElementById('chart-sal-school'), {{
        type: 'bar',
        data: {{ labels: ['—'], datasets: [{{ data: [0], backgroundColor: '#e8e8e8' }}] }},
        options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ display: false }}, x: {{ display: false }} }} }}
      }});
      charts['gmat-school'] = new Chart(document.getElementById('chart-gmat-school'), {{
        type: 'bar',
        data: {{ labels: ['—'], datasets: [{{ data: [0], backgroundColor: '#e8e8e8' }}] }},
        options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ display: false }}, x: {{ display: false }} }} }}
      }});
      return;
    }}
    const schools = [...new Set(rows.map(r => r['School']).filter(Boolean))].sort();
    const avgBy = (col) => {{
      const m = {{}};
      schools.forEach(sch => {{
        const sub = rows.filter(r => r['School'] === sch);
        const v = mean(sub.map(r => r[col]));
        m[sch] = v;
      }});
      return m;
    }};
    const sal = avgBy('Post-MBA Salary');
    const gm = avgBy('GMAT');
    charts['sal-school'] = new Chart(document.getElementById('chart-sal-school'), {{
      type: 'bar',
      data: {{
        labels: schools,
        datasets: [{{ label: 'Avg salary', data: schools.map(s => sal[s]), backgroundColor: 'rgba(34,39,94,0.75)', borderWidth: 0 }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{ callbacks: {{ label: c => fmtMoney(c.parsed.y) }} }}
        }},
        scales: {{
          y: {{ ticks: {{ callback: v => '$' + Number(v)/1000 + 'k', color: '#707070' }}, grid: {{ color: 'rgba(43,43,43,0.08)' }} }},
          x: {{ ticks: {{ color: '#2b2b2b' }}, grid: {{ display: false }} }}
        }}
      }}
    }});
    charts['gmat-school'] = new Chart(document.getElementById('chart-gmat-school'), {{
      type: 'bar',
      data: {{
        labels: schools,
        datasets: [{{ label: 'Avg GMAT', data: schools.map(s => gm[s]), backgroundColor: 'rgba(208,74,2,0.55)', borderColor: '#d04a02', borderWidth: 1 }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          y: {{ beginAtZero: false, ticks: {{ color: '#707070' }}, grid: {{ color: 'rgba(43,43,43,0.08)' }} }},
          x: {{ ticks: {{ color: '#2b2b2b' }}, grid: {{ display: false }} }}
        }}
      }}
    }});

    const ft = pctYes(rows, 'Offer in Big Tech');
    const fc = pctYes(rows, 'Offer in Consulting');
    snap.innerHTML =
      '<div class=\"snap-item\"><span class=\"snap-label\">Avg GPA</span><span class=\"snap-val\">' + (mean(rows.map(r=>r['GPA']))!=null ? mean(rows.map(r=>r['GPA'])).toFixed(2) : '—') + '</span></div>' +
      '<div class=\"snap-item\"><span class=\"snap-label\">Avg work yrs (pre)</span><span class=\"snap-val\">' + (mean(rows.map(r=>r['Work Experience (Years)']))!=null ? mean(rows.map(r=>r['Work Experience (Years)'])).toFixed(1) : '—') + '</span></div>' +
      '<div class=\"snap-item\"><span class=\"snap-label\">% Big Tech offer</span><span class=\"snap-val\">' + ft + '%</span></div>' +
      '<div class=\"snap-item\"><span class=\"snap-label\">% Consulting offer</span><span class=\"snap-val\">' + fc + '%</span></div>';
  }}

  const TABLE_COLS = [
    'Student ID','School','Concentration','GPA','GMAT','Graduation Year','Program Type',
    'Post-MBA Salary','Post-MBA Industry','Job Function','Job Location',
    'Offer in Big Tech','Offer in Consulting','Offer in Big Banks'
  ];

  function renderTable() {{
    const q = (document.getElementById('table-search').value || '').trim().toLowerCase();
    const rows = filtered().filter(r => {{
      if (!q) return true;
      return TABLE_COLS.some(c => String(r[c] ?? '').toLowerCase().includes(q));
    }});
    tableRows = rows;
    page = Math.min(page, Math.max(0, Math.ceil(rows.length / PAGE_SIZE) - 1));
    const thead = document.getElementById('data-thead');
    thead.innerHTML = TABLE_COLS.map(c => '<th>' + c + '</th>').join('');
    const tbody = document.getElementById('data-tbody');
    const start = page * PAGE_SIZE;
    const slice = rows.slice(start, start + PAGE_SIZE);
    tbody.innerHTML = slice.map(r => '<tr>' + TABLE_COLS.map(c => {{
      let v = r[c];
      if (c === 'Post-MBA Salary') v = fmtMoney(v);
      return '<td>' + (v == null ? '' : String(v)) + '</td>';
    }}).join('') + '</tr>').join('');
    document.getElementById('table-count').textContent = rows.length.toLocaleString() + ' rows';
    document.getElementById('page-info').textContent = rows.length ? ('Page ' + (page + 1) + ' of ' + Math.ceil(rows.length / PAGE_SIZE)) : 'No rows';
    document.getElementById('page-prev').disabled = page <= 0;
    document.getElementById('page-next').disabled = start + PAGE_SIZE >= rows.length;
  }}

  function refresh() {{
    const rows = filtered();
    renderKPI(rows);
    renderOverviewCharts(rows);
    renderOfferCharts(rows);
    renderBench(rows);
    renderTable();
  }}

  document.querySelectorAll('nav.tabs button').forEach(btn => {{
    btn.addEventListener('click', () => {{
      document.querySelectorAll('nav.tabs button').forEach(b => {{ b.classList.remove('active'); b.setAttribute('aria-selected','false'); }});
      btn.classList.add('active');
      btn.setAttribute('aria-selected','true');
      const id = btn.getAttribute('data-tab');
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === id));
    }});
  }});

  ['f-year','f-school','f-industry','f-program'].forEach(id => {{
    document.getElementById(id).addEventListener('change', () => {{ page = 0; refresh(); }});
  }});
  document.getElementById('btn-reset').addEventListener('click', () => {{
    ['f-year','f-school','f-industry','f-program'].forEach(id => {{ document.getElementById(id).value = 'all'; }});
    page = 0;
    refresh();
  }});
  document.getElementById('table-search').addEventListener('input', () => {{ page = 0; renderTable(); }});
  document.getElementById('page-prev').addEventListener('click', () => {{ page = Math.max(0, page - 1); renderTable(); }});
  document.getElementById('page-next').addEventListener('click', () => {{ page++; renderTable(); }});

  initFilters();
  refresh();
}})();
</script>
</body>
</html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
