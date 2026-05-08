"""Inject Tab 5 (Candidate Benchmark) into the existing Candidate_Dashboard.html."""
from pathlib import Path

BASE = Path(__file__).resolve().parent
HTML_IN = BASE / "Candidate_Dashboard.html"
ENGINE_JS = BASE / "benchmark_engine.js"
UI_JS = BASE / "benchmark_ui.js"
OUT = BASE / "Candidate_Dashboard_Enhanced.html"

TAB_BUTTON = '<button type="button" role="tab" aria-selected="false" data-tab="t5">🎯 Candidate benchmark</button>'

TAB_CSS = """
  /* --- Tab 5: Candidate Benchmark --- */
  .profile-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-bottom:14px}
  .profile-field{display:flex;flex-direction:column;gap:4px}
  .field-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;font-weight:500}
  .field-input,.field-select{padding:8px 10px;border:1px solid var(--line-2);border-radius:6px;font-family:inherit;font-size:13px;background:#fff;color:var(--cream)}
  .field-input:focus,.field-select:focus{outline:none;border-color:var(--bc-blue);box-shadow:0 0 0 2px rgba(208,74,2,.15)}
  .resume-upload-row{display:flex;gap:10px;align-items:flex-end;flex-wrap:wrap;margin-bottom:14px;padding:12px;background:rgba(208,74,2,.04);border:1px dashed rgba(208,74,2,.3);border-radius:10px}
  .resume-upload-row .field-label{margin-bottom:2px}
  .scanning{color:var(--gold);font-weight:500}
  .btn-benchmark{background:linear-gradient(135deg,#d04a02,#c0392b);color:#fff;border:0;padding:10px 28px;font-size:13px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;border-radius:8px;cursor:pointer;font-family:inherit;transition:filter .15s}
  .btn-benchmark:hover{filter:brightness(1.1)}
  .btn-secondary{background:#fff;color:var(--cream);border:1px solid var(--line-2);padding:10px 20px;font-size:12px;font-weight:500;border-radius:8px;cursor:pointer;font-family:inherit}
  .btn-secondary:hover{border-color:var(--bc-blue)}
  /* Data Quality */
  .dq-header{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px}
  .dq-score{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:500;color:var(--cream)}
  .dq-pct{font-size:12px;color:var(--muted)}
  .dq-bar-wrap{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-bottom:10px}
  .dq-bar{height:100%;background:linear-gradient(90deg,var(--bc-blue),var(--green));border-radius:3px;transition:width .4s}
  .dq-list{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:4px}
  .dq-item{font-size:11px;padding:4px 6px;border-radius:4px}
  .dq-item.missing{background:rgba(192,57,43,.06)}
  .dq-field{font-weight:500;color:var(--cream)}
  .dq-val{color:var(--green);font-family:'JetBrains Mono',monospace;font-size:10px}
  .dq-miss{color:var(--red);font-style:italic}
  /* Recommendation */
  .rec-banner{display:flex;gap:14px;align-items:center;padding:16px 20px;background:linear-gradient(135deg,#22275e,#161a4a);border-radius:var(--ion-radius);margin-bottom:14px;color:#fff}
  .rec-icon{font-size:28px}
  .rec-headline{font-family:'Fraunces',serif;font-size:clamp(1rem,1.8vw,1.3rem);letter-spacing:-.02em}
  .rec-sub{font-size:11px;color:rgba(247,245,244,.75);margin-top:4px;line-height:1.4}
  /* School Cards */
  .school-cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}
  @media(max-width:960px){.school-cards{grid-template-columns:repeat(2,1fr)}}
  .school-card{background:linear-gradient(180deg,#fff,#fafcfd);border:1px solid var(--line-2);border-radius:var(--ion-radius);padding:14px;box-shadow:var(--ion-glow);text-align:center;transition:transform .15s}
  .school-card:hover{transform:translateY(-2px)}
  .school-card.top-pick{border-color:var(--bc-blue);box-shadow:0 0 0 1px rgba(208,74,2,.2),0 8px 24px rgba(208,74,2,.12)}
  .sc-rank{font-size:11px;color:var(--muted);margin-bottom:4px}
  .sc-name{font-family:'Outfit',sans-serif;font-weight:600;font-size:clamp(.95rem,1.3vw,1.1rem);color:var(--cream);margin-bottom:8px}
  .sc-prob{font-family:'Fraunces',serif;font-size:clamp(1.6rem,3vw,2.2rem);color:var(--bc-blue);font-weight:500;line-height:1}
  .sc-prob-label{font-size:10px;color:var(--muted);margin:4px 0 10px;text-transform:uppercase;letter-spacing:.08em}
  .sc-details{border-top:1px solid var(--line);padding-top:8px}
  .sc-detail{display:flex;justify-content:space-between;font-size:11px;padding:3px 0}
  .sc-dl{color:var(--muted)}.sc-dv{font-weight:500;color:var(--cream);font-family:'JetBrains Mono',monospace}
  /* Percentile Bars */
  .percentile-section{margin-top:14px;padding:14px;background:#fff;border:1px solid var(--line-2);border-radius:var(--ion-radius);box-shadow:var(--ion-glow)}
  .pct-bars{margin-top:10px}
  .pct-bar-row{display:flex;align-items:center;gap:10px;margin-bottom:8px}
  .pct-label{font-size:11px;color:var(--muted);width:140px;flex-shrink:0;text-align:right}
  .pct-track{flex:1;height:10px;background:var(--line);border-radius:5px;overflow:hidden}
  .pct-fill{height:100%;border-radius:5px;transition:width .5s}
  .pct-val{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cream);width:40px;font-weight:500}
  .target-row{display:flex;gap:10px;align-items:flex-end;margin-bottom:14px;flex-wrap:wrap}
"""

TAB_HTML = """
    <section id="t5" class="tab-panel" role="tabpanel">
      <div class="tab-scroll">
        <div class="panel-grid">
          <div class="panel accent-left col-12">
            <div class="panel-head">
              <div>
                <div class="panel-title">Resume Upload (Optional)</div>
                <div class="panel-sub">Upload a PDF resume to auto-extract profile data via Gemini AI</div>
              </div>
              <span class="panel-tag">AI</span>
            </div>
            <div class="resume-upload-row">
              <div class="profile-field" style="flex:1;min-width:200px">
                <label class="field-label">Resume file (PDF / TXT)</label>
                <input type="file" id="resume-file" accept=".pdf,.txt,.docx" class="field-input" />
              </div>
              <div class="profile-field" style="flex:1;min-width:200px">
                <label class="field-label">Gemini API Key</label>
                <input type="password" id="gemini-key" class="field-input" placeholder="AIza..." />
              </div>
              <button type="button" class="btn-secondary" id="btn-scan-resume">Scan Resume</button>
            </div>
            <div id="resume-status" style="font-size:12px;color:var(--muted);margin-top:4px"></div>
          </div>

          <div class="panel col-12">
            <div class="panel-head">
              <div>
                <div class="panel-title">Your Profile</div>
                <div class="panel-sub">Enter your academic and professional data points</div>
              </div>
              <span class="panel-tag">input</span>
            </div>
            <div id="profile-form-area"></div>
          </div>

          <div class="panel col-12">
            <div class="panel-head">
              <div>
                <div class="panel-title">Data Quality Check</div>
                <div class="panel-sub">Fields matched against the MBA dataset schema</div>
              </div>
              <span class="panel-tag">quality</span>
            </div>
            <div id="data-quality-area"><p style="color:var(--muted);font-size:12px">Fill in profile fields above — quality check updates automatically.</p></div>
          </div>

          <div class="panel accent-left col-12">
            <div class="panel-head">
              <div>
                <div class="panel-title">Target & Benchmark</div>
                <div class="panel-sub">Select your target industry and run the analysis</div>
              </div>
              <span class="panel-tag">run</span>
            </div>
            <div class="target-row">
              <div class="profile-field">
                <label class="field-label">Target Industry</label>
                <select id="target-industry" class="field-select">
                  <option value="Big Tech">Big Tech</option>
                  <option value="Consulting">Consulting</option>
                  <option value="Big Banks">Big Banks</option>
                </select>
              </div>
              <button type="button" class="btn-benchmark" id="btn-run-benchmark">▶ Run School Benchmark</button>
              <button type="button" class="btn-secondary" id="btn-check-quality">Check Data Quality</button>
            </div>
          </div>

          <div class="panel col-12" id="benchmark-results">
            <p style="color:var(--muted);font-size:12px">Fill in your profile, select a target industry, and click "Run School Benchmark" to see recommendations.</p>
          </div>

          <div class="panel col-6">
            <div class="panel-head">
              <div>
                <div class="panel-title">School probability comparison</div>
                <div class="panel-sub">Offer probability by school for your profile</div>
              </div>
              <span class="panel-tag">%</span>
            </div>
            <div class="chart-box"><canvas id="chart-prob"></canvas></div>
          </div>
          <div class="panel col-6">
            <div class="panel-head">
              <div>
                <div class="panel-title">Profile vs recommended school</div>
                <div class="panel-sub">Your metrics vs school averages</div>
              </div>
              <span class="panel-tag">radar</span>
            </div>
            <div class="chart-box"><canvas id="chart-radar"></canvas></div>
          </div>
        </div>
      </div>
    </section>
"""


def main():
    html = HTML_IN.read_text(encoding="utf-8")

    # 1. Add tab button
    html = html.replace(
        'data-tab="t4">Directory</button>',
        f'data-tab="t4">Directory</button>\n    {TAB_BUTTON}',
    )

    # 2. Inject CSS before </style>
    html = html.replace("</style>", TAB_CSS + "\n</style>")

    # 3. Inject tab panel HTML before closing </div> of dashboard-body
    # Find the last </section> (t4) and add after it
    last_section_end = html.rfind("</section>")
    if last_section_end > 0:
        insert_pos = last_section_end + len("</section>")
        html = html[:insert_pos] + TAB_HTML + html[insert_pos:]

    # 4. Inject JS before </script>
    engine_js = ENGINE_JS.read_text(encoding="utf-8")
    ui_js = UI_JS.read_text(encoding="utf-8")

    # Add pdf.js CDN
    pdf_cdn = '<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>\n'
    html = html.replace("<script>", pdf_cdn + "<script>", 1)

    # Append benchmark JS before final </script>
    last_script_close = html.rfind("</script>")
    inject_js = f"\n// === Benchmark Engine ===\n{engine_js}\n// === Benchmark UI ===\n{ui_js}\n"
    html = html[:last_script_close] + inject_js + html[last_script_close:]

    OUT.write_text(html, encoding="utf-8")
    print(f"✅ Enhanced dashboard: {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
