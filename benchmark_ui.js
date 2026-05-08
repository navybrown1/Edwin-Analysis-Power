// ============================================================
// Candidate Benchmark UI — Tab 5 interactions
// ============================================================
(function () {
  const BE = window.BenchmarkEngine;
  let candidateProfile = {};

  // --- Profile Form ---
  function buildProfileForm() {
    const container = document.getElementById('profile-form-area');
    if (!container) return;

    const categoricalOptions = {
      'Concentration': ['Finance', 'Accounting', 'Business Analytics', 'Information Systems'],
      'Program Type': ['Full-Time', 'Part-Time'],
      'Pre-MBA Industry': ['Tech', 'Finance', 'Consulting', 'Other'],
      'Internship Completed': ['Yes', 'No'],
      'Internship Employer Type': ['Big Tech', 'Big Bank', 'Consulting', 'Other'],
      'Leadership Roles': ['Yes', 'No'],
      'International Experience': ['Yes', 'No'],
    };
    const numericRanges = {
      'GPA': { min: 2.0, max: 4.0, step: 0.01, placeholder: '3.45' },
      'GMAT': { min: 400, max: 800, step: 1, placeholder: '710' },
      'Work Experience (Years)': { min: 0, max: 15, step: 0.5, placeholder: '5' },
      'Pre-MBA Salary': { min: 30000, max: 250000, step: 1000, placeholder: '95000' },
      'Club Involvements': { min: 0, max: 5, step: 1, placeholder: '2' },
      'Languages Spoken': { min: 1, max: 5, step: 1, placeholder: '2' },
    };

    let html = '<div class="profile-grid">';
    // Numeric fields
    for (const [field, range] of Object.entries(numericRanges)) {
      html += `
        <div class="profile-field">
          <label class="field-label">${field}</label>
          <input type="number" class="field-input" id="pf-${slugify(field)}"
            min="${range.min}" max="${range.max}" step="${range.step}"
            placeholder="${range.placeholder}" />
        </div>`;
    }
    // Categorical fields
    for (const [field, options] of Object.entries(categoricalOptions)) {
      html += `
        <div class="profile-field">
          <label class="field-label">${field}</label>
          <select class="field-select" id="pf-${slugify(field)}">
            <option value="">— select —</option>
            ${options.map(o => `<option value="${o}">${o}</option>`).join('')}
          </select>
        </div>`;
    }
    html += '</div>';
    container.innerHTML = html;
  }

  function slugify(s) { return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/-+$/, ''); }

  function collectProfile() {
    const profile = {};
    for (const f of BE.NUMERIC_FIELDS) {
      const el = document.getElementById('pf-' + slugify(f));
      if (el && el.value !== '') profile[f] = parseFloat(el.value);
    }
    for (const f of BE.CATEGORICAL_FIELDS) {
      const el = document.getElementById('pf-' + slugify(f));
      if (el && el.value !== '') profile[f] = el.value;
    }
    return profile;
  }

  function setProfileField(field, value) {
    const el = document.getElementById('pf-' + slugify(field));
    if (el && value != null && value !== '') {
      el.value = value;
    }
  }

  // --- Data Quality Checklist ---
  function renderDataQuality(profile) {
    const checks = BE.checkDataQuality(profile);
    const container = document.getElementById('data-quality-area');
    const filled = checks.filter(c => c.present).length;
    const total = checks.length;
    const pct = Math.round((filled / total) * 100);

    let html = `<div class="dq-header">
      <span class="dq-score">${filled}/${total} fields</span>
      <span class="dq-pct">${pct}% complete</span>
    </div>
    <div class="dq-bar-wrap"><div class="dq-bar" style="width:${pct}%"></div></div>
    <div class="dq-list">`;
    for (const c of checks) {
      const icon = c.present ? '✅' : '❌';
      const val = c.present ? `<span class="dq-val">${c.value}</span>` : '<span class="dq-miss">missing</span>';
      html += `<div class="dq-item ${c.present ? 'filled' : 'missing'}">${icon} <span class="dq-field">${c.field}</span> ${val}</div>`;
    }
    html += '</div>';
    container.innerHTML = html;
    return { filled, total, pct };
  }

  // --- Results Rendering ---
  function renderBenchmarkResults(result) {
    const container = document.getElementById('benchmark-results');
    if (!result) {
      container.innerHTML = '<p style="color:var(--muted)">Fill in your profile and click "Run Benchmark".</p>';
      return;
    }

    // School recommendation banner
    let html = `<div class="rec-banner">
      <div class="rec-icon">🎯</div>
      <div>
        <div class="rec-headline">Recommended School: <strong>${result.recommendation}</strong></div>
        <div class="rec-sub">For maximizing your ${result.targetIndustry} career path, ${result.recommendation} gives you the highest probability based on ${window.__MBA_DATA.length.toLocaleString()} graduate outcomes.</div>
      </div>
    </div>`;

    // School probability cards
    html += '<div class="school-cards">';
    for (let i = 0; i < result.ranked.length; i++) {
      const [school, data] = result.ranked[i];
      const isTop = i === 0;
      const medal = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : '▪️';
      html += `<div class="school-card ${isTop ? 'top-pick' : ''}">
        <div class="sc-rank">${medal} #${i + 1}</div>
        <div class="sc-name">${school}</div>
        <div class="sc-prob">${data.probability}%</div>
        <div class="sc-prob-label">probability of ${result.targetIndustry} offer</div>
        <div class="sc-details">
          <div class="sc-detail"><span class="sc-dl">Avg Salary</span><span class="sc-dv">${BE.fmtMoney(data.avgSalary)}</span></div>
          <div class="sc-detail"><span class="sc-dl">School Offer Rate</span><span class="sc-dv">${data.schoolOfferRate}%</span></div>
          <div class="sc-detail"><span class="sc-dl">Profile Match</span><span class="sc-dv">${data.avgSimilarity}%</span></div>
          <div class="sc-detail"><span class="sc-dl">Cohort Size</span><span class="sc-dv">${data.totalGrads}</span></div>
        </div>
      </div>`;
    }
    html += '</div>';

    // Percentile bars
    html += '<div class="percentile-section"><div class="panel-title">Your Percentile Rankings</div><div class="panel-sub">How you compare against the full MBA cohort</div><div class="pct-bars">';
    for (const [field, pct] of Object.entries(result.percentiles)) {
      const color = pct >= 75 ? 'var(--green)' : pct >= 50 ? 'var(--gold)' : pct >= 25 ? '#e67e22' : 'var(--red)';
      html += `<div class="pct-bar-row">
        <span class="pct-label">${field}</span>
        <div class="pct-track"><div class="pct-fill" style="width:${pct}%;background:${color}"></div></div>
        <span class="pct-val">${pct}th</span>
      </div>`;
    }
    html += '</div></div>';

    container.innerHTML = html;

    // Render probability chart
    renderProbabilityChart(result);
    renderRadarChart(result);
  }

  // --- Charts ---
  let probChart = null;
  let radarChart = null;

  function renderProbabilityChart(result) {
    if (probChart) { probChart.destroy(); probChart = null; }
    const canvas = document.getElementById('chart-prob');
    if (!canvas) return;
    const schools = result.ranked.map(r => r[0]);
    const probs = result.ranked.map(r => r[1].probability);
    const colors = result.ranked.map((_, i) =>
      i === 0 ? '#d04a02' : i === 1 ? '#c0392b' : i === 2 ? '#1580b0' : '#707070');

    probChart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: schools,
        datasets: [{ label: '% Probability', data: probs, backgroundColor: colors, borderWidth: 0 }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: c => c.parsed.y + '%' } },
        },
        scales: {
          y: { max: 100, beginAtZero: true, ticks: { callback: v => v + '%', color: '#707070' }, grid: { color: 'rgba(43,43,43,0.08)' } },
          x: { ticks: { color: '#2b2b2b' }, grid: { display: false } },
        },
      },
    });
  }

  function renderRadarChart(result) {
    if (radarChart) { radarChart.destroy(); radarChart = null; }
    const canvas = document.getElementById('chart-radar');
    if (!canvas || !result.ranked.length) return;

    const profile = collectProfile();
    const labels = ['GPA', 'GMAT', 'Work Exp', 'Clubs', 'Languages'];
    const fields = ['GPA', 'GMAT', 'Work Experience (Years)', 'Club Involvements', 'Languages Spoken'];
    const maxVals = [4.0, 800, 10, 5, 3];

    const normalize = (val, max) => val != null ? Math.min((Number(val) / max) * 100, 100) : 0;

    const datasets = [
      {
        label: 'Your Profile',
        data: fields.map((f, i) => normalize(profile[f], maxVals[i])),
        borderColor: '#d04a02', backgroundColor: 'rgba(208,74,2,0.15)', borderWidth: 2, pointRadius: 4,
      },
    ];
    const topSchool = result.ranked[0];
    if (topSchool) {
      datasets.push({
        label: topSchool[0] + ' Avg',
        data: [
          normalize(topSchool[1].avgGPA, 4.0),
          normalize(topSchool[1].avgGMAT, 800),
          normalize(topSchool[1].avgWorkYears, 10),
          normalize(BE.mean(window.__MBA_DATA.filter(r => r.School === topSchool[0]).map(r => r['Club Involvements'])), 5),
          normalize(BE.mean(window.__MBA_DATA.filter(r => r.School === topSchool[0]).map(r => r['Languages Spoken'])), 3),
        ],
        borderColor: '#1580b0', backgroundColor: 'rgba(21,128,176,0.1)', borderWidth: 2, borderDash: [5, 5], pointRadius: 3,
      });
    }

    radarChart = new Chart(canvas, {
      type: 'radar',
      data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: { r: { max: 100, min: 0, ticks: { display: false }, grid: { color: 'rgba(43,43,43,0.1)' }, pointLabels: { color: '#2b2b2b', font: { size: 11 } } } },
        plugins: { legend: { position: 'bottom', labels: { color: '#2b2b2b', boxWidth: 12, font: { size: 11 } } } },
      },
    });
  }

  // --- Resume Upload ---
  async function handleResumeUpload() {
    const fileInput = document.getElementById('resume-file');
    const apiKeyInput = document.getElementById('gemini-key');
    const statusEl = document.getElementById('resume-status');

    if (!fileInput.files.length) { statusEl.textContent = 'Please select a file.'; return; }
    const apiKey = apiKeyInput ? apiKeyInput.value.trim() : '';
    if (!apiKey) { statusEl.textContent = 'Please enter your Gemini API key.'; return; }

    statusEl.innerHTML = '<span class="scanning">⏳ Scanning resume with Gemini AI...</span>';
    const file = fileInput.files[0];

    try {
      let text = '';
      if (file.name.endsWith('.pdf') && window.pdfjsLib) {
        const arrayBuf = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuf }).promise;
        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const tc = await page.getTextContent();
          text += tc.items.map(item => item.str).join(' ') + '\n';
        }
      } else {
        text = await file.text();
      }

      if (!text.trim()) { statusEl.textContent = 'Could not extract text from file.'; return; }

      const extracted = await BE.extractResumeWithGemini(text, apiKey);
      statusEl.innerHTML = '✅ Resume scanned — fields pre-filled below. Review and correct any values.';

      // Pre-fill form
      for (const [field, value] of Object.entries(extracted)) {
        if (value != null) setProfileField(field, value);
      }
      // Update quality check
      renderDataQuality(collectProfile());
    } catch (err) {
      statusEl.innerHTML = `<span style="color:var(--red)">❌ Error: ${err.message}</span>`;
    }
  }

  // --- Event Binding ---
  function init() {
    buildProfileForm();

    const btnRun = document.getElementById('btn-run-benchmark');
    if (btnRun) {
      btnRun.addEventListener('click', () => {
        const profile = collectProfile();
        const target = document.getElementById('target-industry').value;
        const dq = renderDataQuality(profile);
        if (dq.filled < 4) {
          document.getElementById('benchmark-results').innerHTML =
            '<p style="color:var(--red)">Please fill in at least 4 profile fields to run the benchmark.</p>';
          return;
        }
        const result = BE.runBenchmark(profile, target);
        candidateProfile = profile;
        renderBenchmarkResults(result);
      });
    }

    const btnDQ = document.getElementById('btn-check-quality');
    if (btnDQ) {
      btnDQ.addEventListener('click', () => {
        renderDataQuality(collectProfile());
      });
    }

    const btnResume = document.getElementById('btn-scan-resume');
    if (btnResume) {
      btnResume.addEventListener('click', handleResumeUpload);
    }

    // Auto-run quality check on input change
    document.querySelectorAll('.field-input, .field-select').forEach(el => {
      el.addEventListener('change', () => renderDataQuality(collectProfile()));
    });
  }

  // Wait for DOM
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    setTimeout(init, 100);
  }
})();
