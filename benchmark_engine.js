// ============================================================
// Candidate Benchmark Engine — KNN Probability + Percentiles
// ============================================================
(function () {
  const DATA = window.__MBA_DATA;
  const OFFER_COLS = {
    'Big Tech': 'Offer in Big Tech',
    'Consulting': 'Offer in Consulting',
    'Big Banks': 'Offer in Big Banks',
  };

  const NUMERIC_FIELDS = ['GPA', 'GMAT', 'Work Experience (Years)', 'Pre-MBA Salary',
    'Club Involvements', 'Languages Spoken'];
  const CATEGORICAL_FIELDS = ['Concentration', 'Program Type', 'Pre-MBA Industry',
    'Internship Completed', 'Internship Employer Type', 'Leadership Roles',
    'International Experience'];
  const ALL_PROFILE_FIELDS = [...NUMERIC_FIELDS, ...CATEGORICAL_FIELDS];

  // --- Helpers ---
  function mean(arr) {
    const v = arr.filter(x => x != null && isFinite(Number(x))).map(Number);
    return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null;
  }
  function std(arr) {
    const v = arr.filter(x => x != null && isFinite(Number(x))).map(Number);
    if (v.length < 2) return 1;
    const m = v.reduce((s, x) => s + x, 0) / v.length;
    return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length) || 1;
  }
  function percentile(val, arr) {
    const v = arr.filter(x => x != null && isFinite(Number(x))).map(Number).sort((a, b) => a - b);
    if (!v.length) return 50;
    let count = 0;
    for (const x of v) { if (x <= Number(val)) count++; }
    return Math.round((count / v.length) * 100);
  }
  function fmtMoney(n) {
    if (n == null) return '—';
    return '$' + Math.round(Number(n)).toLocaleString();
  }

  // --- Similarity Scoring ---
  function computeSimilarity(profile, row) {
    let score = 0;
    let maxScore = 0;
    // Numeric: use gaussian-like proximity (closer = higher score)
    for (const f of NUMERIC_FIELDS) {
      if (profile[f] == null || row[f] == null) continue;
      const allVals = DATA.map(r => r[f]).filter(x => x != null);
      const s = std(allVals);
      const dist = Math.abs(Number(profile[f]) - Number(row[f])) / s;
      const sim = Math.exp(-0.5 * dist * dist); // Gaussian similarity
      const weight = (f === 'GPA' || f === 'GMAT') ? 2 : 1;
      score += sim * weight;
      maxScore += weight;
    }
    // Categorical: exact match = 1, else 0
    for (const f of CATEGORICAL_FIELDS) {
      if (profile[f] == null || row[f] == null) continue;
      const weight = (f === 'Concentration' || f === 'Internship Completed') ? 1.5 : 1;
      if (String(profile[f]).toLowerCase() === String(row[f]).toLowerCase()) {
        score += weight;
      }
      maxScore += weight;
    }
    return maxScore > 0 ? score / maxScore : 0;
  }

  // --- Core Benchmark ---
  function runBenchmark(profile, targetIndustry) {
    const offerCol = OFFER_COLS[targetIndustry];
    if (!offerCol) return null;

    const schools = ['Columbia', 'NYU', 'Baruch College', 'Fordham'];
    const results = {};

    for (const school of schools) {
      const schoolRows = DATA.filter(r => r['School'] === school);
      if (!schoolRows.length) {
        results[school] = { probability: 0, matchCount: 0, avgSalary: 0, details: {} };
        continue;
      }
      // Compute similarity for each graduate at this school
      const scored = schoolRows.map(r => ({
        row: r,
        sim: computeSimilarity(profile, r),
      }));
      scored.sort((a, b) => b.sim - a.sim);

      // Take top K neighbors (K = min(50, 30% of school's grads))
      const K = Math.min(50, Math.max(10, Math.round(schoolRows.length * 0.3)));
      const neighbors = scored.slice(0, K);

      // Probability = % of neighbors with offer = "Yes"
      const yesCount = neighbors.filter(n =>
        String(n.row[offerCol]).toLowerCase() === 'yes').length;
      const probability = Math.round((yesCount / neighbors.length) * 1000) / 10;

      // Avg post-MBA salary of neighbors
      const avgSalary = mean(neighbors.map(n => n.row['Post-MBA Salary']));
      // Avg similarity score
      const avgSim = mean(neighbors.map(n => n.sim));

      // School-wide stats
      const schoolOfferRate = Math.round(
        (schoolRows.filter(r => String(r[offerCol]).toLowerCase() === 'yes').length
          / schoolRows.length) * 1000) / 10;

      results[school] = {
        probability,
        matchCount: neighbors.length,
        avgSalary: avgSalary || 0,
        avgSimilarity: Math.round((avgSim || 0) * 100),
        schoolOfferRate,
        totalGrads: schoolRows.length,
        avgGPA: mean(schoolRows.map(r => r['GPA'])),
        avgGMAT: mean(schoolRows.map(r => r['GMAT'])),
        avgWorkYears: mean(schoolRows.map(r => r['Work Experience (Years)'])),
      };
    }

    // Rank schools
    const ranked = Object.entries(results)
      .sort((a, b) => b[1].probability - a[1].probability);
    const recommendation = ranked[0][0];

    // Percentiles against FULL cohort
    const percentiles = {};
    for (const f of NUMERIC_FIELDS) {
      if (profile[f] != null) {
        const allVals = DATA.map(r => r[f]);
        percentiles[f] = percentile(profile[f], allVals);
      }
    }

    return { results, ranked, recommendation, percentiles, targetIndustry };
  }

  // --- Data Quality Check ---
  function checkDataQuality(profile) {
    const checks = [];
    for (const f of ALL_PROFILE_FIELDS) {
      const val = profile[f];
      const present = val != null && val !== '' && val !== undefined;
      checks.push({ field: f, present, value: present ? val : null });
    }
    return checks;
  }

  // --- Resume Extraction via Gemini ---
  async function extractResumeWithGemini(text, apiKey) {
    const prompt = `You are an expert resume parser. Extract the following data points from this resume text. Return ONLY valid JSON with these exact keys (use null if not found):

{
  "GPA": number or null,
  "GMAT": number or null,
  "Work Experience (Years)": number or null,
  "Pre-MBA Salary": number or null,
  "Concentration": "Finance" | "Accounting" | "Business Analytics" | "Information Systems" | null,
  "Program Type": "Full-Time" | "Part-Time" | null,
  "Pre-MBA Industry": "Tech" | "Finance" | "Consulting" | "Other" | null,
  "Internship Completed": "Yes" | "No" | null,
  "Internship Employer Type": "Big Tech" | "Big Bank" | "Consulting" | "Other" | null,
  "Leadership Roles": "Yes" | "No" | null,
  "Club Involvements": number (0-5) or null,
  "International Experience": "Yes" | "No" | null,
  "Languages Spoken": number (1-3) or null
}

RESUME TEXT:
${text}`;

    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.1, maxOutputTokens: 800 },
      }),
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error.message || 'Gemini API error');
    const raw = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
    // Extract JSON from response
    const jsonMatch = raw.match(/\{[\s\S]*\}/);
    if (!jsonMatch) throw new Error('Could not parse Gemini response');
    return JSON.parse(jsonMatch[0]);
  }

  // --- Expose API ---
  window.BenchmarkEngine = {
    runBenchmark,
    checkDataQuality,
    extractResumeWithGemini,
    percentile,
    mean,
    fmtMoney,
    NUMERIC_FIELDS,
    CATEGORICAL_FIELDS,
    ALL_PROFILE_FIELDS,
    OFFER_COLS,
  };
})();
