const fs = require('fs');
const html = fs.readFileSync('/Users/edwinbrown1/.gemini/antigravity/scratch/Edwin-Analysis-Power/Candidate_Dashboard_Enhanced.html', 'utf-8');

// Mock DOM
global.window = {};

// We need to extract the dataset from HTML
const dataMatch = html.match(/window\.__MBA_DATA\s*=\s*(\[.*\]);/);
if (!dataMatch) {
  console.log("Could not find MBA data");
  process.exit(1);
}
window.__MBA_DATA = JSON.parse(dataMatch[1]);

// Extract engine js
const engineMatch = html.match(/\/\/ === Benchmark Engine ===\n([\s\S]*?)\n\/\/ === Benchmark UI ===/);
if (!engineMatch) {
  console.log("Could not find engine code");
  process.exit(1);
}
eval(engineMatch[1]);

const BE = window.BenchmarkEngine;

const testProfile = {
  "GPA": 3.4,
  "GMAT": 710,
  "Concentration": "Finance",
  "Pre-MBA Industry": "Tech",
  "Military Veteran": "Yes"
};

const result = BE.runBenchmark(testProfile, "Big Tech");
console.log("Top Recommendation:", result.recommendation);
console.log("Top Prob:", result.results[result.recommendation].probability);
console.log("Expected Value:", result.results[result.recommendation].expectedValue);
console.log("GPA Percentile:", result.percentiles['GPA']);
