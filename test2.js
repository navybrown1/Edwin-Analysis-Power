const fs = require('fs');
const code = fs.readFileSync('/Users/edwinbrown1/.gemini/antigravity/scratch/Edwin-Analysis-Power/benchmark_engine.js', 'utf8');
console.log(code.includes('expectedValue'));
