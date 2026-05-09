const fs = require('fs');
const content = fs.readFileSync('/Users/edwinbrown1/.gemini/antigravity/scratch/Edwin-Analysis-Power/app.py', 'utf8');
console.log("Critique present:", content.includes('resume_critique'));
console.log("Chat interface present:", content.includes('st.chat_message'));
