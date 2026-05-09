import re

with open("/Users/edwinbrown1/.gemini/antigravity/scratch/Edwin-Analysis-Power/app.py", "r") as f:
    content = f.read()

# 1. Update the prompt
old_prompt = '''                    prompt = f"""You are an expert resume parser. Extract the following data points from this resume text. Return ONLY valid JSON with these exact keys (use null if not found):
{{
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
  "Languages Spoken": number (1-3) or null,
  "Military Veteran": "Yes" | "No" | null
}}

RESUME TEXT:
{text}"""'''

new_prompt = '''                    prompt = f"""You are an expert resume parser and admissions consultant. Extract data from the resume and also provide a brief critique for business school admissions. Return ONLY valid JSON with exactly two top-level keys: "profile" and "critique".

The "profile" object must have these exact keys (use null if not found):
{{
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
  "Languages Spoken": number (1-3) or null,
  "Military Veteran": "Yes" | "No" | null
}}

The "critique" object must have these exact keys:
{{
  "score": number from 1 to 10 evaluating the resume's impact and leadership,
  "strengths": [array of 1-2 strong points],
  "weaknesses": [array of 1-2 areas to improve for an MBA application]
}}

RESUME TEXT:
{text}"""'''

content = content.replace(old_prompt, new_prompt)

# 2. Update the parsing and display logic
old_parse = '''                    import re
                    json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    if json_match:
                        st.session_state.candidate_profile = json.loads(json_match.group(0))
                        st.success("Resume parsed successfully!")
                    else:
                        st.error("Could not parse JSON from Gemini response.")
                except Exception as e:
                    st.error(f"Error parsing resume: {str(e)}")'''

new_parse = '''                    import re
                    json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                        st.session_state.candidate_profile = parsed.get("profile", {})
                        st.session_state.resume_critique = parsed.get("critique", {})
                        st.success("Resume parsed successfully!")
                    else:
                        st.error("Could not parse JSON from Gemini response.")
                except Exception as e:
                    st.error(f"Error parsing resume: {str(e)}")

        if "resume_critique" in st.session_state and st.session_state.resume_critique:
            critique = st.session_state.resume_critique
            with st.expander("🤖 AI Resume Critique", expanded=True):
                st.write(f"**Impact Score:** {critique.get('score', '?')}/10")
                st.write("**Strengths:**")
                for s in critique.get('strengths', []): st.write(f"- {s}")
                st.write("**Areas for Improvement:**")
                for w in critique.get('weaknesses', []): st.write(f"- {w}")'''

content = content.replace(old_parse, new_parse)

# 3. Add Chat Assistant at the bottom
chat_ui = '''
    # --- AI Chat Assistant ---
    st.markdown("---")
    st.markdown("### 💬 Ask the AI Admissions Consultant")
    st.caption("Ask questions about your profile, strategies to improve your chances, or details about the MBA landscape.")
    
    if "consultant_messages" not in st.session_state:
        st.session_state.consultant_messages = []
        
    for msg in st.session_state.consultant_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if user_query := st.chat_input("E.g., How can I compensate for a low GPA?"):
        st.session_state.consultant_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            if not gemini_key:
                st.error("Please provide a Gemini API Key in the sidebar.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=gemini_key)
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        
                        context = "You are an expert MBA Admissions Consultant. Be concise, direct, and encouraging."
                        if "candidate_profile" in st.session_state:
                            context += f"\\n\\nThe user's current profile is: {json.dumps(st.session_state.candidate_profile)}"
                            
                        # Build history
                        history = [{"role": "user", "parts": [context]}, {"role": "model", "parts": ["Understood. I am ready to advise."]}]
                        for m in st.session_state.consultant_messages[:-1]: # exclude latest
                            role = "user" if m["role"] == "user" else "model"
                            history.append({"role": role, "parts": [m["content"]]})
                            
                        chat = model.start_chat(history=history)
                        response = chat.send_message(user_query)
                        st.markdown(response.text)
                        st.session_state.consultant_messages.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    st.stop()
'''

content = content.replace("    st.stop()", chat_ui)

with open("/Users/edwinbrown1/.gemini/antigravity/scratch/Edwin-Analysis-Power/app.py", "w") as f:
    f.write(content)
