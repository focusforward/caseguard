from openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import json

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Session counter
if "total_cases" not in st.session_state:
    st.session_state.total_cases = 0

st.set_page_config(page_title="Clinical Defence Note Generator")

st.title("Clinical Defence Note Generator")
st.caption("Assistive documentation review tool. Clinical decisions remain with treating physician.")

st.info(f"Cases reviewed this session: {st.session_state.total_cases}")

note = st.text_area("Paste Emergency Case Note", height=300)

# ===== SYSTEM PROMPT =====
system_prompt = """
You are a medico-legal documentation auditor assisting doctors.

Your role is NOT to judge medical correctness.
Your role is to evaluate whether the written note can be defended if outcome becomes adverse.

You must follow a strict 5-anchor review:

1) Danger assessment documented
2) Risk context documented
3) Discharge reasoning documented
4) Safety-net advice documented
5) Objective data documented

Count how many anchors are missing.

Important interpretation rule:
Clinical notes are often brief. If a doctor documents absence of a concerning feature
(e.g., "no breathlessness", "active child", "walking normally", "pain reproducible"),
this counts as partial evidence that danger assessment or discharge reasoning was performed.
Do not require perfect wording. Credit reasonable implied clinical thinking.
Only mark an anchor missing if the note gives no indication it was considered.

Risk weighting guidance:
High-risk presentations include chest pain, breathlessness, syncope, focal weakness,
persistent vomiting, head injury, high fever in child, or altered sensorium.

If objective data OR danger assessment is missing in a high-risk presentation,
classification should usually be DANGEROUS.

If only safety-net advice or discharge reasoning is missing,
classification should usually be BORDERLINE.

Low-risk reassurance override:
For common benign presentations, reassuring behaviour is strong evidence of safety.

If a child with fever is described as playful, active, feeding well, or passing urine,
this OVERRIDES missing vitals or formal discharge wording.
Such cases should be classified SAFE unless a specific danger feature is documented.

Do not downgrade to BORDERLINE only because safety-net advice or vitals are not written
when reassuring behaviour is clearly present.

Classification rules:
0–1 missing → SAFE
2–3 missing → BORDERLINE
4–5 missing → DANGEROUS

Return STRICT JSON:
Tone guidance:
Write like a supportive senior colleague, not an auditor.
Avoid accusatory phrases like "lacks", "missing", "inadequate", or "high-risk documentation failure".
Explain gently what is not shown rather than blaming what was not done.
The goal is to help improve chart wording, not critique the doctor.
Keep reasoning calm, brief, and practical.
Defensible note generation:
The "defensible_note" must be a rewritten version of the ORIGINAL note.
It should read like a finished clinical entry ready to paste into the chart.

Rules:
- Keep all original facts
- Add brief clinical reasoning using neutral wording
- Add simple safety-net advice
- Do NOT give suggestions or explanations
- Do NOT say "consider documenting" or "include"
- Write only the final chart sentence(s), not instructions
- Keep it concise (2–4 lines maximum)
-Never invent examination findings, vitals, or investigations that were not stated or clearly implied in the original note.
-If information is absent, phrase it neutrally (e.g., "no concerning features reported") rather than creating new clinical data.
-Preserve the original sentence structure when possible and only expand weak parts instead of completely rewriting the style.



{
 "classification": "",
 "missing_anchors": [],
 "reasoning": "",
 "suggested_documentation": "",
 "defensible_note": ""
}
"""

# ===== BUTTON =====
if st.button("Review Documentation"):

    st.session_state.total_cases += 1

    if note.strip() == "":
        st.warning("Please paste a case note")

    else:
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": note}
                ],
                temperature=0
            )

            raw = response.choices[0].message.content

            # convert JSON text → python dict
            data = json.loads(raw)

            # ---- DISPLAY CLEAN REPORT ----
            st.subheader(f"Risk Level: {data['classification']}")

            st.write("**Suggested Documentation Improvements**")
            st.text_area("Guidance", data["suggested_documentation"], height=150)

            st.write("**Defensible Chart Version (Ready to Paste)**")
            st.text_area("Final Note", data["defensible_note"], height=220)

        except Exception as e:
            st.error("AI generation error:")
            st.code(str(e))
