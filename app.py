import streamlit.components.v1 as components

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

note = st.text_area("Paste Case Note", height=300, key="input_note")

if st.button("Clear"):
    st.session_state.input_note = ""


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
-Do NOT include commentary about documentation quality (e.g., "no objective data documented", "risk factors not mentioned", "information absent").
-Write only patient-care statements suitable for a medical record.
-Never add new clinical facts (positive or negative) that were not stated in the original note.
-Do not assume absence of symptoms, past history, exam findings, or risk factors.
-Only rephrase existing information and add generic safety-net advice.
-The defensible_note must NOT introduce any new clinical findings, including negative findings.
-Do not write statements like "no focal deficits", "no red flags", "no risk factors", or similar unless explicitly present in the original note.
-If information is missing, omit it entirely rather than assuming or negating it.
-You may add neutral process statements such as "symptomatically improved", "clinically stable for discharge", or "return precautions explained".
-These describe clinical reasoning and are allowed even if not explicitly written, because they do not introduce new clinical findings.
-Do NOT switch to giving instructions — always produce a finished note.
-Prefer extending the original sentence instead of rewriting from scratch whenever possible.
-Avoid repeating words or phrases from the original note unnecessarily and produce only one final version of the note.
-Encounter context wording:
If the note describes an outpatient clinic or non-emergency visit (routine exam, chronic symptoms, refraction, irritation, follow-up),
avoid emergency terms such as "stable for discharge" or "discharged".

Use neutral outpatient language instead:
- "advised follow-up"
- "review if symptoms worsen"
- "continue treatment as prescribed"

Reserve "stable for discharge" only for acute emergency presentations.
-For outpatient or clinic encounters, still produce a finished chart entry.
-Never output documentation instructions such as "document", "note", or "include".
-Always rewrite the note directly using appropriate clinic wording.
When clinical details are minimal, use neutral phrasing rather than instructions.

Allowed neutral wording examples:
- "evaluated and treated symptomatically"
- "treatment started as above"
- "advised follow-up if symptoms persist or worsen"

These phrases are preferred instead of asking the user to document more information.
Never switch to giving documentation advice.
If the original note contains very limited clinical information,
produce a minimal defensible note using only the stated facts.
Do not request more details.
Do not switch to advice mode.

Example acceptable minimal output:
"Right eye irritation for 1 day in contact lens user. Treatment started as above. Advised review if symptoms persist or worsen."
Output priority:
The "defensible_note" must always be produced as a complete chart entry.
Even if information is minimal, never replace it with guidance or suggestions.
The "reasoning" field may explain briefly, but the defensible_note is mandatory and takes priority.
For chronic or non-acute complaints (e.g., joint pains, chronic conditions, medication review),
use neutral follow-up phrasing instead of emergency or acute wording.

Acceptable minimal structure:
"[complaint] evaluated. Symptomatic treatment given. Advised follow-up as planned or earlier if symptoms worsen."

Do not switch to instructions when details are limited.















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

# ---- Suggested Documentation ----
st.write("**Suggested Documentation Improvements**")
st.text_area("Guidance", data["suggested_documentation"], height=150, key="guide")

if st.button("Copy Guidance"):
    components.html(f"""
    <script>
    navigator.clipboard.writeText({repr(data["suggested_documentation"])});
    </script>
    """, height=0)

# ---- Final Note ----
st.write("**Defensible Chart Version (Ready to Paste)**")
st.text_area("Final Note", data["defensible_note"], height=220, key="final")

if st.button("Copy Final Note"):
    components.html(f"""
    <script>
    navigator.clipboard.writeText({repr(data["defensible_note"])});
    </script>
    """, height=0)


        except Exception as e:
            st.error("AI generation error:")
            st.code(str(e))
