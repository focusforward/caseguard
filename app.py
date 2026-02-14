from openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import json
import streamlit.components.v1 as components

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

# Input
note = st.text_area("Paste Case Note", height=300, key="input_note")

if st.button("Clear"):
    st.session_state.input_note = ""

# ===== SYSTEM PROMPT =====
system_prompt = """
You are a clinical documentation assistant.
Rewrite rough clinical notes into concise, defensible chart entries.
Do not add new clinical findings.
Do not give advice about documentation.
Always output a finished chart-ready note.

Return STRICT JSON:
{
 "classification": "",
 "missing_anchors": [],
 "reasoning": "",
 "suggested_documentation": "",
 "defensible_note": ""
}

Examples:

INPUT:
25 yr old chest pain pain killer given discharged

OUTPUT:
{
 "classification": "DANGEROUS",
 "missing_anchors": [],
 "reasoning": "High risk symptom without documented evaluation.",
 "suggested_documentation": "Add reasoning and safety-net advice.",
 "defensible_note": "25-year-old with chest pain treated symptomatically. Advised urgent return if pain persists, worsens, or new symptoms develop."
}

INPUT:
4 yr fever playful eating well pcm discharge

OUTPUT:
{
 "classification": "SAFE",
 "missing_anchors": [],
 "reasoning": "Reassuring behaviour in febrile child.",
 "suggested_documentation": "Optional safety-net advice.",
 "defensible_note": "4-year-old with fever, playful and tolerating feeds. Paracetamol given. Advised review if fever persists or child becomes unwell."
}

INPUT:
20 yr contact lens irritation right eye 1 day moxifloxacin ketorolac

OUTPUT:
{
 "classification": "SAFE",
 "missing_anchors": [],
 "reasoning": "Minor outpatient condition.",
 "suggested_documentation": "",
 "defensible_note": "20-year-old contact lens user with right eye irritation for 1 day. Started on moxifloxacin and ketorolac. Advised review if symptoms worsen or vision changes."
}

INPUT:
65 yr female dm htn joint pain painkillers review 2 weeks

OUTPUT:
{
 "classification": "SAFE",
 "missing_anchors": [],
 "reasoning": "Chronic complaint follow-up.",
 "suggested_documentation": "",
 "defensible_note": "65-year-old with diabetes and hypertension presenting with joint pains. Symptomatic treatment given. Review in 2 weeks or earlier if worsening."
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
            data = json.loads(raw)

            # ---- OUTPUT ----
            st.subheader(f"Risk Level: {data['classification']}")

            # Suggested documentation
            st.write("**Suggested Documentation Improvements**")
            st.text_area("Guidance", data["suggested_documentation"], height=150, key="guide")

            if st.button("Copy Guidance"):
                components.html(f"""
                <script>
                navigator.clipboard.writeText({repr(data["suggested_documentation"])});
                </script>
                """, height=0)

            # Final note
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
