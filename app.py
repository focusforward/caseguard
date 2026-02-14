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
You are a medico-legal documentation auditor assisting doctors.

Your role is NOT to judge medical correctness.
Your role is to improve documentation defensibility.

Always produce a finished chart-ready note in the defensible_note field.

Return STRICT JSON:
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
