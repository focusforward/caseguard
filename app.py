from openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
from rules import analyze_note
from collections import Counter

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Session counters
if "total_cases" not in st.session_state:
    st.session_state.total_cases = 0

if "risk_log" not in st.session_state:
    st.session_state.risk_log = []

st.set_page_config(page_title="Clinical Defence Note Generator")

st.title("Clinical Defence Note Generator")
st.caption("Assistive documentation review tool. Clinical decisions remain with treating physician.")

st.info(f"Cases reviewed this session: {st.session_state.total_cases}")

note = st.text_area("Paste Emergency Case Note", height=300)

if st.button("Review Documentation"):

    st.session_state.total_cases += 1

    if note.strip() == "":
        st.warning("Please paste a case note")

    else:
        flags = analyze_note(note)

        if len(flags) == 0:
            st.success("No major documentation clarification required")

        else:
            st.subheader("Clarifications Recommended")
            for f in flags:
                st.warning(f)

            # store risks
            st.session_state.risk_log.extend(flags)

            prompt = f"""
You are assisting a doctor in documenting reasoning.

Patient note:
{note}

Concerns:
{flags}

Write a short clinical justification paragraph explaining reasonable medical decision-making.
Do NOT give treatment advice. Only explain reasoning.
"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role":"user","content":prompt}]
                )

                defence_text = response.choices[0].message.content

                st.subheader("Suggested Documentation")
                st.text_area("Editable Note", defence_text, height=200)
                st.caption("Doctor may edit before pasting into case sheet.")

            except Exception as e:
                st.error("AI generation error:")
                st.code(str(e))


# -------- SESSION REPORT --------
st.divider()
st.subheader("Session Risk Summary")

if st.session_state.total_cases > 0:

    total = st.session_state.total_cases
    risks = len(st.session_state.risk_log)

    st.write(f"Total cases reviewed: {total}")
    st.write(f"Total documentation risks detected: {risks}")

    counts = Counter(st.session_state.risk_log)

    st.write("Top recurring gaps:")
    for k, v in counts.items():
        st.write(f"{k} : {v}")
