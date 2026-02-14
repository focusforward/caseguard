from openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import json
import streamlit.components.v1 as components

# -------------------- LOAD KEY --------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------- SYSTEM PROMPT --------------------
system_prompt = """
You are a clinical documentation assistant.
Rewrite rough clinical notes into concise, defensible chart entries.
Do not add new clinical findings.
The suggested_documentation field should contain a brief friendly clarification suggestion, not commands. Avoid starting sentences with words like "Document", "Include", or "Record".
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
 "suggested_documentation": "",
 "defensible_note": "4-year-old with fever, playful and tolerating feeds. Paracetamol given. Advised review if fever persists or child becomes unwell."
}

INPUT:
4 yr repeated vomiting sleepy after treatment discharged
OUTPUT:
{
 "classification": "BORDERLINE",
 "missing_anchors": [],
 "reasoning": "Post-vomiting drowsiness may require observation documentation.",
 "suggested_documentation": "Clarify alertness and hydration status.",
 "defensible_note": "4-year-old with multiple vomiting episodes treated with fluids. Child comfortable and arousable at discharge. Advised urgent return if persistent vomiting, lethargy, or poor intake."
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

# -------------------- SESSION STATE --------------------
if "result" not in st.session_state:
    st.session_state.result = None

if "total_cases" not in st.session_state:
    st.session_state.total_cases = 0

# -------------------- UI --------------------
st.set_page_config(page_title="Clinical Defence Note Generator")

st.title("Clinical Defence Note Generator")
st.caption("Assistive documentation review tool. Clinical decisions remain with treating physician.")
st.info(f"Cases reviewed this session: {st.session_state.total_cases}")

# initialize once
if "note_value" not in st.session_state:
    st.session_state.note_value = ""

note = st.text_area("Paste Case Note", height=250, key="note_value")

col1, col2 = st.columns(2)

# -------- REVIEW BUTTON --------
with col1:
    if st.button("Review Documentation"):
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
                    temperature=0,
                 response_format={"type": "json_object"}
  )

                raw = response.choices[0].message.content
                st.session_state.result = json.loads(raw)
                st.session_state.total_cases += 1

            except Exception as e:
                st.error("AI generation error:")
                st.code(str(e))

# -------- CLEAR BUTTON --------
with col2:
    if st.button("Clear"):
        st.session_state.clear()
        st.rerun()

# -------------------- DISPLAY --------------------
if st.session_state.result:

    data = st.session_state.result

    st.subheader(f"Risk Level: {data.get('classification','')}")

    # -------- GUIDANCE --------
    guidance = data.get("suggested_documentation","").strip()

    if guidance:
        st.write("### Suggested Documentation Improvements")
        st.text_area("Guidance", guidance, height=130)

        components.html(f"""
            <textarea id="guidecopy" style="width:100%;height:60px;">{guidance}</textarea>
            <button onclick="copyGuide()">Copy Guidance</button>
            <script>
            function copyGuide() {{
                var copyText = document.getElementById("guidecopy");
                copyText.select();
                document.execCommand("copy");
            }}
            </script>
        """, height=120)

    else:
        st.success("No additional documentation improvement needed")

    # -------- FINAL NOTE --------
    final_note = data.get("defensible_note","")

    st.write("### Defensible Chart Version (Ready to Paste)")
    st.text_area("Final Note", final_note, height=180)

    components.html(f"""
        <textarea id="finalcopy" style="width:100%;height:80px;">{final_note}</textarea>
        <button onclick="copyFinal()">Copy Final Note</button>
        <script>
        function copyFinal() {{
            var copyText = document.getElementById("finalcopy");
            copyText.select();
            document.execCommand("copy");
        }}
        </script>
    """, height=140)
