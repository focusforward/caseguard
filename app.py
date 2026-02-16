from openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import json
import requests
import streamlit.components.v1 as components
# -------------------- CASEGUARD RISK ENGINE v1 --------------------
import re

def rule_classify(note: str):
    text = note.lower()

    # ---------------- DISPOSITION ----------------
    admitted = bool(re.search(r'\b(admit|admitted|icu|shifted|referred|transferred|ward|ot)\b', text))
    discharged = bool(re.search(r'\b(discharge|discharged|home|opd|review|sent home)\b', text))

    # ---------------- PRESENTATION GROUPS ----------------
    trauma = any(k in text for k in ["rta","accident","fall","trauma","assault","hit"])
    high_energy = any(k in text for k in ["fall from height","10ft","high speed","railway","ejection","run over"])
    head = "head injury" in text
    chest = "chest pain" in text or "jaw pain" in text
    abdomen = any(k in text for k in ["abd pain","abdominal pain","rlq pain","guarding"])
    neuro = any(k in text for k in ["seizure","unconscious","syncope","weakness","slurring"])
    hypoxia = bool(re.search(r'\b(8[0-9]%|spo2 8|saturation 8)\b', text))
    tachy = bool(re.search(r'\b(pulse 1[3-9][0-9]|hr 1[3-9][0-9]|140 bpm)\b', text))

    # ---------------- INVESTIGATIONS ----------------
    imaging = any(k in text for k in ["xray","ct","mri","scan"])
    cardiac_tests = any(k in text for k in ["ecg","troponin"])
    abdomen_scan = any(k in text for k in ["usg","ultrasound","ct abdomen"])

    # =====================================================
    # ADMITTED PATIENTS (ESCALATION PROTECTS DOCTOR)
    # =====================================================
    if admitted:

        # missed mandatory immediate step → borderline
        if chest and not cardiac_tests:
            return "BORDERLINE"

        if head and not imaging:
            return "BORDERLINE"

        if abdomen and not abdomen_scan:
            return "BORDERLINE"

        return "SAFE"

    # =====================================================
    # DISCHARGED PATIENTS (HIGH RISK ZONE)
    # =====================================================
    if discharged:

        # life threatening physiology
        if hypoxia or tachy:
            return "DANGEROUS"

        # high energy trauma always dangerous if discharged
        if high_energy:
            return "DANGEROUS"

        # trauma without imaging
        if trauma and not imaging:
            return "DANGEROUS"

        # head injury
        if head and not imaging:
            return "DANGEROUS"

        # chest pain without cardiac rule-out
        if chest and not cardiac_tests:
            return "DANGEROUS"

        # abdomen red flags without scan
        if abdomen and not abdomen_scan:
            return "DANGEROUS"

        # neurological event discharged
        if neuro and not imaging:
            return "DANGEROUS"
        # hemodynamic instability
        if re.search(r'\b(bp\s*9\d\/\d\d|bp\s*8\d\/\d\d|hypotension|shock)\b', text):
            return "DANGEROUS"

    return None


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
The suggested_documentation field is only for strengthening wording when risk remains uncertain.
If a serious condition has been reasonably excluded (e.g., negative imaging with improvement),
avoid requesting additional examination details.
In such cases guidance should be minimal or empty.
Never state that something was not documented, missing, or absent in the medical record. 
The defensible_note must read as a normal clinical entry, not an audit comment.
The defensible_note must never describe documentation quality or completeness.
Do not write phrases like "not detailed", "not recorded", "not mentioned", or similar.
Instead, rewrite the note into a complete neutral clinical statement using only known facts and neutral safety wording.
Defensible note writing style (mandatory pattern):

Whenever possible, structure the final note in this order:

1. Patient + complaint
2. Treatment given
3. Patient condition after treatment (observable state)
4. Clear return precautions

Prefer outcome-based wording:
Use phrases like:
"symptomatically improved"
"comfortable after treatment"
"tolerating orally"
"no new complaints during observation"

Avoid vague process claims:
Do NOT use phrases like:
"assessment done"
"evaluated"
"clinically assessed"
"examined and stable"
unless specific findings are explicitly present in the original note.
Primary task priority:

Your first responsibility is to determine medico-legal defensibility of the clinical decision,
NOT to improve documentation wording.

Classification must be based on whether a reasonable clinician could safely defend the discharge
if the patient later has a serious outcome.

If a serious condition was not reasonably excluded for a high-risk presentation,
the case must be classified DANGEROUS even if the wording sounds acceptable.

Documentation improvement suggestions must never downgrade the risk classification.
High-Risk Presentation Override:

Certain complaints carry inherent medico-legal risk even if documentation appears complete.

For the following presentations, SAFE classification should generally NOT be given unless
clear exclusion reasoning or definitive investigation is mentioned:

Trauma / RTA / falls with limb pain or swelling
Head injury
Chest pain
Severe abdominal pain
Focal neurological symptoms or seizures
Persistent vomiting in child
Altered consciousness
Breathlessness

In these cases:
If exclusion reasoning or investigation is not explicitly stated,
classification must be DANGEROUS when discharge occurs after only symptomatic treatment
and no exclusion of serious injury is documented.
BORDERLINE should only be used when partial evaluation is mentioned but not clearly adequate.
Classification independence rule:

The risk classification must be determined using strict medico-legal reasoning
and must ignore tone guidance.

Tone rules apply only to the wording of reasoning and suggested_documentation,
not to the classification decision.

If a high-risk condition is not reasonably excluded, classification must be DANGEROUS
even if the explanation remains gentle and non-accusatory.


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
60M diabetes weakness sugar 50 given dextrose discharge

OUTPUT:
{
 "classification": "BORDERLINE",
 "missing_anchors": [
   "post-treatment reassessment",
   "cause assessment",
   "observation reasoning",
   "safety-net advice"
 ],
 "reasoning": "Hypoglycaemia corrected but stability and recurrence risk not documented.",
 "suggested_documentation": "Document repeat glucose after recovery, possible cause, oral intake, and return precautions.",
 "defensible_note": "60-year-old diabetic presented with hypoglycaemia (glucose 50). After dextrose, repeat glucose normal and patient asymptomatic, tolerating oral intake. Likely medication-related episode. Advised regular meals and to return if dizziness, sweating, or altered sensorium recur."
}

INPUT:
3 yr old breathless wheeze given nebulization sleeping peacefully discharge

OUTPUT:
{
 "classification": "BORDERLINE",
 "missing_anchors": [
   "objective reassessment",
   "severity documentation",
   "safety-net advice",
   "discharge reasoning"
 ],
 "reasoning": "Symptomatic improvement documented but stability not objectively demonstrated.",
 "suggested_documentation": "Document respiratory rate, work of breathing, oxygen saturation, and clear return precautions after treatment response.",
 "defensible_note": "3-year-old with wheeze improved after nebulisation. Respiratory rate and oxygen saturation normal with no retractions. Tolerating oral intake. Parents advised to return for fast breathing, retractions, poor feeding, or worsening symptoms."
}

INPUT:
45M severe headache BP 160/95 CT head negative pain improved discharged

OUTPUT:
{
 "classification": "SAFE",
 "missing_anchors": [],
 "reasoning": "Serious intracranial causes evaluated and excluded with symptom improvement.",
 "suggested_documentation": "Brief reassurance context if assessed.",
 "defensible_note": "45-year-old male with severe headache. BP 160/95. CT head negative. Pain improved after treatment. Discharged with advice to return if worsening headache, neurological symptoms, or persistent vomiting."
}

INPUT:
32F RLQ pain CBC normal CT negative appendicitis discharged analgesics

OUTPUT:
{
 "classification": "SAFE",
 "missing_anchors": [],
 "reasoning": "High-risk condition investigated and excluded; symptoms improved allowing safe discharge.",
 "suggested_documentation": "Optional brief clinical context if assessed.",
 "defensible_note": "32-year-old female with right lower quadrant pain. CBC normal. CT abdomen/pelvis negative for appendicitis. Discharged with analgesics and advised return if worsening pain, fever, or new symptoms."
}

INPUT:
35 yr backache after long sitting painkiller given review

OUTPUT:
{
 "classification": "SAFE",
 "missing_anchors": [],
 "reasoning": "Likely mechanical musculoskeletal pain.",
 "suggested_documentation": "Briefly mention absence of concerning features if assessed.",
 "defensible_note": "35-year-old with back pain after prolonged sitting. Analgesic given. Advised review in 2 weeks or earlier if worsening or neurological symptoms develop."
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
st.set_page_config(page_title="CaseGuard")
# -------------------- LOGIN GATE --------------------
import requests
from datetime import datetime

SHEET_URL = "https://opensheet.elk.sh/1NA4S23i9t_q9D40EaedCvuuoN2EJdnGpDbtQnhM86_M/Sheet1"

def check_access(email):
    try:
        users = requests.get(SHEET_URL, timeout=5).json()
        for u in users:
            if u["email"].strip().lower() == email.strip().lower():
                expiry = datetime.strptime(u["expiry"], "%d-%m-%Y")
                if expiry >= datetime.today():
                    return True
        return False
    except:
        return False

# session login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("CaseGuard Access")

    email_input = st.text_input("Enter your registered email")

    if st.button("Login"):
        if check_access(email_input):
            st.session_state.logged_in = True
            st.success("Access granted")
            st.rerun()
        else:
            st.error("No active subscription")

    st.stop()

st.title("CaseGuard")
st.caption("Medico-legal documentation assistant for everyday clinical practice.")
st.info(f"Cases reviewed this session: {st.session_state.total_cases}")

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
                forced_class = rule_classify(note)
                status = st.status("Analyzing medico-legal risk...", expanded=False)

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
                status.update(label="Analysis complete", state="complete")

                st.session_state.result = json.loads(raw)
                if forced_class:
                    st.session_state.result["classification"] = forced_class
                st.session_state.total_cases += 1

                # ---- GOOGLE USAGE TRACKING ----
                try:
                    requests.post(
                        "https://script.google.com/macros/s/AKfycbxj3IVtwoDowJsyj7v7AAikZyquduybRFbil6xOJsv709bF5SfyYuenb3OrNyhrdtha/exec",
                        json={
                            "risk": st.session_state.result.get("classification",""),
                            "length": len(note)
                        },
                        timeout=2
                    )
                except:
                    pass

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

    data = dict(st.session_state.result)  # make a copy

    # Final authority: rule engine overrides AI classification
    forced_class = rule_classify(note)
    if forced_class:
        data["classification"] = forced_class

    risk = data.get("classification","")
    guidance = data.get("suggested_documentation","").strip()
    final_note = data.get("defensible_note","")

    st.subheader(f"Risk Level: {risk}")

    # Guidance
    if risk != "SAFE" and guidance:
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

    elif risk == "SAFE":
        st.success("No additional documentation improvement needed")

    # Final Note
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
