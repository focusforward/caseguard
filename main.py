from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

app = FastAPI()
app.mount("/static", StaticFiles(directory="."), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- LOAD KEY --------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# -------------------- LOGIN CHECK --------------------
SHEET_URL = "https://docs.google.com/spreadsheets/d/1NA4S23i9t_q9D40EaedCvuuoN2EJdnGpDbtQnhM86_M/export?format=csv&gid=0"


def check_access(email):
    try:
        r = requests.get(SHEET_URL, timeout=5)
        lines = r.text.splitlines()[1:]  # skip header

        for line in lines:
            e, expiry = line.split(",")
            if e.strip().lower() == email.strip().lower():
                expiry_date = datetime.strptime(expiry.strip(), "%d-%m-%Y")
                if expiry_date >= datetime.today():
                    return True
        return False
    except:
        return False
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
# what the website will send
class CaseInput(BaseModel):
    email: str
    note: str
@app.post("/analyze")
def analyze_case(data: CaseInput):

    # subscription check
    if not check_access(data.email):
        return {"error": "No active subscription"}

    # rule engine classification
    forced_class = rule_classify(data.note)

    # AI rewrite
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data.note}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    # rule engine overrides AI classification
    if forced_class:
        result["classification"] = forced_class

    return result
    import os
    @app.get("/")
    def homepage():
        return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))



