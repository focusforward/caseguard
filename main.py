from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# -------------------- HEALTH CHECK --------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------- LOAD KEY --------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------- LOGIN CHECK --------------------
SHEET_URL = "https://docs.google.com/spreadsheets/d/1NA4S23i9t_q9D40EaedCvuuoN2EJdnGpDbtQnhM86_M/export?format=csv&gid=0"

def check_access(email: str) -> bool:
    try:
        r = requests.get(SHEET_URL, timeout=5)
        r.raise_for_status()
        lines = r.text.splitlines()[1:]  # skip header
        for line in lines:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            e = parts[0].strip().lower()
            expiry_str = parts[1].strip()
            if e == email.strip().lower():
                expiry_date = datetime.strptime(expiry_str, "%d-%m-%Y")
                return expiry_date >= datetime.today()
        return False
    except Exception:
        return False


# ============================================================
# RULE ENGINE
# ============================================================

def _imaging_done(text: str) -> bool:
    """
    Returns True only if imaging was actually performed (not just advised/ordered/pending/negated).
    Handles common Indian notation variations.
    """
    # Imaging was advised/ordered but not yet done
    pending = bool(re.search(
        r'\b(ct|x[\s-]?ray|mri|scan|xray)\b.{0,20}\b(advised|planned|ordered|pending|to be done|recommended|will be|for consideration)\b',
        text
    ))
    pending = pending or bool(re.search(
        r'\b(advise[d]?|plan[ned]?|order[ed]?|recommend[ed]?)\b.{0,20}\b(ct|x[\s-]?ray|mri|scan|xray)\b',
        text
    ))
    # Imaging explicitly negated: "no ct", "no xray", "ct not done", "without imaging"
    negated = bool(re.search(
        r'\b(no|without|not done|not performed|not available)\b.{0,10}\b(ct|x[\s-]?ray|mri|scan|xray|imaging)\b',
        text
    ))
    done = bool(re.search(r'\b(ct|x[\s-]?ray|mri|scan|xray)\b', text))
    return done and not pending and not negated

def _cardiac_done(text: str) -> bool:
    """ECG or troponin present and not just advised."""
    pending = bool(re.search(
        r'\b(ecg|troponin|trop)\b.{0,20}\b(advised|planned|ordered|pending)\b', text
    ))
    pending = pending or bool(re.search(
        r'\b(advise[d]?|plan[ned]?|order[ed]?)\b.{0,20}\b(ecg|troponin|trop)\b', text
    ))
    done = bool(re.search(r'\b(ecg|troponin|trop)\b', text))
    return done and not pending

def _abdomen_scan_done(text: str) -> bool:
    pending = bool(re.search(
        r'\b(usg|ultrasound|ct abdomen|ct belly)\b.{0,20}\b(advised|planned|ordered|pending)\b', text
    ))
    pending = pending or bool(re.search(
        r'\b(advise[d]?|plan[ned]?|order[ed]?)\b.{0,20}\b(usg|ultrasound|ct abdomen)\b', text
    ))
    done = bool(re.search(r'\b(usg|ultrasound|ct abdomen|ct belly)\b', text))
    return done and not pending

def _result_negative(text: str) -> bool:
    """Check if investigation result was explicitly negative/normal."""
    return bool(re.search(
        r'\b(negative|normal|no fracture|no bleed|no pe|clear|unremarkable|within normal|wn[l]?)\b', text
    ))

def _improvement_noted(text: str) -> bool:
    """Check if clinical improvement was documented."""
    return bool(re.search(
        r'\b(improved|relief|better|settled|comfortable|afebrile|asymptomatic|pain free|resolved|responding)\b', text
    ))

def _hemodynamic_instability(text: str) -> bool:
    """Detect shock/hypotension with flexible BP notation (BP90/60, BP-90/60, BP 90/60)."""
    return bool(re.search(
        r'\b(hypotension|shock|hypotensive)\b'
        r'|bp[\s\-]*[89]\d[\s]*/[\s]*\d\d'   # BP 90/60, BP-88/50
        r'|\b(pulse|hr|heart rate)[\s\-]*1[5-9]\d\b'  # HR 150+
        r'|\bsbp[\s\-]*[89]\d\b',
        text
    ))

def rule_classify(note: str) -> dict:
    """
    Returns dict with:
      - classification: "SAFE" | "BORDERLINE" | "DANGEROUS" | None
      - rule_flags: list of triggered rule names (for AI context)
      - pending_investigations: list of what was advised but not done
    """
    text = note.lower()

    # ---- DISPOSITION ----
    admitted = bool(re.search(
        r'\b(admit|admitted|icu|shifted|transferred|ward|ot)\b', text
    ))
    referred_out = bool(re.search(
        r'\b(referred|refer)\b', text
    ))
    discharged = bool(re.search(
        r'\b(discharge|discharged|home|sent home)\b', text
    ))
    # review/opd alone = ambiguous follow-up, not confirmed discharge
    review_only = (
        bool(re.search(r'\b(review|opd follow)\b', text))
        and not discharged and not admitted
    )

    # Admission takes priority over discharge if both appear
    if admitted and discharged:
        discharged = False

    # ---- PRESENTATION ----
    trauma        = bool(re.search(r'\b(rta|road traffic|accident|fall|trauma|assault|hit by)\b', text))
    high_energy   = bool(re.search(r'\b(fall from height|high speed|railway|ejection|run over|polytrauma)\b', text))
    head          = bool(re.search(r'\b(head injury|head trauma|scalp|loc|loss of consciousness)\b', text))
    chest         = bool(re.search(r'\b(chest pain|jaw pain|left arm pain|cardiac|heart attack|myocardial)\b', text))
    abdomen       = bool(re.search(r'\b(abd(ominal)? pain|rlq|llq|epigastric|guarding|rigidity|peritonitis)\b', text))
    neuro         = bool(re.search(r'\b(seizure|fit|unconscious|syncope|collapse|weakness|facial droop|slurring|aphasia|stroke|tia)\b', text))
    paed_age      = bool(re.search(r'\b([0-9]|1[0-5])\s*(yr|year|month|mo|m)\s*(old)?\b', text)) and not bool(re.search(r'\b(1[6-9]|[2-9][0-9])\s*(yr|year)\b', text))
    breathless    = bool(re.search(r'\b(breathless|sob|dyspnoea|respiratory distress|spo2|saturation)\b', text))
    hypoxia       = bool(re.search(r'\bspo2[\s\-]*[78]\d\b|saturation[\s\-]*[78]\d|\b[78]\d\s*%\s*(spo2|saturation|o2)\b', text))
    tachy         = bool(re.search(r'\b(pulse|hr|heart rate)[\s\-]*1[3-9][0-9]\b', text))

    # ---- INVESTIGATIONS ----
    imaging_done    = _imaging_done(text)
    cardiac_done    = _cardiac_done(text)
    abdomen_done    = _abdomen_scan_done(text)
    result_negative = _result_negative(text)
    improved        = _improvement_noted(text)
    haemodynamic_shock = _hemodynamic_instability(text)

    # Track what was pending
    pending = []
    if bool(re.search(r'\b(ct|x[\s-]?ray|mri|scan|xray)\b', text)) and not imaging_done:
        pending.append("imaging (advised but result unknown)")
    if bool(re.search(r'\b(ecg|troponin)\b', text)) and not cardiac_done:
        pending.append("cardiac workup (advised but result unknown)")
    if bool(re.search(r'\b(usg|ultrasound)\b', text)) and not abdomen_done:
        pending.append("abdominal scan (advised but result unknown)")

    rule_flags = []

    # ============================================================
    # REFERRED OUT — treat as intermediate (neither admitted nor
    # fully discharged with own workup)
    # ============================================================
    if referred_out and not admitted:
        # Referral is generally protective but note must justify it
        rule_flags.append("referred_out")
        # Falls through to further checks below

    # ============================================================
    # ADMITTED PATIENTS
    # ============================================================
    if admitted:
        if haemodynamic_shock:
            rule_flags.append("haemodynamic_instability_admitted")
            return {"classification": "BORDERLINE", "rule_flags": rule_flags, "pending": pending}

        if chest and not cardiac_done:
            rule_flags.append("chest_pain_no_cardiac_workup_admitted")
            return {"classification": "BORDERLINE", "rule_flags": rule_flags, "pending": pending}

        if head and not imaging_done:
            rule_flags.append("head_injury_no_imaging_admitted")
            return {"classification": "BORDERLINE", "rule_flags": rule_flags, "pending": pending}

        if abdomen and not abdomen_done:
            rule_flags.append("abdominal_pain_no_scan_admitted")
            return {"classification": "BORDERLINE", "rule_flags": rule_flags, "pending": pending}

        if hypoxia:
            rule_flags.append("hypoxia_admitted")
            return {"classification": "BORDERLINE", "rule_flags": rule_flags, "pending": pending}

        return {"classification": "SAFE", "rule_flags": rule_flags, "pending": pending}

    # ============================================================
    # DISCHARGED / REVIEW ONLY PATIENTS
    # ============================================================
    if discharged or review_only:

        # Immediately life-threatening physiology at discharge = always DANGEROUS
        if haemodynamic_shock:
            rule_flags.append("haemodynamic_instability_discharged")
            return {"classification": "DANGEROUS", "rule_flags": rule_flags, "pending": pending}

        if hypoxia:
            rule_flags.append("hypoxia_discharged")
            return {"classification": "DANGEROUS", "rule_flags": rule_flags, "pending": pending}

        if tachy:
            rule_flags.append("tachycardia_discharged")
            return {"classification": "DANGEROUS", "rule_flags": rule_flags, "pending": pending}

        # High energy trauma always dangerous if discharged
        if high_energy:
            rule_flags.append("high_energy_trauma_discharged")
            return {"classification": "DANGEROUS", "rule_flags": rule_flags, "pending": pending}

        # Head injury without imaging
        if head and not imaging_done:
            rule_flags.append("head_injury_no_imaging_discharged")
            return {"classification": "DANGEROUS", "rule_flags": rule_flags, "pending": pending}

        # Head injury with imaging — check if result documented
        if head and imaging_done and not result_negative:
            rule_flags.append("head_injury_imaging_result_unclear")
            return {"classification": "BORDERLINE", "rule_flags": rule_flags, "pending": pending}

        # Chest pain without cardiac rule-out
        if chest and not cardiac_done:
            rule_flags.append("chest_pain_no_cardiac_workup_discharged")
            return {"classification": "DANGEROUS", "rule_flags": rule_flags, "pending": pending}

        # Chest pain with workup — check result documented
        if chest and cardiac_done and not result_negative:
            rule_flags.append("chest_pain_workup_result_unclear")
            return {"classification": "BORDERLINE", "rule_flags": rule_flags, "pending": pending}

        # Trauma without imaging
        if trauma and not imaging_done:
            rule_flags.append("trauma_no_imaging_discharged")
            return {"classification": "DANGEROUS", "rule_flags": rule_flags, "pending": pending}

        # Abdominal red flags without scan
        if abdomen and not abdomen_done:
            rule_flags.append("abdominal_pain_no_scan_discharged")
            return {"classification": "DANGEROUS", "rule_flags": rule_flags, "pending": pending}

        # Neurological event without imaging
        if neuro and not imaging_done:
            rule_flags.append("neuro_event_no_imaging_discharged")
            return {"classification": "DANGEROUS", "rule_flags": rule_flags, "pending": pending}

        # Breathlessness without documented saturation or assessment
        if breathless and not improved and not result_negative:
            rule_flags.append("breathlessness_no_resolution_documented")
            return {"classification": "BORDERLINE", "rule_flags": rule_flags, "pending": pending}

        # Paediatric persistent vomiting
        if paed_age and bool(re.search(r'\b(vomit|vomiting)\b', text)) and not improved:
            rule_flags.append("paed_persistent_vomiting_no_improvement")
            return {"classification": "BORDERLINE", "rule_flags": rule_flags, "pending": pending}

    # No rule triggered — let AI decide
    return {"classification": None, "rule_flags": [], "pending": pending}


# ============================================================
# SYSTEM PROMPT — restructured for clarity and accuracy
# ============================================================

system_prompt = """
You are CaseGuard — a medico-legal clinical documentation reviewer used by emergency and outpatient doctors in India.

Your job is to:
1. Classify the medico-legal risk of the clinical decision documented
2. Identify what is missing from a defensibility standpoint
3. Rewrite the note as a clean, defensible chart entry

═══════════════════════════════════════════
RULE 1 — NEVER FABRICATE CLINICAL FACTS
═══════════════════════════════════════════
This is the most important rule. It overrides everything else.

- NEVER add clinical findings not stated in the note
- NEVER assume a test result (e.g., do NOT write "CT normal" if only "CT advised" was stated)
- NEVER infer that a patient improved unless improvement is explicitly stated
- NEVER add vitals, physical findings, or history not present in the original note
- If an investigation was "advised", "ordered", or "planned" — it is PENDING. Write it as pending.
- Only use facts explicitly present in the note

═══════════════════════════════════════════
RULE 2 — CLASSIFICATION LOGIC
═══════════════════════════════════════════
Classify based strictly on whether a reasonable clinician could defend this decision if the patient deteriorated.

SAFE: The clinical decision is defensible. High-risk conditions were investigated and reasonably excluded, 
      OR the presentation is genuinely low-risk. Improvement was documented. Safety netting was given.

BORDERLINE: Partial evaluation exists but gaps remain. The decision is not clearly indefensible, 
            but documentation is incomplete enough that a medicolegal challenge is possible.

DANGEROUS: A high-risk condition was not reasonably excluded before discharge. No documentation 
           of investigation, clinical reasoning, or exclusion of serious diagnosis exists.

High-risk presentations — SAFE only if exclusion is explicitly documented:
- Chest pain / jaw pain / left arm pain
- Head injury / LOC
- Trauma (RTA, fall, assault)
- Severe abdominal pain with guarding or peritonism
- Focal neurology / stroke / seizure / syncope
- Breathlessness with hypoxia
- Altered consciousness
- High energy mechanism of injury

DO NOT downgrade a DANGEROUS classification because the note sounds polite or the treatment seems reasonable.
DO NOT upgrade a DANGEROUS classification based on documentation improvements alone — 
   if the clinical action itself was indefensible, it remains DANGEROUS.

═══════════════════════════════════════════
RULE 3 — MISSING ANCHORS
═══════════════════════════════════════════
missing_anchors = things a medicolegal reviewer would immediately notice are absent.
For DANGEROUS cases: always populate missing_anchors — this is where the risk lives.
For BORDERLINE: populate with what's partially missing.
For SAFE: leave empty unless a minor addition would strengthen the note.

Examples of anchors:
- "ECG result not documented"
- "Troponin not done or result unknown"  
- "Neurological status at discharge not recorded"
- "No return precautions documented"
- "CT result pending — not yet available"
- "Mechanism of injury not detailed"
- "Post-treatment reassessment not documented"

═══════════════════════════════════════════
RULE 4 — DEFENSIBLE NOTE WRITING
═══════════════════════════════════════════
Structure (in this order when possible):
1. Patient demographics + presenting complaint
2. Key findings or investigations (only if documented)
3. Treatment given
4. Patient status after treatment (only if documented)
5. Plan / disposition
6. Return precautions (always include)

Strict style rules:
- Write ONLY facts present in the original note
- For pending investigations: "CT head advised — result awaited"
- For unknown results: "ECG done — result not documented in this entry"
- Use outcome language only if improvement is stated: "symptomatically improved", "tolerating orally"
- Do NOT use: "assessed", "evaluated", "examined and stable", "clinically assessed" — unless examination findings are actually stated
- Do NOT use: "not documented", "not recorded", "not mentioned" — instead write around the gap
- Do NOT describe documentation quality in the note itself
- Keep it concise. One short paragraph. Chart-ready.

═══════════════════════════════════════════
RULE 5 — SUGGESTED DOCUMENTATION
═══════════════════════════════════════════
- Brief, friendly, clinically practical
- Tells the doctor what to ADD to the existing note — not what they did wrong
- Never use: "Document", "Include", "Record", "Note that"
- For SAFE cases: keep this minimal or empty
- For DANGEROUS cases: this should be the most helpful field — tell them exactly what would have made this defensible
- Maximum 2 sentences

═══════════════════════════════════════════
RULE 6 — REASONING
═══════════════════════════════════════════
- One or two sentences maximum
- Explain WHY the classification was given from a medico-legal standpoint
- Do not repeat the note back
- Do not use soft language to soften a DANGEROUS classification

═══════════════════════════════════════════
OUTPUT FORMAT — STRICT JSON ONLY
═══════════════════════════════════════════
{
  "classification": "SAFE" | "BORDERLINE" | "DANGEROUS",
  "missing_anchors": [],
  "reasoning": "",
  "suggested_documentation": "",
  "defensible_note": ""
}

No text outside the JSON. No markdown. No explanation. Just the JSON object.

═══════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════

INPUT: 25 yr old chest pain pain killer given discharged
OUTPUT:
{
  "classification": "DANGEROUS",
  "missing_anchors": ["ECG not documented", "troponin not done", "cardiac cause not excluded", "no return precautions recorded"],
  "reasoning": "Chest pain discharged with only analgesia — no cardiac workup documented. Indefensible if patient has ACS.",
  "suggested_documentation": "Adding ECG result, troponin, and a clear reason for excluding cardiac cause would significantly strengthen this entry.",
  "defensible_note": "25-year-old with chest pain. Analgesic administered. ECG and troponin not documented in this entry. Advised urgent return if pain persists, worsens, or new symptoms develop."
}

INPUT: 44 yr old head injury dizziness bp normal ct scan advised admitted
OUTPUT:
{
  "classification": "BORDERLINE",
  "missing_anchors": ["CT result pending — not yet available", "neurological status not recorded", "monitoring plan not documented"],
  "reasoning": "Admission is appropriate and protective. However, CT result is outstanding and neurological observations are not documented.",
  "suggested_documentation": "Adding neuro observations and updating this entry with CT result once available will complete the record.",
  "defensible_note": "44-year-old with head injury and dizziness. Blood pressure normal. CT head advised — result awaited. Admitted for observation and neurological monitoring pending CT outcome."
}

INPUT: 60M diabetes weakness sugar 50 given dextrose discharge
OUTPUT:
{
  "classification": "BORDERLINE",
  "missing_anchors": ["post-treatment glucose not documented", "cause of hypoglycaemia not noted", "oral intake status not recorded", "return precautions not documented"],
  "reasoning": "Hypoglycaemia treated but no post-correction reassessment documented — recurrence risk and stability are unconfirmed.",
  "suggested_documentation": "Adding a post-dextrose glucose reading, oral tolerance, and return advice would make this defensible.",
  "defensible_note": "60-year-old diabetic with hypoglycaemia (glucose 50). IV dextrose administered. Discharged — post-treatment glucose and oral tolerance not documented in this entry. Advised return if dizziness, sweating, or altered sensorium recur."
}

INPUT: 3 yr old breathless wheeze given nebulization sleeping peacefully discharge
OUTPUT:
{
  "classification": "BORDERLINE",
  "missing_anchors": ["SpO2 not documented", "respiratory rate not recorded", "post-nebulisation objective status not noted", "return precautions not documented"],
  "reasoning": "Sleeping peacefully suggests improvement but no objective parameters (SpO2, RR) are recorded to demonstrate clinical stability at discharge.",
  "suggested_documentation": "Adding SpO2 and respiratory rate after nebulisation, and return precautions, would make this defensible.",
  "defensible_note": "3-year-old with wheeze treated with nebulisation. Child settled and comfortable post-treatment. Parents advised urgent return for fast breathing, recession, poor feeding, or worsening symptoms."
}

INPUT: 45M severe headache BP 160/95 CT head negative pain improved discharged
OUTPUT:
{
  "classification": "SAFE",
  "missing_anchors": [],
  "reasoning": "Serious intracranial pathology excluded with negative CT. Symptom improvement documented. High-risk presentation adequately evaluated.",
  "suggested_documentation": "",
  "defensible_note": "45-year-old male with severe headache. BP 160/95. CT head negative. Pain improved with treatment. Discharged with advice to return urgently for worsening headache, vomiting, or neurological symptoms."
}

INPUT: 32F RLQ pain CBC normal CT negative appendicitis discharged analgesics
OUTPUT:
{
  "classification": "SAFE",
  "missing_anchors": [],
  "reasoning": "Appendicitis excluded with CT. CBC normal. Analgesics given with appropriate discharge.",
  "suggested_documentation": "",
  "defensible_note": "32-year-old female with right lower quadrant pain. CBC normal. CT abdomen negative for appendicitis. Discharged with analgesics. Advised return if pain worsens, fever develops, or new symptoms arise."
}

INPUT: 35 yr backache after long sitting painkiller given review
OUTPUT:
{
  "classification": "SAFE",
  "missing_anchors": [],
  "reasoning": "Mechanical low back pain with clear precipitant. No red flag features mentioned.",
  "suggested_documentation": "",
  "defensible_note": "35-year-old with low back pain following prolonged sitting. Analgesic given. Review advised — return earlier if worsening, radiating pain, or weakness in legs."
}

INPUT: 4 yr fever playful eating well pcm discharge
OUTPUT:
{
  "classification": "SAFE",
  "missing_anchors": [],
  "reasoning": "Reassuring clinical picture in a febrile child. Active, feeding well — low risk for serious bacterial infection.",
  "suggested_documentation": "",
  "defensible_note": "4-year-old with fever, playful and tolerating feeds well. Paracetamol administered. Advised return if fever persists beyond 3 days, child becomes lethargic, or rash develops."
}

INPUT: 4 yr repeated vomiting sleepy after treatment discharged
OUTPUT:
{
  "classification": "BORDERLINE",
  "missing_anchors": ["alertness level not objectively described", "hydration status not documented", "return precautions not recorded"],
  "reasoning": "Post-treatment drowsiness in a child requires documented confirmation of arousability and adequate hydration before safe discharge.",
  "suggested_documentation": "Adding that the child was arousable, tolerating sips, and documenting clear return precautions would make this defensible.",
  "defensible_note": "4-year-old with repeated vomiting treated with fluids. Child drowsy but arousable post-treatment. Discharged — parents advised urgent return if child cannot be roused, vomiting resumes, or oral intake remains poor."
}

INPUT: 20 yr contact lens irritation right eye 1 day moxifloxacin ketorolac
OUTPUT:
{
  "classification": "SAFE",
  "missing_anchors": [],
  "reasoning": "Minor ophthalmic complaint in a contact lens wearer. Low risk presentation.",
  "suggested_documentation": "",
  "defensible_note": "20-year-old contact lens user with right eye irritation for 1 day. Moxifloxacin and ketorolac prescribed. Advised review if symptoms worsen, vision changes, or pain increases."
}

INPUT: 65 yr female dm htn joint pain painkillers review 2 weeks
OUTPUT:
{
  "classification": "SAFE",
  "missing_anchors": [],
  "reasoning": "Chronic musculoskeletal complaint in a known diabetic-hypertensive patient. Routine follow-up appropriate.",
  "suggested_documentation": "",
  "defensible_note": "65-year-old female with diabetes and hypertension presenting with joint pain. Symptomatic analgesia given. Review in 2 weeks or earlier if symptoms worsen significantly."
}
"""


# ============================================================
# REQUEST MODEL
# ============================================================

class CaseInput(BaseModel):
    email: str
    note: str


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post("/analyze")
def analyze_case(data: CaseInput):

    # Input guards
    note = data.note.strip()
    if len(note) < 5:
        return {"error": "Note is too short to analyze."}
    if len(note) > 3000:
        return {"error": "Note is too long. Please keep the note under 3000 characters."}

    # Subscription check
    if not check_access(data.email):
        return {"error": "No active subscription"}

    # Rule engine
    rule_result = rule_classify(note)
    forced_class = rule_result["classification"]
    rule_flags   = rule_result["rule_flags"]
    pending      = rule_result["pending"]

    # Build context block for AI so its language is coherent with rule decision
    context_lines = []

    if forced_class:
        context_lines.append(f"RULE ENGINE CLASSIFICATION: {forced_class}")
        context_lines.append("Your classification field MUST match this exactly.")

    if rule_flags:
        context_lines.append(f"TRIGGERED RISK FLAGS: {', '.join(rule_flags)}")
        context_lines.append("Your reasoning and missing_anchors must address these flags.")

    if pending:
        context_lines.append(
            "PENDING INVESTIGATIONS (advised but NOT completed): " + "; ".join(pending)
        )
        context_lines.append(
            "CRITICAL: Do NOT state results for these investigations. "
            "Write them as pending/awaited in the defensible_note."
        )

    if context_lines:
        context_block = "\n\n[SYSTEM CONTEXT — NOT PART OF CLINICAL NOTE]\n" + "\n".join(context_lines)
        user_message = note + context_block
    else:
        user_message = note

    # Call AI
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    # Rule engine always wins on classification
    if forced_class:
        result["classification"] = forced_class

    # Append pending investigations to missing_anchors if AI missed them
    if pending:
        existing_anchors = result.get("missing_anchors", [])
        for p in pending:
            if not any(p[:15].lower() in a.lower() for a in existing_anchors):
                existing_anchors.append(p)
        result["missing_anchors"] = existing_anchors

    return result
