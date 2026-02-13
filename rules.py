import re

def get_age(text):
    match = re.search(r'(\d+)\s*yr', text)
    if match:
        return int(match.group(1))
    return None


def analyze_note(note):

    flags = []
    text = note.lower()
    age = get_age(text)

    # ---------------- SHOCK / SEPSIS ----------------
    if "bp" in text and ("norad" in text or "vasopressor" in text) and "fluid" not in text:
        flags.append("Shock management without fluid reasoning documentation")

    # ---------------- HEAD INJURY ----------------
    if "head injury" in text and "ct" not in text:
        flags.append("Head injury without imaging explanation")

    # ---------------- CHEST PAIN ----------------
    if "chest pain" in text and "ecg" not in text:
        flags.append("Chest pain without cardiac evaluation documentation")

    # ---------------- DISCHARGE RISK ----------------
    if "discharged" in text and ("pain" in text or "vomiting" in text or "fever" in text):
        flags.append("Symptomatic patient discharged without observation reasoning")

    # ---------------- RENAL RISK DRUG ----------------
    if ("creatinine" in text or "cr" in text) and ("gentamicin" in text or "gentamycin" in text):
        flags.append("Nephrotoxic antibiotic in renal impairment — justification needed")

    # ---------------- DIABETIC FOOT INFECTION ----------------
    if "diabetic" in text and ("foot" in text or "ulcer" in text or "redness" in text or "swelling" in text):
        if "pulses" not in text and "systemic" not in text and "sepsis" not in text:
            flags.append("Diabetic infection — severity assessment (systemic signs / vascular status) not documented")

    # ---------------- PEDIATRIC FEVER ----------------
    if age is not None and age <= 12 and "fever" in text:
        if "activity" not in text and "feeding" not in text and "urine" not in text:
            flags.append("Child fever discharge — hydration/activity status not documented")

    # ---------------- TRAUMA WITHOUT XRAY ----------------
    if ("injury" in text or "fall" in text or "skid" in text or "swelling" in text):
        if "xray" not in text and "fracture" not in text:
            flags.append("Trauma case — fracture exclusion not documented")

    return flags
