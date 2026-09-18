"""
Tools available to the MedicationAgent.

Backed by the SQLite health data store (`App/db/database.py`). All tools
take an explicit `patient_id` so the agent stays stateless between calls —
the graph state carries the active patient's id, not the tools.
"""

from langchain_core.tools import tool

from App.db import database as db

# A small, illustrative set of well-known interacting drug/class pairs.
# This is NOT a substitute for a real drug-interaction database (e.g. RxNorm /
# DDInter) and is explicitly scoped as "basic interaction awareness" per the
# project's medical-safety requirements — never treated as exhaustive.
KNOWN_INTERACTIONS = [
    ({"warfarin"}, {"aspirin", "ibuprofen", "naproxen"},
     "Increased bleeding risk when combined with NSAIDs/antiplatelets."),
    ({"metformin"}, {"alcohol"},
     "Combining with alcohol raises the risk of lactic acidosis."),
    ({"lisinopril", "enalapril", "ramipril"}, {"potassium", "spironolactone"},
     "ACE inhibitors + potassium-sparing agents can cause dangerous hyperkalemia."),
    ({"sertraline", "fluoxetine", "citalopram"}, {"tramadol", "st john's wort"},
     "Combining SSRIs with these increases serotonin syndrome risk."),
    ({"simvastatin", "atorvastatin"}, {"grapefruit"},
     "Grapefruit can raise statin levels and side-effect risk."),
]


def _check_interactions(med_names: list[str]) -> list[str]:
    lowered = {m.strip().lower() for m in med_names}
    findings = []
    for group_a, group_b, note in KNOWN_INTERACTIONS:
        if lowered & group_a and lowered & group_b:
            findings.append(note)
    return findings


@tool
def add_medication(patient_id: int, name: str, dosage: str, frequency_per_day: int,
                    times: str = "") -> str:
    """Add a new medication to a patient's tracked medication list.
    `times` is an optional comma-separated list of reminder times, e.g. '08:00,20:00'."""
    med_id = db.add_medication(patient_id, name, dosage, frequency_per_day, times)
    return f"Added medication '{name}' ({dosage}, {frequency_per_day}x/day) with id {med_id}."


@tool
def log_medication_dose(medication_id: int, status: str = "taken") -> str:
    """Log a dose event for a medication. status must be one of:
    'taken', 'missed', 'skipped'."""
    if status not in ("taken", "missed", "skipped"):
        return "Invalid status. Use 'taken', 'missed', or 'skipped'."
    db.log_medication_dose(medication_id, status)
    return f"Logged dose as '{status}' for medication id {medication_id}."


@tool
def check_medication_adherence(patient_id: int) -> str:
    """Check a patient's overall medication adherence rate and current missed-dose
    streak. Use this to answer questions about how well a patient is following
    their medication schedule."""
    rate = db.get_adherence_rate(patient_id)
    streak = db.recent_missed_streak(patient_id)
    if rate is None:
        return "No medication doses have been logged yet for this patient."
    return (
        f"Adherence rate: {rate * 100:.1f}% of logged doses taken. "
        f"Current consecutive missed-dose streak: {streak}."
    )


@tool
def list_patient_medications(patient_id: int) -> str:
    """List a patient's currently active medications with dosage and schedule."""
    meds = db.list_medications(patient_id, active_only=True)
    if not meds:
        return "No active medications on file for this patient."
    lines = [f"- {m['name']} ({m['dosage']}), {m['frequency_per_day']}x/day, times: {m['times'] or 'not set'}"
              for m in meds]
    return "\n".join(lines)


@tool
def check_drug_interactions(patient_id: int) -> str:
    """Check a patient's active medications for known basic interaction concerns.
    This uses a small illustrative reference list, not a full clinical drug
    database — always advise the patient to confirm with a pharmacist or doctor."""
    meds = db.list_medications(patient_id, active_only=True)
    names = [m["name"] for m in meds]
    if len(names) < 2:
        return "Fewer than two active medications on file; no interaction check needed."
    findings = _check_interactions(names)
    if not findings:
        return (
            f"No known interactions found among tracked medications ({', '.join(names)}) "
            "in this basic reference list. This is not exhaustive — confirm with a pharmacist."
        )
    return "Potential interaction concerns found:\n" + "\n".join(f"- {f}" for f in findings)


TOOLS = [
    add_medication,
    log_medication_dose,
    check_medication_adherence,
    list_patient_medications,
    check_drug_interactions,
]
