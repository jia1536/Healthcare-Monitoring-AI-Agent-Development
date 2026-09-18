"""
Cross-domain analytics used at the end of every workflow run:
  - detect_health_risks: rule-based + trend-based risk flags, written to the
    `alerts` table (the "real-time monitoring with alert systems" milestone).
  - build_health_report: aggregates medication + fitness + goal state into a
    single text report, saved to `health_reports` (the "automated health
    report generation" milestone).

These are plain Python functions (not LangChain @tools) because they run
deterministically as part of the graph's `finalize` node rather than being
something the LLM decides whether to call.
"""

from datetime import datetime

import numpy as np

from App.db import database as db

ADHERENCE_RISK_THRESHOLD = 0.7
MISSED_STREAK_RISK_THRESHOLD = 3
SLEEP_RISK_HOURS = 5.5
STEPS_DECLINE_SLOPE_THRESHOLD = -300.0  # steps/day lost per day, over the window


def _trend_slope(values):
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    y = np.array(values, dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def detect_health_risks(patient_id: int) -> list[dict]:
    """Run rule-based + trend-based checks and persist any new alerts.
    Returns the list of alerts raised on this run (may be empty)."""
    raised = []

    # --- medication adherence risk ---
    rate = db.get_adherence_rate(patient_id)
    if rate is not None and rate < ADHERENCE_RISK_THRESHOLD:
        msg = f"Medication adherence is {rate * 100:.0f}%, below the {ADHERENCE_RISK_THRESHOLD*100:.0f}% target."
        db.create_alert(patient_id, "low_adherence", msg, severity="warning")
        raised.append({"type": "low_adherence", "message": msg, "severity": "warning"})

    streak = db.recent_missed_streak(patient_id)
    if streak >= MISSED_STREAK_RISK_THRESHOLD:
        msg = f"{streak} consecutive missed doses detected."
        db.create_alert(patient_id, "missed_streak", msg, severity="critical")
        raised.append({"type": "missed_streak", "message": msg, "severity": "critical"})

    # --- fitness / sleep trend risk ---
    history = list(reversed(db.get_fitness_history(patient_id, limit=7)))
    sleep_vals = [r["sleep_hours"] for r in history if r["sleep_hours"] is not None]
    if sleep_vals and (sum(sleep_vals) / len(sleep_vals)) < SLEEP_RISK_HOURS:
        avg = sum(sleep_vals) / len(sleep_vals)
        msg = f"Average sleep over recent logs is {avg:.1f}h, below the recommended ~7h."
        db.create_alert(patient_id, "low_sleep", msg, severity="info")
        raised.append({"type": "low_sleep", "message": msg, "severity": "info"})

    steps_vals = [r["steps"] for r in history if r["steps"] is not None]
    slope = _trend_slope(steps_vals)
    if steps_vals and slope < STEPS_DECLINE_SLOPE_THRESHOLD:
        msg = f"Daily step count is trending down (slope {slope:.0f} steps/day over the recent window)."
        db.create_alert(patient_id, "activity_decline", msg, severity="info")
        raised.append({"type": "activity_decline", "message": msg, "severity": "info"})

    return raised


def build_health_report(patient_id: int, latest_agent_response: str = "") -> str:
    """Aggregate medication, fitness, goal, and alert state into one report
    and persist it. `latest_agent_response` is the current turn's LLM answer,
    included as context, not overwritten."""
    patient = db.get_patient(patient_id)
    meds = db.list_medications(patient_id)
    rate = db.get_adherence_rate(patient_id)
    history = db.get_fitness_history(patient_id, limit=7)
    goals = db.list_health_goals(patient_id)
    alerts = db.list_alerts(patient_id, unacknowledged_only=True)

    lines = [f"# Health Report — {patient['name'] if patient else patient_id}",
             f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_", ""]

    if latest_agent_response:
        lines += ["## Latest response", latest_agent_response, ""]

    lines.append("## Medications")
    if meds:
        for m in meds:
            lines.append(f"- {m['name']} ({m['dosage']}), {m['frequency_per_day']}x/day")
        lines.append(f"- Adherence rate: {rate*100:.0f}%" if rate is not None else "- Adherence rate: no doses logged yet")
    else:
        lines.append("- None on file")

    lines.append("\n## Recent Fitness (last 7 logs)")
    if history:
        for r in reversed(history):
            lines.append(f"- {r['log_date']}: {r['steps'] or '-'} steps, "
                          f"{r['sleep_hours'] or '-'}h sleep, HR avg {r['heart_rate_avg'] or '-'}")
    else:
        lines.append("- No fitness data logged yet")

    lines.append("\n## Goals")
    if goals:
        for g in goals:
            lines.append(f"- {g['goal_type']}: target {g['target_value']} ({g['status']})")
    else:
        lines.append("- No goals set")

    lines.append("\n## Active Alerts")
    if alerts:
        for a in alerts:
            lines.append(f"- [{a['severity'].upper()}] {a['message']}")
    else:
        lines.append("- None")

    lines.append("\n---\n_This report is for personal tracking only and is not a substitute "
                  "for professional medical advice._")

    report_text = "\n".join(lines)
    db.save_health_report(patient_id, report_text)
    return report_text
