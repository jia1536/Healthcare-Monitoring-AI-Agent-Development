"""
Tools available to the FitnessAgent.

Trend detection uses a simple linear-regression slope (numpy.polyfit) over
recent daily values — a lightweight, honest stand-in for the "ML-based
pattern recognition" milestone rather than an overclaimed black-box model.
"""

from datetime import date

import numpy as np
from langchain_core.tools import tool

from App.db import database as db


def _trend_slope(values: list[float]) -> float:
    """Fit a degree-1 polynomial (linear regression) over the sequence and
    return its slope. Positive = increasing trend, negative = declining."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    y = np.array(values, dtype=float)
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


@tool
def log_fitness_entry(patient_id: int, log_date: str, steps: int = None,
                       calories: int = None, sleep_hours: float = None,
                       heart_rate_avg: int = None) -> str:
    """Log or update a patient's fitness/activity entry for a given date
    (YYYY-MM-DD). Any field left out is not updated."""
    db.log_fitness_entry(patient_id, log_date, steps, calories, sleep_hours, heart_rate_avg)
    return f"Logged fitness entry for {log_date}."


@tool
def get_fitness_summary(patient_id: int, days: int = 7) -> str:
    """Summarize a patient's fitness data over the last N days: averages and
    the trend direction (increasing/declining/flat) for steps and sleep,
    computed via linear regression over the recent daily values."""
    rows = db.get_fitness_history(patient_id, limit=days)
    if not rows:
        return "No fitness data logged yet for this patient."

    rows = list(reversed(rows))  # oldest -> newest for trend direction
    steps = [r["steps"] for r in rows if r["steps"] is not None]
    sleep = [r["sleep_hours"] for r in rows if r["sleep_hours"] is not None]

    lines = [f"Fitness summary over last {len(rows)} logged day(s):"]
    if steps:
        avg_steps = sum(steps) / len(steps)
        slope = _trend_slope(steps)
        direction = "increasing" if slope > 5 else "declining" if slope < -5 else "flat"
        lines.append(f"- Steps: avg {avg_steps:.0f}/day, trend {direction} (slope {slope:.1f}/day)")
    if sleep:
        avg_sleep = sum(sleep) / len(sleep)
        slope = _trend_slope(sleep)
        direction = "increasing" if slope > 0.1 else "declining" if slope < -0.1 else "flat"
        lines.append(f"- Sleep: avg {avg_sleep:.1f}h/night, trend {direction} (slope {slope:.2f}h/day)")
    return "\n".join(lines)


@tool
def set_health_goal(patient_id: int, goal_type: str, target_value: float,
                     target_date: str = None) -> str:
    """Set a wellness goal for a patient, e.g. goal_type='daily_steps',
    target_value=8000. target_date is optional (YYYY-MM-DD)."""
    goal_id = db.set_health_goal(patient_id, goal_type, target_value, target_date)
    return f"Set goal '{goal_type}' with target {target_value} (id {goal_id})."


@tool
def check_goal_progress(patient_id: int) -> str:
    """Check progress on a patient's active health goals against their most
    recent fitness data."""
    goals = db.list_health_goals(patient_id, status="active")
    if not goals:
        return "No active health goals set for this patient."

    recent = db.get_fitness_history(patient_id, limit=7)
    lines = []
    for g in goals:
        latest_value = None
        if g["goal_type"] == "daily_steps" and recent:
            vals = [r["steps"] for r in recent if r["steps"] is not None]
            latest_value = (sum(vals) / len(vals)) if vals else None
        elif g["goal_type"] == "sleep_hours" and recent:
            vals = [r["sleep_hours"] for r in recent if r["sleep_hours"] is not None]
            latest_value = (sum(vals) / len(vals)) if vals else None

        if latest_value is None:
            lines.append(f"- {g['goal_type']}: target {g['target_value']}, no recent data to compare.")
        else:
            pct = min(100, round(latest_value / g["target_value"] * 100)) if g["target_value"] else 0
            lines.append(f"- {g['goal_type']}: target {g['target_value']}, "
                          f"recent avg {latest_value:.1f} ({pct}% of target)")
    return "\n".join(lines)


TOOLS = [
    log_fitness_entry,
    get_fitness_summary,
    set_health_goal,
    check_goal_progress,
]
