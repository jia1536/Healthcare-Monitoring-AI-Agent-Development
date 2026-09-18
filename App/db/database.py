"""
Health data store for the healthcare monitoring platform (Track B).

SQLite is used deliberately instead of Postgres/Redis so the whole project
stays runnable with zero external infrastructure (`pip install` + run) while
still modeling a realistic multi-table health-record lifecycle:

    patients          -- one row per user of the platform
    medications       -- prescribed medications a patient is tracking
    medication_logs   -- one row per dose event (taken/missed/skipped)
    fitness_logs      -- daily fitness/activity/sleep entries
    health_goals      -- wellness goals with progress tracking
    alerts            -- system-generated risk/adherence alerts
    health_reports     -- saved snapshots of generated health reports

PRIVACY NOTE: this is a development database for a student project. It
stores only whatever sample/mock data the user enters locally in
`data/health_data.db` (gitignored). Never load real patient data into it.
See docs/architecture.md for the fuller privacy & compliance notes.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date
from typing import Optional

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "health_data.db",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    age         INTEGER,
    gender      TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS medications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL REFERENCES patients(id),
    name            TEXT NOT NULL,
    dosage          TEXT,
    frequency_per_day INTEGER NOT NULL DEFAULT 1,
    times           TEXT,             -- comma-separated HH:MM reminder times
    start_date      TEXT NOT NULL,
    active          INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS medication_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    medication_id   INTEGER NOT NULL REFERENCES medications(id),
    logged_at       TEXT NOT NULL,
    status          TEXT NOT NULL CHECK(status IN ('taken', 'missed', 'skipped'))
);

CREATE TABLE IF NOT EXISTS fitness_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL REFERENCES patients(id),
    log_date        TEXT NOT NULL,
    steps           INTEGER,
    calories        INTEGER,
    sleep_hours     REAL,
    heart_rate_avg  INTEGER,
    UNIQUE(patient_id, log_date)
);

CREATE TABLE IF NOT EXISTS health_goals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL REFERENCES patients(id),
    goal_type       TEXT NOT NULL,     -- e.g. 'daily_steps', 'sleep_hours', 'weight_kg'
    target_value    REAL NOT NULL,
    start_date      TEXT NOT NULL,
    target_date     TEXT,
    status          TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','achieved','abandoned'))
);

CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL REFERENCES patients(id),
    alert_type      TEXT NOT NULL,
    message         TEXT NOT NULL,
    severity        TEXT NOT NULL CHECK(severity IN ('info','warning','critical')),
    created_at      TEXT NOT NULL,
    acknowledged    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS health_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL REFERENCES patients(id),
    generated_at    TEXT NOT NULL,
    report_text     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT NOT NULL UNIQUE,
    role            TEXT NOT NULL CHECK(role IN ('patient','doctor','caregiver')),
    linked_patient_id INTEGER REFERENCES patients(id),  -- NULL for doctor (sees all patients)
    created_at      TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------- patients
def create_patient(name: str, age: Optional[int] = None, gender: Optional[str] = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO patients (name, age, gender, created_at) VALUES (?, ?, ?, ?)",
            (name, age, gender, datetime.now().isoformat()),
        )
        return cur.lastrowid


def list_patients():
    with get_conn() as conn:
        return conn.execute("SELECT * FROM patients ORDER BY id").fetchall()


def get_patient(patient_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()


# ------------------------------------------------------------- medications
def add_medication(patient_id: int, name: str, dosage: str, frequency_per_day: int,
                    times: str = "", start_date: Optional[str] = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO medications
               (patient_id, name, dosage, frequency_per_day, times, start_date, active)
               VALUES (?, ?, ?, ?, ?, ?, 1)""",
            (patient_id, name, dosage, frequency_per_day, times, start_date or date.today().isoformat()),
        )
        return cur.lastrowid


def list_medications(patient_id: int, active_only: bool = True):
    with get_conn() as conn:
        q = "SELECT * FROM medications WHERE patient_id = ?"
        if active_only:
            q += " AND active = 1"
        return conn.execute(q, (patient_id,)).fetchall()


def log_medication_dose(medication_id: int, status: str = "taken") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO medication_logs (medication_id, logged_at, status) VALUES (?, ?, ?)",
            (medication_id, datetime.now().isoformat(), status),
        )
        return cur.lastrowid


def get_adherence_rate(patient_id: int) -> Optional[float]:
    """Fraction of logged doses (across all active medications) marked 'taken'."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT ml.status FROM medication_logs ml
               JOIN medications m ON m.id = ml.medication_id
               WHERE m.patient_id = ?""",
            (patient_id,),
        ).fetchall()
    if not rows:
        return None
    taken = sum(1 for r in rows if r["status"] == "taken")
    return round(taken / len(rows), 3)


def recent_missed_streak(patient_id: int) -> int:
    """Count how many of the most recent consecutive dose logs were missed."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT ml.status FROM medication_logs ml
               JOIN medications m ON m.id = ml.medication_id
               WHERE m.patient_id = ?
               ORDER BY ml.logged_at DESC""",
            (patient_id,),
        ).fetchall()
    streak = 0
    for r in rows:
        if r["status"] == "missed":
            streak += 1
        else:
            break
    return streak


# --------------------------------------------------------------- fitness
def log_fitness_entry(patient_id: int, log_date: str, steps: int = None,
                       calories: int = None, sleep_hours: float = None,
                       heart_rate_avg: int = None) -> int:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO fitness_logs (patient_id, log_date, steps, calories, sleep_hours, heart_rate_avg)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(patient_id, log_date) DO UPDATE SET
                 steps=excluded.steps, calories=excluded.calories,
                 sleep_hours=excluded.sleep_hours, heart_rate_avg=excluded.heart_rate_avg""",
            (patient_id, log_date, steps, calories, sleep_hours, heart_rate_avg),
        )
        row = conn.execute(
            "SELECT id FROM fitness_logs WHERE patient_id=? AND log_date=?",
            (patient_id, log_date),
        ).fetchone()
        return row["id"]


def get_fitness_history(patient_id: int, limit: int = 30):
    with get_conn() as conn:
        return conn.execute(
            """SELECT * FROM fitness_logs WHERE patient_id = ?
               ORDER BY log_date DESC LIMIT ?""",
            (patient_id, limit),
        ).fetchall()


# ------------------------------------------------------------------ goals
def set_health_goal(patient_id: int, goal_type: str, target_value: float,
                     target_date: Optional[str] = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO health_goals (patient_id, goal_type, target_value, start_date, target_date, status)
               VALUES (?, ?, ?, ?, ?, 'active')""",
            (patient_id, goal_type, target_value, date.today().isoformat(), target_date),
        )
        return cur.lastrowid


def list_health_goals(patient_id: int, status: str = "active"):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM health_goals WHERE patient_id = ? AND status = ?",
            (patient_id, status),
        ).fetchall()


def update_goal_status(goal_id: int, status: str):
    with get_conn() as conn:
        conn.execute("UPDATE health_goals SET status = ? WHERE id = ?", (status, goal_id))


# ----------------------------------------------------------------- alerts
def create_alert(patient_id: int, alert_type: str, message: str, severity: str = "info") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO alerts (patient_id, alert_type, message, severity, created_at, acknowledged)
               VALUES (?, ?, ?, ?, ?, 0)""",
            (patient_id, alert_type, message, severity, datetime.now().isoformat()),
        )
        return cur.lastrowid


def list_alerts(patient_id: int, unacknowledged_only: bool = True):
    with get_conn() as conn:
        q = "SELECT * FROM alerts WHERE patient_id = ?"
        if unacknowledged_only:
            q += " AND acknowledged = 0"
        q += " ORDER BY created_at DESC"
        return conn.execute(q, (patient_id,)).fetchall()


def acknowledge_alert(alert_id: int):
    with get_conn() as conn:
        conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))


# ------------------------------------------------------------------ reports
def save_health_report(patient_id: int, report_text: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO health_reports (patient_id, generated_at, report_text) VALUES (?, ?, ?)",
            (patient_id, datetime.now().isoformat(), report_text),
        )
        return cur.lastrowid


def list_health_reports(patient_id: int, limit: int = 10):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM health_reports WHERE patient_id = ? ORDER BY generated_at DESC LIMIT ?",
            (patient_id, limit),
        ).fetchall()


# ------------------------------------------------------------------- users
# Lightweight role-based access, scoped for a fast student-project pass: a
# username + selected role, no passwords. This gates *what the UI shows and
# lets you do*, not a security boundary against a malicious actor — see
# docs/architecture.md for why that trade-off is the right one here.
def get_user(username: str):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def create_user(username: str, role: str, linked_patient_id: Optional[int] = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, role, linked_patient_id, created_at) VALUES (?, ?, ?, ?)",
            (username, role, linked_patient_id, datetime.now().isoformat()),
        )
        return cur.lastrowid
