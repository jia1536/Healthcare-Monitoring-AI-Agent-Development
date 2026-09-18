"""
Patient-facing dashboard for the Healthcare Monitoring AI Agent (Track B).

Run with:
    streamlit run streamlit_app.py

Requires GROQ_API_KEY set in the environment (or a .env file) for the chat
tab, which invokes the multi-agent LangGraph workflow. The Medications,
Fitness, Goals, and Alerts tabs work directly against the local SQLite store
and don't require an API key.
"""

from datetime import date

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from App.db import database as db
from App.tools.analytics_tools import build_health_report, detect_health_risks
from App.auth import require_login, logout_button

load_dotenv()
# Streamlit Community Cloud stores secrets in st.secrets. Mirror the Groq key
# into the environment so the agent layer can use the same configuration
# locally (.env) and in the cloud.
try:
    if not os.getenv("GROQ_API_KEY") and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="Healthcare Monitoring AI Agent", page_icon="🩺", layout="wide")
db.init_db()

user = require_login()
role = user["role"]
can_edit = role == "patient"          # only patients log their own doses/entries/goals
can_chat = role in ("patient", "doctor")  # caregivers get view-only monitoring, not the chat agent


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🩺 " + {"patient": "Patient", "doctor": "Doctor", "caregiver": "Caregiver"}[role])
st.sidebar.caption(f"Signed in as **{user['username']}**")
logout_button()

if role == "doctor":
    # Doctors can review any patient's record.
    patients = db.list_patients()
    patient_options = {f"{p['name']} (#{p['id']})": p["id"] for p in patients}

    with st.sidebar.expander("Add new patient", expanded=(len(patients) == 0)):
        new_name = st.text_input("Name", key="new_patient_name")
        new_age = st.number_input("Age", min_value=0, max_value=120, value=30, key="new_patient_age")
        new_gender = st.selectbox("Gender", ["", "Female", "Male", "Other"], key="new_patient_gender")
        if st.button("Create patient"):
            if new_name.strip():
                new_id = db.create_patient(new_name.strip(), int(new_age), new_gender or None)
                st.success(f"Created patient '{new_name}' (#{new_id})")
                st.rerun()
            else:
                st.warning("Enter a name first.")

    if not patient_options:
        st.sidebar.info("Add a patient to get started.")
        st.stop()

    selected_label = st.sidebar.selectbox("Viewing patient", list(patient_options.keys()))
    patient_id = patient_options[selected_label]
else:
    # Patient and caregiver accounts are locked to the record they linked at login.
    patient_id = user["linked_patient_id"]
    if patient_id is None:
        st.sidebar.error("This account isn't linked to a patient record. Log out and sign in again.")
        st.stop()
    patient_record = db.get_patient(patient_id)
    st.sidebar.markdown(f"**Patient:** {patient_record['name'] if patient_record else patient_id}")

st.sidebar.markdown("---")
alerts = db.list_alerts(patient_id, unacknowledged_only=True)
if alerts:
    st.sidebar.subheader(f"⚠️ Alerts ({len(alerts)})")
    for a in alerts[:5]:
        icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}.get(a["severity"], "⚪")
        st.sidebar.markdown(f"{icon} {a['message']}")
else:
    st.sidebar.success("No active alerts")

st.title("Healthcare Monitoring AI Agent")
st.caption("Track B — multi-agent demo. Informational only; not a substitute for professional medical advice.")

if can_chat:
    tab_chat, tab_meds, tab_fitness, tab_goals, tab_report = st.tabs(
        ["💬 Chat", "💊 Medications", "🏃 Fitness", "🎯 Goals", "📄 Health Report"]
    )
else:
    tab_chat = None
    tab_meds, tab_fitness, tab_goals, tab_report = st.tabs(
        ["💊 Medications", "🏃 Fitness", "🎯 Goals", "📄 Health Report"]
    )
    st.sidebar.caption("Caregiver view: monitoring only — chat and logging are reserved for the patient's own account.")


# ------------------------------------------------------------------- chat
if can_chat:
    with tab_chat:
        st.subheader("Ask the assistant")
        st.caption("Routes automatically to the medical-info, medication, or fitness agent based on your question.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg_role, content in st.session_state.chat_history:
            with st.chat_message(msg_role):
                st.write(content)

        user_query = st.chat_input("Ask about symptoms, your medications, or your fitness trends...")
        if user_query:
            st.session_state.chat_history.append(("user", user_query))
            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        from workflow import ExecuteWorkflow
                        if "workflow" not in st.session_state:
                            st.session_state.workflow = ExecuteWorkflow()
                        result = st.session_state.workflow.run_workflow(user_query, patient_id=patient_id)
                        answer = result["messages"][-1].content
                    except Exception as e:
                        answer = (f"Couldn't reach the agent workflow ({e}). "
                                  "Check that GROQ_API_KEY is set in your .env file.")
                    st.write(answer)
                    st.session_state.chat_history.append(("assistant", answer))


# ------------------------------------------------------------- medications
with tab_meds:
    st.subheader("Medications")
    if not can_edit:
        st.caption("View only — dose logging is done from the patient's own account.")

    if can_edit:
        with st.form("add_med_form", clear_on_submit=True):
            cols = st.columns(4)
            m_name = cols[0].text_input("Name")
            m_dosage = cols[1].text_input("Dosage", placeholder="e.g. 500mg")
            m_freq = cols[2].number_input("Times/day", min_value=1, max_value=6, value=1)
            m_times = cols[3].text_input("Reminder times", placeholder="08:00,20:00")
            if st.form_submit_button("Add medication") and m_name.strip():
                db.add_medication(patient_id, m_name.strip(), m_dosage, int(m_freq), m_times)
                st.success(f"Added {m_name}")
                st.rerun()

    meds = db.list_medications(patient_id)
    if meds:
        for m in meds:
            with st.container(border=True):
                if can_edit:
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
                else:
                    c1, c2, c3 = st.columns([3, 2, 2])
                c1.markdown(f"**{m['name']}** — {m['dosage']}")
                c2.caption(f"{m['frequency_per_day']}x/day")
                c3.caption(m["times"] or "no times set")
                if can_edit:
                    b1, b2, b3 = c4.columns(3)
                    if b1.button("Taken", key=f"taken_{m['id']}"):
                        db.log_medication_dose(m["id"], "taken")
                        st.rerun()
                    if b2.button("Missed", key=f"missed_{m['id']}"):
                        db.log_medication_dose(m["id"], "missed")
                        st.rerun()
                    if b3.button("Skip", key=f"skip_{m['id']}"):
                        db.log_medication_dose(m["id"], "skipped")
                        st.rerun()

        rate = db.get_adherence_rate(patient_id)
        if rate is not None:
            st.metric("Adherence rate", f"{rate*100:.0f}%")
    else:
        st.info("No medications added yet.")


# ----------------------------------------------------------------- fitness
with tab_fitness:
    st.subheader("Fitness & sleep log")
    if not can_edit:
        st.caption("View only — fitness entries are logged from the patient's own account.")

    if can_edit:
        with st.form("fitness_form", clear_on_submit=True):
            cols = st.columns(5)
            f_date = cols[0].date_input("Date", value=date.today())
            f_steps = cols[1].number_input("Steps", min_value=0, value=0)
            f_cal = cols[2].number_input("Calories", min_value=0, value=0)
            f_sleep = cols[3].number_input("Sleep (h)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            f_hr = cols[4].number_input("Avg heart rate", min_value=0, value=0)
            if st.form_submit_button("Log entry"):
                db.log_fitness_entry(patient_id, f_date.isoformat(), int(f_steps), int(f_cal), float(f_sleep), int(f_hr))
                st.success(f"Logged entry for {f_date}")
                st.rerun()

    history = db.get_fitness_history(patient_id, limit=30)
    if history:
        df = pd.DataFrame([dict(r) for r in reversed(history)])
        df["log_date"] = pd.to_datetime(df["log_date"])
        df = df.set_index("log_date")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Steps**")
            st.line_chart(df["steps"])
        with c2:
            st.markdown("**Sleep (hours)**")
            st.line_chart(df["sleep_hours"])
    else:
        st.info("No fitness entries logged yet.")


# ------------------------------------------------------------------- goals
with tab_goals:
    st.subheader("Wellness goals")
    if not can_edit:
        st.caption("View only — goals are set from the patient's own account.")

    if can_edit:
        with st.form("goal_form", clear_on_submit=True):
            cols = st.columns(3)
            g_type = cols[0].selectbox("Goal type", ["daily_steps", "sleep_hours"])
            g_target = cols[1].number_input("Target value", min_value=0.0, value=8000.0)
            g_target_date = cols[2].date_input("Target date (optional)", value=None)
            if st.form_submit_button("Set goal"):
                db.set_health_goal(patient_id, g_type, g_target,
                                    g_target_date.isoformat() if g_target_date else None)
                st.success("Goal set")
                st.rerun()

    goals = db.list_health_goals(patient_id)
    if goals:
        for g in goals:
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{g['goal_type']}** — target {g['target_value']} "
                            f"(status: {g['status']}, target date: {g['target_date'] or 'n/a'})")
                if can_edit and g["status"] == "active" and c2.button("Mark achieved", key=f"achieve_{g['id']}"):
                    db.update_goal_status(g["id"], "achieved")
                    st.rerun()
    else:
        st.info("No goals set yet.")


# ------------------------------------------------------------------ report
with tab_report:
    st.subheader("Generated health report")
    if st.button("Regenerate report"):
        detect_health_risks(patient_id)
        st.rerun()

    reports = db.list_health_reports(patient_id, limit=1)
    if not reports:
        report_text = build_health_report(patient_id)
    else:
        report_text = reports[0]["report_text"]

    st.markdown(report_text)
    st.download_button("Download report (.md)", report_text,
                        file_name=f"health_report_patient_{patient_id}.md")
