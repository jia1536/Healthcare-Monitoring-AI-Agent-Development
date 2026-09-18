"""
Lightweight role-based access for the Streamlit dashboard.

This is a username + role login, no passwords — the goal is to gate what
the UI shows and lets you do (Patient / Doctor / Caregiver), not to provide
a real security boundary against a malicious actor. See docs/architecture.md
for why that's the right scope for a fast student-project pass, and what a
production version would need (password hashing, sessions, HTTPS-only
cookies) that this deliberately skips.

Usage (from streamlit_app.py):
    from App.auth import require_login
    user = require_login()   # blocks with a login form until logged in
    # user = {"username": ..., "role": "patient"|"doctor"|"caregiver",
    #         "linked_patient_id": int | None}
"""

import streamlit as st

from App.db import database as db

ROLES = ["patient", "doctor", "caregiver"]
ROLE_LABELS = {"patient": "Patient", "doctor": "Doctor", "caregiver": "Caregiver"}


def _login_form():
    st.title("🩺 Healthcare Monitoring AI Agent")
    st.subheader("Sign in")
    st.caption("Username + role only — no password. This gates what you can see and do, "
               "not a production security boundary (see docs/architecture.md).")

    username = st.text_input("Username", key="login_username")
    existing = db.get_user(username.strip()) if username.strip() else None

    if existing:
        st.info(f"Welcome back — logging in as **{ROLE_LABELS[existing['role']]}**.")
        if st.button("Log in", type="primary"):
            st.session_state.user = {
                "username": existing["username"],
                "role": existing["role"],
                "linked_patient_id": existing["linked_patient_id"],
            }
            st.rerun()
        return

    role = st.selectbox("Role", ROLES, format_func=lambda r: ROLE_LABELS[r], key="login_role")
    patients = db.list_patients()
    patient_options = {f"{p['name']} (#{p['id']})": p["id"] for p in patients}

    linked_patient_id = None

    if role == "patient":
        st.markdown("**Link to your patient record**")
        choice = st.radio("", ["Create a new patient record", "Use an existing one"],
                           horizontal=True, label_visibility="collapsed")
        if choice == "Create a new patient record":
            name = st.text_input("Full name", key="new_patient_name_login")
            age = st.number_input("Age", min_value=0, max_value=120, value=30, key="new_patient_age_login")
            gender = st.selectbox("Gender", ["", "Female", "Male", "Other"], key="new_patient_gender_login")
        elif patient_options:
            picked = st.selectbox("Existing patient record", list(patient_options.keys()))
            linked_patient_id = patient_options[picked]
        else:
            st.warning("No existing patient records yet — create a new one instead.")

    elif role == "caregiver":
        st.markdown("**Which patient are you caring for?**")
        if not patient_options:
            st.warning("No patient records exist yet. Ask the patient to sign in and create "
                       "their record first, then sign in here as their caregiver.")
        else:
            picked = st.selectbox("Patient", list(patient_options.keys()))
            linked_patient_id = patient_options[picked]

    else:  # doctor
        st.caption("Doctors can view any patient's record from the dashboard once signed in.")

    disabled = not username.strip() or (
        role == "caregiver" and not patient_options
    )

    if st.button("Create account & log in", type="primary", disabled=disabled):
        if role == "patient" and linked_patient_id is None:
            name = st.session_state.get("new_patient_name_login", "").strip()
            if not name:
                st.warning("Enter a name for the new patient record.")
                return
            age = st.session_state.get("new_patient_age_login", 30)
            gender = st.session_state.get("new_patient_gender_login") or None
            linked_patient_id = db.create_patient(name, int(age), gender)

        db.create_user(username.strip(), role, linked_patient_id)
        st.session_state.user = {
            "username": username.strip(),
            "role": role,
            "linked_patient_id": linked_patient_id,
        }
        st.rerun()


def require_login() -> dict:
    """Blocks (via st.stop after rendering a login form) until a user is
    logged in this session. Returns the logged-in user dict otherwise."""
    if "user" not in st.session_state:
        _login_form()
        st.stop()
    return st.session_state.user


def logout_button():
    if st.sidebar.button("Log out"):
        del st.session_state["user"]
        st.rerun()
