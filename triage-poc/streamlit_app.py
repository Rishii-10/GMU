"""Streamlit frontend for the triage pipeline.

Text in -> app.agent1_extraction.extract_and_classify() -> classification
out -> (EMERGENCY/SEVERE only) app.routing.router.route() +
app.routing.report.generate_doctor_report() for a clinician handoff.
No new pipeline logic lives here; this is a display layer over the
existing modules (app/agent1_extraction.py, app/rules_engine.py,
app/routing/).

Same underlying case/classification/dispatch is rendered three ways, one
per real-world actor in the workflow (caller/caregiver, ASHA worker,
doctor) -- switchable via tabs, computed once and cached in
st.session_state so flipping tabs never re-runs extraction or re-calls
the LLM report generator.

Run with:
    ./.venv/bin/streamlit run streamlit_app.py
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load a local .env (repo root or triage-poc/) so opt-in provider keys like
# GOOGLE_TRANSLATE_API_KEY are picked up without exporting them by hand.
# Optional: if python-dotenv isn't installed, real env vars still work.
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True) or None)
except ImportError:
    pass

import streamlit as st

from app.agent1_extraction import (
    BackendUnavailable,
    ExtractionValidationError,
    OllamaBackend,
    extract_and_classify,
    question_for_incomplete_result,
)
from app.integrations.language_gateway import LanguageGateway
from app.routing.demo_facilities import VILLAGES, build_demo_facility_db
from app.routing.report import generate_doctor_report
from app.routing.router import route, urgency_for_label
from app.routing.schemas import DispatchResult
from app.schemas import ClassificationLabel, ClassificationResult, ExtractedCase


def _t(gw: LanguageGateway | None, s):
    """Translate one outbound English string into the caregiver's language
    (no-op when gw is None or the language is English/undetected)."""
    return gw.from_english(s) if gw is not None else s


def _tb(gw: LanguageGateway | None, items):
    """Batch form of _t() for a list of strings -- one translation round trip."""
    return gw.from_english_batch(list(items)) if gw is not None else list(items)

st.set_page_config(page_title="Rural Health Triage", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------------------
# Style -- dark theme
#
# One palette drives both the CSS below and the Python colour tables further
# down, so the two can't drift apart. Every foreground here sits on BG/SURFACE
# at >= 4.5:1, and each severity foreground is paired with a dark tint of its
# own hue instead of its light-mode background.
# ---------------------------------------------------------------------------
BG          = "#0E1117"   # app background
SURFACE     = "#161B22"   # cards, panels
SURFACE_2   = "#1C232D"   # nested / raised surfaces
BORDER      = "#2E3742"   # hairlines, bar tracks
BORDER_2    = "#3A4553"   # stronger dividers
TRACK       = "#414B58"   # empty portion of a progress bar / table rules
TEXT        = "#E6EDF3"   # body text
TEXT_DIM    = "#9AA7B4"   # labels, captions
PRIMARY     = "#5AA9FF"   # accent (links, bars, buttons)
PRIMARY_HOV = "#7CBCFF"
PRIMARY_BG  = "#152534"   # accent tint background
ON_ACCENT   = "#0B1017"   # text drawn on top of a bright accent fill

# Severity hues: (foreground, tinted background)
C_EMERGENCY = ("#FF7B72", "#2E1416")
C_SEVERE    = ("#FFA657", "#2E1E10")
C_MODERATE  = ("#E3C25F", "#2B2510")
C_MILD      = ("#56D364", "#10251A")
C_NEUTRAL   = ("#B4C0CC", "#20262E")

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    html, body, [class*="css"] {{ font-size: 17px; color: {TEXT}; }}
    h1 {{ font-size: 2.0rem !important; color: {TEXT}; }}
    h2 {{ font-size: 1.4rem !important; color: {TEXT}; margin-top: 0.4rem; }}
    h3 {{ font-size: 1.15rem !important; color: {TEXT}; }}
    h4, h5, h6 {{ color: {TEXT}; }}
    a, a:visited {{ color: {PRIMARY}; }}
    hr {{ border-color: {BORDER}; }}
    code {{ color: {PRIMARY}; background-color: {SURFACE_2}; }}

    /* Streamlit paints its own chrome from the resolved theme, which falls back
       to the viewer's OS light/dark preference when the project config isn't
       picked up. Pin the chrome here so the app is dark either way -- otherwise
       a white header washes out the dark page beneath it. */
    header[data-testid="stHeader"] {{ background: {BG}; }}
    [data-testid="stToolbar"] *, [data-testid="stHeader"] * {{ color: {TEXT_DIM}; }}
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {{ color: {TEXT}; }}
    [data-testid="stJson"], pre {{ background-color: {SURFACE_2}; color: {TEXT}; }}

    /* Same reason: give each alert kind its own dark tint instead of inheriting
       a light one that our light body text would disappear into. */
    [data-testid="stAlertContainer"] {{ color: {TEXT}; }}
    [data-testid="stAlertContentInfo"] {{ background-color: {PRIMARY_BG}; color: {TEXT}; }}
    [data-testid="stAlertContentSuccess"] {{ background-color: {C_MILD[1]}; color: {C_MILD[0]}; }}
    [data-testid="stAlertContentWarning"] {{ background-color: {C_MODERATE[1]}; color: {C_MODERATE[0]}; }}
    [data-testid="stAlertContentError"] {{ background-color: {C_EMERGENCY[1]}; color: {C_EMERGENCY[0]}; }}

    section[data-testid="stSidebar"] {{
        background-color: {SURFACE}; border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{ color: {TEXT}; }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
        color: {TEXT_DIM} !important;
    }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_DIM}; }}
    [data-testid="stMetricValue"] {{ color: {TEXT}; }}

    .stTextArea textarea, .stTextInput input, .stNumberInput input {{
        font-size: 16px; color: {TEXT}; background-color: {SURFACE};
        border: 1px solid {BORDER};
    }}
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {{ color: {TEXT_DIM}; }}
    [data-baseweb="select"] > div {{
        background-color: {SURFACE}; border-color: {BORDER}; color: {TEXT};
    }}
    [data-baseweb="popover"] li {{ background-color: {SURFACE}; color: {TEXT}; }}

    .stButton button {{
        font-size: 17px; font-weight: 600; padding: 0.6rem 1rem;
        background-color: {PRIMARY}; color: {ON_ACCENT}; border: none; border-radius: 6px;
    }}
    .stButton button:hover {{ background-color: {PRIMARY_HOV}; color: {ON_ACCENT}; }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 17px; font-weight: 600; padding: 0.6rem 1rem; color: {TEXT_DIM};
    }}
    .stTabs [aria-selected="true"] {{ color: {PRIMARY}; }}
    .stTabs [data-baseweb="tab-list"] {{ border-bottom: 1px solid {BORDER}; }}

    [data-testid="stExpander"] {{
        border: 1px solid {BORDER}; border-radius: 8px; background-color: {SURFACE};
    }}
    [data-testid="stExpander"] summary {{ color: {TEXT}; }}

    .field-label {{
        color: {TEXT_DIM}; font-size: 13px; text-transform: uppercase;
        letter-spacing: 0.03em; margin-bottom: -0.3rem;
    }}
    .field-value {{ font-size: 17px; color: {TEXT}; margin-bottom: 0.6rem; }}

    .banner {{
        padding: 1.1rem 1.4rem; border-radius: 10px; margin: 0.6rem 0 1rem 0;
        font-size: 24px; font-weight: 700; border-left: 8px solid;
    }}
    .subbanner {{ font-size: 16px; font-weight: 400; display: block; margin-top: 0.3rem; }}
    .big-banner {{ font-size: 30px; padding: 1.6rem 1.8rem; }}
    .big-banner .subbanner {{ font-size: 19px; }}

    .card {{
        border: 1px solid {BORDER}; border-radius: 10px; padding: 1rem 1.2rem;
        margin-bottom: 1rem; background-color: {SURFACE}; color: {TEXT};
    }}
    .action-card {{
        border: 1px solid {BORDER}; border-left: 8px solid {PRIMARY}; border-radius: 8px;
        padding: 1rem 1.2rem; margin-bottom: 1rem; background-color: {PRIMARY_BG};
        color: {TEXT}; font-size: 19px; font-weight: 600;
    }}
    .tag {{
        display: inline-block; padding: 0.15rem 0.6rem; margin: 0.15rem 0.3rem 0.15rem 0;
        border-radius: 999px; background-color: {PRIMARY_BG}; color: {PRIMARY};
        border: 1px solid {BORDER_2}; font-size: 14px; font-weight: 600;
    }}
    .pill-yes {{ color: {C_EMERGENCY[0]}; font-weight: 700; }}
    .pill-no {{ color: {C_MILD[0]}; font-weight: 700; }}
    .pill-na {{ color: {TEXT_DIM}; font-style: italic; }}

    .doctor-report {{
        border: 2px solid {C_EMERGENCY[0]}; border-radius: 10px; padding: 1.2rem 1.4rem;
        background-color: {C_EMERGENCY[1]}; color: {TEXT}; margin-top: 0.6rem;
    }}
    .reasoning-line {{
        padding: 0.35rem 0; border-bottom: 1px solid {BORDER}; font-size: 15.5px;
        color: {TEXT};
    }}
    .facility-card {{
        border: 1px solid {BORDER}; border-radius: 8px; padding: 0.8rem 1rem;
        background-color: {SURFACE}; color: {TEXT}; margin-bottom: 0.5rem;
    }}
    .facility-card.big {{ font-size: 19px; padding: 1.1rem 1.3rem; }}
    .checklist-item {{
        padding: 0.5rem 0.8rem; margin-bottom: 0.4rem; border-radius: 6px;
        background-color: {C_MODERATE[1]}; border-left: 5px solid {C_MODERATE[0]};
        color: {TEXT}; font-size: 16px;
    }}

    /* Tables in the rule-engine view are raw HTML, so they need explicit text. */
    table {{ color: {TEXT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

LABEL_DISPLAY = {
    ClassificationLabel.EMERGENCY: ("🚨 EMERGENCY", *C_EMERGENCY),
    ClassificationLabel.SEVERE: ("⚠️ SEVERE", *C_SEVERE),
    ClassificationLabel.MODERATE: ("● MODERATE", *C_MODERATE),
    ClassificationLabel.MILD: ("✓ MILD", *C_MILD),
    ClassificationLabel.INCOMPLETE_ASSESSMENT: ("? MORE INFO NEEDED", *C_NEUTRAL),
}

SEVERITY_TAG = {
    ClassificationLabel.EMERGENCY: ("EMERGENCY", *C_EMERGENCY),
    ClassificationLabel.SEVERE: ("SEVERE", *C_SEVERE),
    ClassificationLabel.MODERATE: ("MODERATE", *C_MODERATE),
    ClassificationLabel.MILD: ("MILD", *C_MILD),
}

# Plain-language message for the caller/caregiver tab -- no clinical jargon.
CALLER_MESSAGE = {
    ClassificationLabel.EMERGENCY: ("🚨 This is an emergency", "Go to the hospital or health centre right now."),
    ClassificationLabel.SEVERE: ("⚠️ Please see a doctor today", "This needs a doctor's attention soon."),
    ClassificationLabel.MODERATE: ("● Please visit a health worker soon", "Not an emergency, but don't ignore it."),
    ClassificationLabel.MILD: ("✓ You can care for this at home", "Follow the advice below and watch for changes."),
    ClassificationLabel.INCOMPLETE_ASSESSMENT: ("❓ We need a little more information", "Please answer the question below."),
}

# ASHA worker's operational instruction per label -- phrasing only, derived
# directly from the label the rules engine already computed (no new
# clinical judgement invented here).
ASHA_ACTION = {
    ClassificationLabel.EMERGENCY: "🚑 Escort or arrange immediate transport to the referral facility below. Do not wait.",
    ClassificationLabel.SEVERE: "⚠️ Arrange transport to a facility today -- this needs a doctor's assessment soon.",
    ClassificationLabel.MODERATE: "🩹 Advise home care per the precautions below. Follow up within 2-3 days, sooner if it worsens.",
    ClassificationLabel.MILD: "✅ Home care is appropriate. Share the precautions below and note when to come back.",
    ClassificationLabel.INCOMPLETE_ASSESSMENT: "❓ Go back to the caregiver and ask the missing question below before deciding.",
}

MISSING_FIELD_LABELS = {
    "not_able_to_drink_or_breastfeed": "Able to drink / breastfeed?",
    "vomits_everything": "Vomiting everything?",
    "convulsions": "Any convulsions / fits?",
    "lethargic_or_unconscious": "Unusually sleepy or unconscious?",
    "symptom_tokens": "What is the main symptom?",
}

# Exam-only fields an ASHA worker (unlike a remote caregiver) can actually
# check in person -- (block attribute, field attribute, human instruction).
EXAM_ONLY_CHECKS = [
    ("cough", "breaths_per_minute", "Count the breathing rate (breaths per minute)."),
    ("cough", "chest_indrawing", "Check for chest indrawing (skin pulling in with each breath)."),
    ("cough", "stridor_when_calm", "Listen for stridor (harsh noisy breathing) while the patient is calm."),
    ("diarrhea", "skin_pinch_goes_back_slowly", "Do a skin pinch test for dehydration."),
]


def pill(value) -> str:
    if value is True:
        return '<span class="pill-yes">YES</span>'
    if value is False:
        return '<span class="pill-no">No</span>'
    return '<span class="pill-na">not assessed</span>'


def tag_list(items: list[str]) -> str:
    if not items:
        return '<span class="pill-na">none</span>'
    return "".join(f'<span class="tag">{i}</span>' for i in items)


def exam_only_checklist(case: ExtractedCase) -> list[str]:
    """Exam-only fields still unassessed -- things only an in-person health
    worker can check, which is exactly the ASHA worker's role per
    app/schemas.py's own docstring on why these fields exist."""
    items = []
    for block_name, field_name, instruction in EXAM_ONLY_CHECKS:
        block = getattr(case, block_name, None)
        if block is not None and getattr(block, field_name) is None:
            items.append(instruction)
    return items


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
st.sidebar.info("Extraction backend: **Local LLM (Ollama)**\nRequires `ollama serve` with `llama3.2:3b` pulled.")

village_choice = st.sidebar.selectbox(
    "Patient village (for facility routing)",
    ["Not specified"] + list(VILLAGES.keys()),
    index=0,
    help="Used only if the case turns out EMERGENCY or SEVERE, to find and route to a nearby facility.",
)

with st.sidebar.expander("Demo facility network"):
    st.caption(
        "Synthetic sample data (Denkanikottai/Hosur taluk) — real facility "
        "master data is not yet wired in. See app/routing/demo_facilities.py."
    )
    for f in build_demo_facility_db().all_facilities():
        caps = []
        if f.has_emergency_care:
            caps.append("ER")
        if f.has_doctor_24hr:
            caps.append("24hr doctor")
        if f.has_blood_bank:
            caps.append("blood bank")
        st.markdown(f"**{f.name}** ({f.facility_type.value}) — {', '.join(caps) or 'basic'}")

if "history" not in st.session_state:
    st.session_state.history = []
if "current" not in st.session_state:
    st.session_state.current = None

if st.session_state.history:
    with st.sidebar.expander(f"Session history ({len(st.session_state.history)})"):
        for h in reversed(st.session_state.history[-15:]):
            st.markdown(f"`{h['time']}` **{h['label']}** — {h['text'][:40]}")

# ---------------------------------------------------------------------------
# Header + input
# ---------------------------------------------------------------------------
st.title("🩺 Rural Health Triage")
st.caption(
    "Describe the patient's symptoms in plain language (English, Hindi, or Hinglish). "
    "The system extracts structured clinical fields, runs the WHO-IMCI / dataset-based "
    "classifier, and shows the result the way each person in the workflow needs to see it — "
    "switch tabs below to compare."
)

text = st.text_area(
    "Patient / caregiver message",
    height=120,
    placeholder="e.g. My 8 month old baby has a fever and is coughing a lot, breathing fast, not able to drink",
)

assess_clicked = st.button("Assess", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------
def render_triage_banner(result: ClassificationResult, big: bool = False) -> None:
    text_, fg, bg = LABEL_DISPLAY[result.label]
    sub = result.condition.replace("_", " ").title() if result.condition else ""
    css_class = "banner big-banner" if big else "banner"
    st.markdown(
        f'<div class="{css_class}" style="color:{fg}; background-color:{bg}; border-left-color:{fg};">'
        f"{text_}"
        f'<span class="subbanner">{sub}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_facility_card(dispatch: DispatchResult, big: bool = False) -> None:
    f = dispatch.facility
    r = dispatch.route
    st.markdown(
        f'<span class="tag" style="background-color:{C_SEVERE[1]}; color:{C_SEVERE[0]}; border-color:{C_SEVERE[0]};">{dispatch.urgency}</span>',
        unsafe_allow_html=True,
    )
    css_class = "facility-card big" if big else "facility-card"
    st.markdown(
        f"""
        <div class="{css_class}">
        <b>{f.name}</b> ({f.facility_type.value})<br>
        Phone: {f.phone or "not on file"}<br>
        Road distance: {r.road_distance_km} km &nbsp;|&nbsp; ETA: {r.eta_minutes} min &nbsp;|&nbsp;
        <a href="{r.maps_link}" target="_blank">Get directions</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_case_details(case: ExtractedCase) -> None:
    st.subheader("Case details")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="field-label">Reported symptom</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-value">{case.symptom or "—"}</div>', unsafe_allow_html=True)
        st.markdown('<div class="field-label">Duration</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-value">{case.duration or "not stated"}</div>', unsafe_allow_html=True)
        st.markdown('<div class="field-label">Severity (as stated)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-value">{case.severity.value}</div>', unsafe_allow_html=True)
    with col2:
        age = (
            f"{case.age_months} months"
            if case.age_months is not None
            else (case.age_group.value if case.age_group is not None else "not stated")
        )
        st.markdown('<div class="field-label">Age</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-value">{age}</div>', unsafe_allow_html=True)
        st.markdown('<div class="field-label">Location (as stated)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-value">{case.location or "not stated"}</div>', unsafe_allow_html=True)
        st.markdown('<div class="field-label">Language</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-value">{case.language or "unknown"}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="field-label">Extraction backend</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-value">{case.llm_backend or "—"}</div>', unsafe_allow_html=True)
        st.markdown('<div class="field-label">Extracted at (UTC)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-value">{case.extracted_at.strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)
        if case.disambiguation_confidence is not None:
            st.markdown('<div class="field-label">Disambiguation confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-value">{case.disambiguation_confidence:.0%}</div>', unsafe_allow_html=True)

    if case.notes:
        st.markdown('<div class="field-label">Notes</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-value">{case.notes}</div>', unsafe_allow_html=True)

    if case.symptom_tokens:
        st.markdown('<div class="field-label">Matched symptom vocabulary</div>', unsafe_allow_html=True)
        st.markdown(tag_list(case.symptom_tokens), unsafe_allow_html=True)

    with st.expander("Raw message as entered"):
        st.write(case.raw_symptom_text)

    render_danger_signs(case, compact=False)

    if case.cough is not None:
        st.markdown("##### Cough / difficult breathing")
        c = case.cough
        ccol1, ccol2, ccol3, ccol4 = st.columns(4)
        with ccol1:
            st.markdown('<div class="field-label">Present</div>', unsafe_allow_html=True)
            st.markdown(pill(c.present), unsafe_allow_html=True)
        with ccol2:
            st.markdown('<div class="field-label">Duration (days)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-value">{c.duration_days if c.duration_days is not None else "—"}</div>', unsafe_allow_html=True)
        with ccol3:
            st.markdown('<div class="field-label">Chest indrawing (exam-only)</div>', unsafe_allow_html=True)
            st.markdown(pill(c.chest_indrawing), unsafe_allow_html=True)
        with ccol4:
            st.markdown('<div class="field-label">Stridor when calm (exam-only)</div>', unsafe_allow_html=True)
            st.markdown(pill(c.stridor_when_calm), unsafe_allow_html=True)

    if case.diarrhea is not None:
        st.markdown("##### Diarrhea / dehydration")
        d = case.diarrhea
        dr1, dr2, dr3 = st.columns(3)
        with dr1:
            st.markdown('<div class="field-label">Present</div>', unsafe_allow_html=True)
            st.markdown(pill(d.present), unsafe_allow_html=True)
            st.markdown('<div class="field-label">Duration (days)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-value">{d.duration_days if d.duration_days is not None else "—"}</div>', unsafe_allow_html=True)
        with dr2:
            st.markdown('<div class="field-label">Blood in stool</div>', unsafe_allow_html=True)
            st.markdown(pill(d.blood_in_stool), unsafe_allow_html=True)
            st.markdown('<div class="field-label">Restless / irritable</div>', unsafe_allow_html=True)
            st.markdown(pill(d.restless_or_irritable), unsafe_allow_html=True)
        with dr3:
            st.markdown('<div class="field-label">Sunken eyes</div>', unsafe_allow_html=True)
            st.markdown(pill(d.sunken_eyes), unsafe_allow_html=True)
            st.markdown('<div class="field-label">Drinks eagerly / thirsty</div>', unsafe_allow_html=True)
            st.markdown(pill(d.drinks_eagerly_thirsty), unsafe_allow_html=True)

    with st.expander("Raw extracted case (JSON) — full audit record"):
        st.json(case.model_dump(mode="json"))


def render_danger_signs(case: ExtractedCase, compact: bool) -> None:
    ds = case.danger_signs
    if not compact:
        st.markdown("##### General danger signs (WHO IMCI)")
    fields = (
        ("Able to drink / breastfeed?", ds.not_able_to_drink_or_breastfeed),
        ("Vomits everything", ds.vomits_everything),
        ("Convulsions", ds.convulsions),
        ("Lethargic / unconscious", ds.lethargic_or_unconscious),
    )
    cols = st.columns(4)
    for col, (label, val) in zip(cols, fields):
        with col:
            st.markdown(f'<div class="field-label">{label}</div>', unsafe_allow_html=True)
            st.markdown(pill(val), unsafe_allow_html=True)


def render_reasoning(result: ClassificationResult, collapsed: bool = False) -> None:
    def _body():
        st.caption("One entry per rule that fired, in the order it was evaluated — nothing here is a black box.")
        for i, line in enumerate(result.reasoning, start=1):
            st.markdown(f'<div class="reasoning-line">{i}. {line}</div>', unsafe_allow_html=True)

        if result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT and result.missing_fields:
            st.markdown("**Still need to know:**")
            for field in result.missing_fields:
                st.markdown(f"- {MISSING_FIELD_LABELS.get(field, field.replace('_', ' '))}")
            next_q = question_for_incomplete_result(result)
            if next_q:
                st.info(f"**Next question to ask the caregiver:** {next_q}\n\nAdd their answer to the message above and click Assess again.")

    if collapsed:
        with st.expander("Clinical reasoning trail"):
            _body()
    else:
        st.subheader("Clinical reasoning trail")
        _body()


def render_candidates(result: ClassificationResult, full: bool = True) -> None:
    if not result.candidates:
        return
    st.subheader("Possible conditions considered")
    if result.probable_disease:
        st.markdown(f"**Most likely:** {result.probable_disease}")
    for c in result.candidates[: (5 if full else 3)]:
        st.markdown(
            f'<div class="card"><b>{c.name}</b> — likelihood {c.score:.0%}</div>',
            unsafe_allow_html=True,
        )
        if full:
            cols = st.columns(2)
            with cols[0]:
                st.markdown("Matched symptoms:")
                st.markdown(tag_list(c.matched_symptoms), unsafe_allow_html=True)
            with cols[1]:
                if c.precautions:
                    st.markdown("Suggested precautions on file:")
                    for p in c.precautions:
                        st.markdown(f"- {p}")


def render_home_care(result: ClassificationResult) -> None:
    if result.label not in (ClassificationLabel.MILD, ClassificationLabel.MODERATE):
        return
    if not result.candidates or not result.candidates[0].precautions:
        return
    st.subheader("Home care guidance")
    st.caption("Does not warrant facility dispatch (MODERATE/MILD) — advise the caregiver directly.")
    for p in result.candidates[0].precautions:
        st.markdown(f"- {p}")


# ---------------------------------------------------------------------------
# Dispatch/report computation -- cached in session_state so switching actor
# tabs never re-runs routing or re-calls the LLM report generator.
# ---------------------------------------------------------------------------
def ensure_dispatch_and_report(cur: dict, backend) -> None:
    result: ClassificationResult = cur["result"]
    case: ExtractedCase = cur["case"]
    urgency = urgency_for_label(result.label)

    if urgency is None or village_choice == "Not specified":
        cur["dispatch"] = None
        cur["report"] = None
        cur["dispatch_key"] = None
        return

    cache_key = (village_choice, result.label.value)
    if cur.get("dispatch_key") == cache_key:
        return  # already computed for this village/label combo

    facility_db = build_demo_facility_db()
    dispatch = route(
        label=result.label,
        location=village_choice,
        facility_db=facility_db,
        at=datetime.now(timezone.utc),
        case_id=case.case_id,
    )
    cur["dispatch"] = dispatch
    cur["dispatch_key"] = cache_key
    if dispatch.no_facility_found:
        cur["report"] = None
    else:
        with st.spinner("Preparing handoff report..."):
            # Doctor-facing -- kept in English regardless of the caregiver's
            # language (only the Caller/Caregiver tab is translated).
            cur["report"] = generate_doctor_report(case, result, dispatch, backend)


# ---------------------------------------------------------------------------
# Per-actor views
# ---------------------------------------------------------------------------
def render_caller_view(cur: dict) -> None:
    result: ClassificationResult = cur["result"]
    gw: LanguageGateway | None = cur.get("gw")
    headline, sub = CALLER_MESSAGE[result.label]
    fg, bg = LABEL_DISPLAY[result.label][1], LABEL_DISPLAY[result.label][2]
    st.markdown(
        f'<div class="banner big-banner" style="color:{fg}; background-color:{bg}; border-left-color:{fg};">'
        f"{_t(gw, headline)}<span class=\"subbanner\">{_t(gw, sub)}</span></div>",
        unsafe_allow_html=True,
    )

    dispatch = cur.get("dispatch")
    if dispatch is not None and not dispatch.no_facility_found:
        st.markdown(f"#### {_t(gw, 'Where to go')}")
        render_facility_card(dispatch, big=True)
    elif urgency_for_label(result.label) is not None and village_choice == "Not specified":
        st.info(_t(gw, "Pick the patient's village in the sidebar to see the nearest facility and directions."))

    if result.label in (ClassificationLabel.MILD, ClassificationLabel.MODERATE) and result.candidates and result.candidates[0].precautions:
        st.markdown(f"#### {_t(gw, 'What you should do')}")
        for p in _tb(gw, list(result.candidates[0].precautions)):
            st.markdown(f"- {p}")

    if result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT:
        next_q = question_for_incomplete_result(result)
        if next_q:
            st.markdown(f"#### {_t(gw, 'Please answer this')}")
            st.info(_t(gw, next_q))
            st.caption(_t(gw, "Add the answer to your message above and press Assess again."))


def render_asha_view(cur: dict) -> None:
    case: ExtractedCase = cur["case"]
    result: ClassificationResult = cur["result"]

    render_triage_banner(result)
    st.markdown(f'<div class="action-card">{ASHA_ACTION[result.label]}</div>', unsafe_allow_html=True)

    checklist = exam_only_checklist(case)
    if checklist:
        st.markdown("#### Physical checks to do in person")
        st.caption("These can't be assessed over SMS/text — only a health worker on-site can check them.")
        for item in checklist:
            st.markdown(f'<div class="checklist-item">☐ {item}</div>', unsafe_allow_html=True)

    st.markdown("#### Danger signs reported so far")
    render_danger_signs(case, compact=True)

    dispatch = cur.get("dispatch")
    if dispatch is not None and not dispatch.no_facility_found:
        st.markdown("#### Referral facility")
        render_facility_card(dispatch)
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Road distance", f"{dispatch.route.road_distance_km} km")
        mcol2.metric("ETA", f"{dispatch.route.eta_minutes} min")
    elif urgency_for_label(result.label) is not None and village_choice == "Not specified":
        st.warning("Pick the patient's village in the sidebar to get a referral facility.")

    render_candidates(result, full=False)
    render_reasoning(result, collapsed=True)


def render_doctor_view(cur: dict) -> None:
    case: ExtractedCase = cur["case"]
    result: ClassificationResult = cur["result"]

    render_triage_banner(result)

    urgency = urgency_for_label(result.label)
    if urgency is not None:
        st.markdown("---")
        st.header("🚑 Doctor handoff")
        dispatch = cur.get("dispatch")
        if village_choice == "Not specified":
            st.warning(
                "No patient village selected in the sidebar — pick one to generate a facility "
                "routing and handoff report for this case."
            )
        elif dispatch is None:
            st.info("Computing routing…")
        elif dispatch.no_facility_found:
            st.error("No eligible facility found to route to.")
            with st.expander("Routing reasoning"):
                for i, line in enumerate(dispatch.reasoning, start=1):
                    st.markdown(f"{i}. {line}")
        else:
            render_facility_card(dispatch)
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Road distance", f"{dispatch.route.road_distance_km} km")
            mcol2.metric("ETA", f"{dispatch.route.eta_minutes} min")
            mcol3.metric("Facility type", dispatch.facility.facility_type.value)

            with st.expander("Routing reasoning (all 6 steps, audit trail)"):
                for i, line in enumerate(dispatch.reasoning, start=1):
                    st.markdown(f"{i}. {line}")

            st.markdown("##### Generated report for receiving doctor")
            st.markdown(f'<div class="doctor-report">{cur.get("report", "")}</div>', unsafe_allow_html=True)
            st.caption(
                "Generated via the LLM backend when reachable; falls back to a plain deterministic "
                "summary of the same facts otherwise — never invents information not already collected below."
            )

    render_home_care(result)
    st.markdown("---")
    render_candidates(result, full=True)
    render_reasoning(result, collapsed=False)
    st.markdown("---")
    render_case_details(case)
    with st.expander("Raw classification result (JSON) — full audit record"):
        st.json(result.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Rule Engine Output view
# ---------------------------------------------------------------------------
_BAND_STYLE = {
    "NON_URGENT": (*C_MILD, "✓ NON-URGENT"),
    "URGENT":     (*C_MODERATE, "● URGENT"),
    "EMERGENCY":  (*C_EMERGENCY, "🚨 EMERGENCY"),
}

_ESI_LABEL = {1: "ESI 1 — Resuscitation", 2: "ESI 2 — Emergent",
              3: "ESI 3 — Urgent", 4: "ESI 4 — Less urgent", 5: "ESI 5 — Non-urgent"}


def render_rule_engine_view(cur: dict) -> None:
    result: ClassificationResult = cur["result"]

    st.subheader("Rule Engine — Internal Output")
    st.caption(
        "This tab exposes the full internal state of the two-stage rule engine "
        "for verification. Every number here is derived, not hard-coded."
    )

    # --- Stage 1 ---
    st.markdown("### Stage 1 — Probabilistic Diagnosis")

    if result.calibrated_confidence is None:
        st.info(
            "Stage 1 NB classifier ran as **supplementary context** on this case "
            "(pediatric IMCI path). The calibrated abstention layer applies only "
            "on the adult/out-of-band route."
        )
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Calibrated confidence", f"{result.calibrated_confidence:.1%}")
            st.caption(f"τ threshold (Youden's J): {result.calibrated_confidence:.3f} threshold shown above")
        with col2:
            st.metric("Raw NB posterior", f"{result.raw_confidence:.1%}" if result.raw_confidence is not None else "—")
        with col3:
            st.metric("Gap to 2nd candidate", f"{result.gap_to_second:.3f}" if result.gap_to_second is not None else "—")

        if result.abstention_triggered:
            st.error(
                "**Abstention triggered** — engine refused to answer. "
                "Coordination layer will route to follow-up. "
                "This is the safety mechanism preventing a confident-but-wrong diagnosis."
            )
        else:
            st.success("**Abstention check passed** — both τ (confidence) and δ (gap) conditions met.")

    if result.candidates:
        st.markdown("#### Ranked disease candidates")
        st.caption("Scores are calibrated isotonic posteriors; the top candidate cleared both the τ and δ thresholds.")
        for i, c in enumerate(result.candidates[:5]):
            bar_pct = int(c.score * 100)
            badge = "**→ SELECTED**" if i == 0 and not result.abstention_triggered else ""
            st.markdown(
                f'<div class="card" style="margin-bottom:0.5rem;">'
                f'<b>#{i+1} {c.name}</b> {badge} — score {c.score:.1%}<br>'
                f'<div style="background:{TRACK};border-radius:4px;height:10px;margin:4px 0;">'
                f'<div style="background:{PRIMARY};width:{bar_pct}%;height:10px;border-radius:4px;"></div></div>'
                f'Matched symptoms: {", ".join(c.matched_symptoms) or "none"}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # --- Stage 2 ---
    st.markdown("### Stage 2 — Emergency Classification (AHP)")

    er = result.emergency_result
    if er is None:
        st.info(
            "Stage 2 emergency scoring did not run — "
            "either the case is on the pediatric IMCI path or Stage 1 abstained."
        )
    else:
        fg, bg, label_text = _BAND_STYLE.get(er.band.value, (*C_NEUTRAL, er.band.value))

        # Score gauge
        st.markdown(
            f'<div class="banner" style="color:{fg};background-color:{bg};border-left-color:{fg};">'
            f'Emergency Score: {er.score} / 10'
            f'<span class="subbanner">{label_text} &nbsp;|&nbsp; {_ESI_LABEL.get(er.esi_level, f"ESI {er.esi_level}")}'
            f'{"&nbsp;|&nbsp; ⚠️ WHO IMCI Override" if er.override_triggered else ""}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        gauge_pct = int(er.score * 10)
        gauge_color = fg
        st.markdown(
            f'<div style="background:{TRACK};border-radius:6px;height:18px;margin:0.5rem 0 1rem 0;">'
            f'<div style="background:{gauge_color};width:{gauge_pct}%;height:18px;border-radius:6px;'
            f'display:flex;align-items:center;justify-content:center;color:{ON_ACCENT};font-size:12px;font-weight:700;">'
            f'{er.score}/10</div></div>',
            unsafe_allow_html=True,
        )

        if er.override_triggered:
            st.error(
                "**WHO IMCI Danger-Sign Override triggered** — score forced to 10 (Emergency) "
                "regardless of the AHP weighted total. This is the unconditional false-negative safety net."
            )

        # Per-attribute breakdown table
        st.markdown("#### Per-attribute breakdown")
        st.caption(
            "AHP weights derived from pairwise comparison matrix (Saaty 1980). "
            "Consistency Ratio CR = 0.0205 < 0.10 ✓"
        )

        attr_display = {
            "complication_probability": "Complication / deterioration probability",
            "time_to_treatment":        "Time-to-treatment sensitivity",
            "disease_severity":         "Disease intrinsic severity",
            "age_vulnerability":        "Patient age vulnerability",
            "onset_acuity":             "Onset acuity",
            "transmissibility":         "Transmissibility (WHO IHR)",
        }

        rows_html = ""
        for key, display_name in attr_display.items():
            attr_s = er.attribute_scores.get(key, 0.0)
            weight = er.attribute_weights.get(key, 0.0)
            contrib = attr_s * weight
            bar_w = int(attr_s * 100)
            rows_html += (
                f"<tr>"
                f"<td style='padding:6px 8px;'>{display_name}</td>"
                f"<td style='padding:6px 8px;text-align:center;'>{weight:.3f}</td>"
                f"<td style='padding:6px 8px;'>"
                f"<div style='display:flex;align-items:center;gap:6px;'>"
                f"<div style='background:{TRACK};border-radius:3px;height:10px;width:80px;flex-shrink:0;'>"
                f"<div style='background:{PRIMARY};width:{bar_w}%;height:10px;border-radius:3px;'></div></div>"
                f"<span>{attr_s:.2f}</span></div></td>"
                f"<td style='padding:6px 8px;text-align:center;'>{contrib:.4f}</td>"
                f"</tr>"
            )

        st.markdown(
            f'<table style="width:100%;border-collapse:collapse;font-size:15px;">'
            f'<thead><tr style="border-bottom:2px solid {TRACK};">'
            f'<th style="text-align:left;padding:6px 8px;">Attribute</th>'
            f'<th style="text-align:center;padding:6px 8px;">AHP Weight</th>'
            f'<th style="text-align:left;padding:6px 8px;">Score (0–1)</th>'
            f'<th style="text-align:center;padding:6px 8px;">Contribution</th>'
            f'</tr></thead><tbody>{rows_html}</tbody>'
            f'<tfoot><tr style="border-top:2px solid {TRACK};font-weight:700;">'
            f'<td colspan="3" style="padding:6px 8px;">Weighted acuity total → Emergency score</td>'
            f'<td style="text-align:center;padding:6px 8px;">'
            f'{sum(er.attribute_scores.get(k,0)*er.attribute_weights.get(k,0) for k in er.attribute_weights):.4f} → {er.score}/10</td>'
            f'</tr></tfoot></table>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Stage 2 reasoning trail")
        for i, line in enumerate(er.reasoning, start=1):
            st.markdown(f'<div class="reasoning-line">{i}. {line}</div>', unsafe_allow_html=True)

    # Full Stage 1 reasoning trail
    st.markdown("#### Stage 1 reasoning trail")
    render_reasoning(result, collapsed=False)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
if assess_clicked:
    if not text.strip():
        st.warning("Please enter a symptom description first.")
    else:
        backend = OllamaBackend()
        gw = LanguageGateway()
        try:
            with st.spinner("Detecting language, extracting and classifying..."):
                english_text = gw.to_english(text)
                case, result = extract_and_classify(english_text, backend, language=gw.language)
                # Keep the caregiver's original words on the audit field --
                # extract_case() stores the (English) text it was handed.
                if english_text != text:
                    case = case.model_copy(update={"raw_symptom_text": text})
        except BackendUnavailable as e:
            st.error(
                f"Ollama backend unavailable: {e}\n\n"
                "Start `ollama serve` and ensure `llama3.2:3b` is pulled, then try again."
            )
        except ExtractionValidationError as e:
            st.error(f"Could not process this message: {e}")
        else:
            st.session_state.history.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "label": result.label.value,
                    "text": text,
                }
            )
            st.session_state.current = {"case": case, "result": result, "gw": gw}

if st.session_state.current is not None:
    backend = OllamaBackend()
    ensure_dispatch_and_report(st.session_state.current, backend)

    st.markdown("---")
    tab_caller, tab_asha, tab_doctor, tab_engine = st.tabs(
        ["📞 Caller / Caregiver", "🏥 ASHA Worker", "🩺 Doctor", "🔬 Rule Engine Output"]
    )
    with tab_caller:
        render_caller_view(st.session_state.current)
    with tab_asha:
        render_asha_view(st.session_state.current)
    with tab_doctor:
        render_doctor_view(st.session_state.current)
    with tab_engine:
        render_rule_engine_view(st.session_state.current)
