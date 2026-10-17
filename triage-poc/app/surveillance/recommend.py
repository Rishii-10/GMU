"""Map a detected outbreak disease to a recommended public-health action.

A small, transparent rules table -- the agent proposes WHAT to do, not just
what it sees. Diseases are grouped into public-health response categories; every
one of the 41 dataset diseases resolves to a category (falling back to a generic
field-confirmation action), so recommend_action() always returns something
usable.

This is an operational prompt for a field response, NOT a clinical instruction.
"""
from __future__ import annotations

# disease (exact dataset name) -> response category
_CATEGORY: dict[str, str] = {
    # water / sanitation (faeco-oral)
    "Gastroenteritis": "waterborne",
    "Typhoid": "waterborne",
    "hepatitis A": "waterborne",
    "Hepatitis E": "waterborne",
    "Jaundice": "waterborne",
    # vector-borne
    "Dengue": "vector",
    "Malaria": "vector",
    # respiratory / airborne
    "Pneumonia": "respiratory",
    "Tuberculosis": "respiratory",
    "Bronchial Asthma": "respiratory",
    "Common Cold": "respiratory",
    # bloodborne / sexual
    "AIDS": "bloodborne",
    "Hepatitis B": "bloodborne",
    "Hepatitis C": "bloodborne",
    "Hepatitis D": "bloodborne",
    # contact / skin
    "Chicken pox": "contact",
    "Impetigo": "contact",
    "Fungal infection": "contact",
    "Psoriasis": "contact",
    "Acne": "contact",
    # medication / adverse event
    "Drug Reaction": "adverse_event",
    # non-communicable (surveillance = service-load signal, not an outbreak)
    "Diabetes": "ncd",
    "Hypertension": "ncd",
    "Hyperthyroidism": "ncd",
    "Hypothyroidism": "ncd",
    "Hypoglycemia": "ncd",
    "Arthritis": "ncd",
    "Osteoarthristis": "ncd",
    "Cervical spondylosis": "ncd",
    "Migraine": "ncd",
    "GERD": "ncd",
    "Peptic ulcer diseae": "ncd",
    "Varicose veins": "ncd",
    "Dimorphic hemmorhoids(piles)": "ncd",
    "(vertigo) Paroymsal  Positional Vertigo": "ncd",
    "Allergy": "ncd",
    "Urinary tract infection": "ncd",
    "Alcoholic hepatitis": "ncd",
    "Chronic cholestasis": "ncd",
    "Heart attack": "acute_cardiac",
    "Paralysis (brain hemorrhage)": "acute_neuro",
}

_ACTION: dict[str, str] = {
    "waterborne": "Inspect local water sources and sanitation; distribute ORS and chlorination advice; verify safe drinking water in the affected village.",
    "vector": "Trigger vector-control / source reduction (standing water, fogging) and distribute bed nets in the affected village.",
    "respiratory": "Mount active case-finding for pneumonia/TB; check for crowding and indoor air quality; ensure referral capacity.",
    "bloodborne": "Review needle/blood-safety and awareness; arrange confirmatory testing and counselling; no containment campaign needed.",
    "contact": "Advise hygiene and isolation of active cases; check schools/anganwadis for spread; provide topical treatment supplies.",
    "adverse_event": "Flag a possible medication/adverse-event cluster; review recently dispensed drugs in the village; report to pharmacovigilance.",
    "ncd": "Likely a chronic-care service-load signal, not a transmissible outbreak: plan screening/follow-up capacity rather than containment.",
    "acute_cardiac": "Acute cardiac cluster is unusual: verify cases and ensure emergency referral/transport readiness.",
    "acute_neuro": "Acute neurological cluster is unusual: verify cases and ensure emergency referral/transport readiness.",
    "unknown": "Confirm the signal with a field visit before acting; collect line-list data for the affected village.",
}


def category_for(disease: str) -> str:
    return _CATEGORY.get(disease, "unknown")


def recommend_action(disease: str) -> str:
    """Recommended public-health action for a disease cluster. Always returns a
    usable string (generic field-confirmation for anything unmapped)."""
    return _ACTION[category_for(disease)]
