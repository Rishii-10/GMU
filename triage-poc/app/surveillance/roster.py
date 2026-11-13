"""Who the surveillance agent can contact, per village.

DEMONSTRATION ROSTER: synthetic names and +91-00000 placeholder numbers for the
same five demo villages as app.routing.demo_facilities. No real person or number.
A deployment replaces ROSTER with its ASHA / ANM / block-officer directory; the
agent only needs `contacts_for(area, roles)`.
"""
from __future__ import annotations

ROLE_ASHA = "asha"
ROLE_ANM = "anm"
ROLE_DISTRICT_OFFICER = "district_officer"

ROSTER: dict[str, list[dict]] = {
    "Denkanikottai": [
        {"name": "ASHA Denkanikottai-1 (demo)", "role": ROLE_ASHA, "phone": "+910000000101"},
        {"name": "ANM Denkanikottai (demo)", "role": ROLE_ANM, "phone": "+910000000102"},
    ],
    "Anchetty": [
        {"name": "ASHA Anchetty-1 (demo)", "role": ROLE_ASHA, "phone": "+910000000201"},
        {"name": "ASHA Anchetty-2 (demo)", "role": ROLE_ASHA, "phone": "+910000000202"},
        {"name": "ANM Anchetty (demo)", "role": ROLE_ANM, "phone": "+910000000203"},
    ],
    "Thally": [
        {"name": "ASHA Thally-1 (demo)", "role": ROLE_ASHA, "phone": "+910000000301"},
    ],
    "Kelamangalam": [
        {"name": "ASHA Kelamangalam-1 (demo)", "role": ROLE_ASHA, "phone": "+910000000401"},
    ],
    "Bannerghatta": [
        {"name": "ASHA Bannerghatta-1 (demo)", "role": ROLE_ASHA, "phone": "+910000000501"},
    ],
}

DISTRICT_OFFICER = {"name": "District Surveillance Officer (demo)", "role": ROLE_DISTRICT_OFFICER,
                    "phone": "+910000000901"}


def contacts_for(area: str, roles: tuple[str, ...]) -> list[dict]:
    out = [c for c in ROSTER.get(area, []) if c["role"] in roles]
    if ROLE_DISTRICT_OFFICER in roles:
        out.append(DISTRICT_OFFICER)
    return out
