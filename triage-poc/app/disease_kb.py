"""
Disease knowledge base loaded from CSVs: the two user-supplied
(`data/disease_symptoms.csv`, `data/disease_precautions.csv`) plus an
optional third, `data/disease_severity.csv` (disease -> EMERGENCY/SEVERE/
MODERATE/MILD), which app.disease_classifier.severity_for_disease() uses
to turn a ranked disease candidate into a triage tier for the adult/
out-of-band route.

Deterministic, no LLM: this module is pure CSV parsing + in-memory lookup
tables, following the same "no AI in the data layer" rule as
app/rules_engine.py. app/disease_classifier.py (Phase 2) is the only thing
that turns this into a ranked decision.

SEVERITY CSV -- decision, stated explicitly
---------------------------------------------
Disease severity is a clinical judgment, not something mechanically
derivable from the two symptom/precaution CSVs. A naive derivation --
e.g. flagging a disease EMERGENCY because its precautions mention "call
ambulance" or "nearest hospital" -- was tried and rejected: it correctly
flags Heart attack (precautions literally say "call ambulance") but
MISSES Paralysis (brain hemorrhage) entirely, whose precautions in this
dataset are just "massage, eat healthy, exercise, consult doctor" despite
a brain hemorrhage being an equally acute emergency. The precaution text
in this dataset is generic self-care advice, not a severity-labelled
clinical source, so keyword-deriving severity from it would silently
create a clinically unsafe result for exactly the case that matters most.

Given that, severity is kept as an explicit, versioned, EDITABLE data file
(this repo ships a curated default covering all 41 diseases) rather than
buried in Python source -- anyone with real clinical severity data can
replace `data/disease_severity.csv` with better values and nothing in the
code needs to change. Missing rows (or a missing file entirely) degrade
gracefully: app.disease_classifier.severity_for_disease() falls back to
DEFAULT_UNKNOWN_DISEASE_SEVERITY (MODERATE) rather than erroring, same
"honest fallback, no crash" posture as the rest of this codebase.

NORMALIZATION -- decision, stated explicitly
---------------------------------------------
The raw CSV has known formatting artifacts: leading/trailing whitespace on
every cell (e.g. " skin_rash"), and a handful of tokens with a stray
embedded space instead of an underscore (e.g. "dischromic _patches",
"foul_smell_of urine", "spotting_ urination" -- confirmed by a full scan of
all 131 unique symptom tokens in the dataset). Every symptom token is
normalized to lowercase snake_case: strip, collapse internal whitespace to
"_", so "dischromic _patches" and "dischromic_patches" both key the same
token. This is a formatting fix, not a semantic judgment call -- no token is
renamed, merged, or reinterpreted beyond whitespace/case normalization.

Disease names are normalized for whitespace only (strip trailing space,
e.g. "Diabetes " -> "Diabetes"), NOT lowercased -- they are shown to a
doctor/caregiver as-is and should stay human-readable exactly as authored
in the CSV.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_WHITESPACE_RUN = re.compile(r"\s+")
_UNDERSCORE_RUN = re.compile(r"_+")


def normalize_symptom_token(raw: str) -> str:
    """Lowercase snake_case, collapsing any run of whitespace to a single
    underscore. See module docstring for why this is needed.

    Collapses repeated underscores too: some tokens already have an
    underscore adjacent to the stray space (e.g. "dischromic _patches" ->
    whitespace-to-underscore alone would give "dischromic__patches"), so
    underscore-run collapsing has to run after the whitespace substitution.
    """
    token = raw.strip().lower()
    token = _WHITESPACE_RUN.sub("_", token)
    token = _UNDERSCORE_RUN.sub("_", token)
    return token


def normalize_disease_name(raw: str) -> str:
    """Whitespace-only cleanup -- disease names stay human-readable and are
    never lowercased (see module docstring)."""
    return raw.strip()


class DiseaseKB:
    """In-memory knowledge base: disease -> symptom tokens, disease ->
    precautions. Built once at construction from the two CSVs; every lookup
    afterward is a plain dict access, no I/O.
    """

    def __init__(
        self,
        symptoms_by_disease: dict[str, set[str]],
        precautions_by_disease: dict[str, list[str]],
        severity_by_disease: Optional[dict[str, str]] = None,
    ):
        self.symptoms_by_disease = symptoms_by_disease
        self.precautions_by_disease = precautions_by_disease
        self.severity_by_disease = severity_by_disease or {}

    @classmethod
    def load(cls, data_dir: Path | str = DEFAULT_DATA_DIR) -> "DiseaseKB":
        data_dir = Path(data_dir)
        symptoms_by_disease = _load_symptoms(data_dir / "disease_symptoms.csv")
        precautions_by_disease = _load_precautions(data_dir / "disease_precautions.csv")
        severity_by_disease = _load_severity(data_dir / "disease_severity.csv")
        return cls(symptoms_by_disease, precautions_by_disease, severity_by_disease)

    @property
    def diseases(self) -> list[str]:
        return sorted(self.symptoms_by_disease.keys())

    @property
    def vocabulary(self) -> list[str]:
        """Every distinct normalized symptom token across all diseases --
        the vocabulary app/disambiguation.py's FAISS index is built from
        (Phase 3)."""
        vocab: set[str] = set()
        for tokens in self.symptoms_by_disease.values():
            vocab |= tokens
        return sorted(vocab)

    def symptoms_for(self, disease: str) -> set[str]:
        return self.symptoms_by_disease.get(disease, set())

    def precautions_for(self, disease: str) -> list[str]:
        return self.precautions_by_disease.get(disease, [])

    def severity_for(self, disease: str) -> Optional[str]:
        """Raw severity string from data/disease_severity.csv (e.g.
        "EMERGENCY"), or None if the disease has no row on file --
        app.disease_classifier.severity_for_disease() is what applies the
        DEFAULT_UNKNOWN_DISEASE_SEVERITY fallback for that case."""
        return self.severity_by_disease.get(disease)


def _load_symptoms(path: Path) -> dict[str, set[str]]:
    symptoms_by_disease: dict[str, set[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header: Disease, Symptom_1..17
        for row in reader:
            if not row or not row[0].strip():
                continue
            disease = normalize_disease_name(row[0])
            tokens = {normalize_symptom_token(c) for c in row[1:] if c.strip()}
            symptoms_by_disease.setdefault(disease, set()).update(tokens)
    return symptoms_by_disease


def _load_precautions(path: Path) -> dict[str, list[str]]:
    precautions_by_disease: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header: Disease, Precaution_1..4
        for row in reader:
            if not row or not row[0].strip():
                continue
            disease = normalize_disease_name(row[0])
            precautions = [c.strip() for c in row[1:] if c.strip()]
            precautions_by_disease[disease] = precautions
    return precautions_by_disease


def _load_severity(path: Path) -> dict[str, str]:
    """Optional file (see module docstring for why this is a curated,
    editable data file rather than something derived from the other two
    CSVs). Missing file, or a row missing its severity value, is not an
    error -- just absent from the returned dict; the caller
    (app.disease_classifier.severity_for_disease) supplies the fallback."""
    severity_by_disease: dict[str, str] = {}
    if not path.exists():
        return severity_by_disease
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header: Disease, Severity
        for row in reader:
            if not row or not row[0].strip():
                continue
            if len(row) < 2 or not row[1].strip():
                continue
            disease = normalize_disease_name(row[0])
            severity_by_disease[disease] = row[1].strip().upper()
    return severity_by_disease
