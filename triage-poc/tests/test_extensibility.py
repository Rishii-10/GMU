"""Extensibility evidence as a build-checkable test (Pillar 4).

The reusability claim is only as good as the coupling, so we assert it:
  * Agent 2 (surveillance) imports ONLY the shared contract + a few declared
    reuse points, and NEVER the triage decision core (app.rules_engine). It
    reads the recorded case stream, it does not reach into how triage decides.
  * The framework's model backend is a genuine swap point: several interchangeable
    LLMBackend implementations exist behind one interface.
These are the 'low touch-count to extend' evidence the paper points to.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_APP = Path(__file__).resolve().parent.parent / "app"

# The only triage/app modules Agent 2 is allowed to depend on: the shared data
# contract, the case stream it reads, severity labels, the KB, the reusable LLM
# backend, and village coordinates. NOT rules_engine / disambiguation / the
# follow-up modules -- i.e. none of the triage decision pipeline.
_ALLOWED = {
    "app.schemas", "app.case_store", "app.disease_classifier", "app.disease_kb",
    "app.agent1_extraction", "app.routing.demo_facilities",
}
_FORBIDDEN_CORE = "app.rules_engine"


def _app_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods.update(n.name for n in node.names)
    return {m for m in mods if m.startswith("app.")}


def _surveillance_imports() -> set[str]:
    out: set[str] = set()
    for f in sorted((_APP / "surveillance").glob("*.py")):
        out |= _app_imports(f)
    return {m for m in out if not m.startswith("app.surveillance")}


def test_agent2_never_imports_the_triage_decision_core():
    assert _FORBIDDEN_CORE not in _surveillance_imports()


def test_agent2_only_depends_on_declared_seams():
    extra = _surveillance_imports() - _ALLOWED
    assert not extra, f"Agent 2 reached beyond the declared seams: {extra}"


def test_model_backend_is_a_real_swap_point():
    from app.agent1_extraction import (
        GroqBackend, LLMBackend, OllamaBackend, RegexBackend,
    )
    for impl in (OllamaBackend, GroqBackend, RegexBackend):
        assert issubclass(impl, LLMBackend)  # interchangeable behind one interface


def test_surveillance_consumes_the_case_stream_via_a_duck_typed_interface():
    # The extension point, at the loosest possible coupling: Agent 2 reads the
    # stream through a minimal all_cases() interface, not a hard dependency on
    # the concrete CaseStore -- any object that yields the recorded rows works.
    from app.surveillance.ingest import from_case_store

    class FakeStore:
        def all_cases(self, since=None):
            return [{
                "area": "V", "probable_disease": "Gastroenteritis", "label": "SEVERE",
                "symptom_tokens": ["x"], "age_months": 30,
                "recorded_at": "2026-10-17T00:00:00+00:00", "language": "en",
            }]

    cases = from_case_store(FakeStore())
    assert len(cases) == 1 and cases[0].area == "V"
    assert "app.case_store" not in _surveillance_imports()  # not even a hard import
