"""Deterministic detection of symptoms the caller explicitly DENIES.

The shared case record distinguishes three states for a symptom: reported
present (`ExtractedCase.symptom_tokens`), reported absent
(`ExtractedCase.absent_symptom_tokens`) and not mentioned (in neither list).
This module protects the middle state on the FIRST message.

Measured failure this closes (llama3.2:3b, live, 2026-11-13):
  - "I have fever but no cough"            -> LLM symptoms ["fever"]; the denial was lost.
  - "chest pain, not breathless, no sweating" -> ["chest pain"]; both denials lost.
  - "sir dard hai, ulti nahi hai"           -> ["dard", "ulti"]; "no vomiting" became
                                               vomiting PRESENT (meaning inverted).
and the embedding matcher alone maps "no cough" -> cough at 0.879, so a negated
phrase that reaches it is read as a positive finding.

Two layers, independent of the LLM:
  1. `negated_core(clause)` decides whether ONE short clause is a denial and,
     if so, returns the symptom phrase with the negation words stripped.
  2. `split_denials(raw_text, symptoms, denied)` scans the raw message clause by
     clause, moves any extracted symptom that only occurs inside a negated
     clause into the denied list, and recovers denials the LLM dropped.

Deliberately conservative about what counts as a denial: an inability ("not
able to drink", "can't breathe", "khana nahi kha raha") is a POSITIVE symptom,
not a denial, and is never moved. When a phrase appears both negated and
non-negated in the same message, present wins (the conservative reading).
"""
from __future__ import annotations

import re

# Clause boundaries for the raw-message scan. Broader than the symptom-list
# splitter: sentence punctuation and contrastive conjunctions ("fever but no
# cough") separate a denial from the positive symptom next to it.
_CLAUSE_SPLIT = re.compile(
    r"\s*(?:[,;.!?\n]|&|\+|/|और|तथा|लेकिन|மற்றும்|\band\b|\baur\b|\bplus\b|\bbut\b|\blekin\b|\bmagar\b)\s*",
    re.IGNORECASE,
)

# English: a denial is a clause that OPENS with a negation cue (after an
# optional subject/aux such as "he has", "there is", "also").
_EN_LEAD = r"(?:(?:also|and|but|i|he|she|they|patient|child|baby|there|it)\s+)?(?:(?:has|have|had|is|was|are|am|'s|s)\s+)?"
_EN_NEG = re.compile(
    r"^" + _EN_LEAD + r"(?:"
    r"no\s+signs?\s+of|(?:do|does|did)\s*n[o']?t\s+have|(?:do|does|did)\s+not\s+have|"
    r"(?:has|have)\s*n[o']?t\s+(?:had|got)|(?:is|are|was)\s*n[o']?t\s+having|not\s+having|"
    r"(?:is|are|was)\s*n[o']?t|never\s+had|never|without|denies|denied|deny|free\s+of|no|not"
    r")\b\s*(?:any\s+|much\s+|more\s+|a\s+)?(?P<core>.+)$",
    re.IGNORECASE,
)

# "not"/"no" followed by these is an INABILITY or a general complaint, which is
# itself a symptom ("not able to drink" is a WHO danger sign), never a denial.
_EN_INABILITY = re.compile(
    r"\b(?:not|unable|can'?t|cannot|could\s*n'?t|is\s*n'?t|does\s*n'?t|does\s+not|did\s*n'?t|won'?t|no\s+longer)\s+"
    r"(?:able|eat\w*|drink\w*|feed\w*|breast\w*|sleep\w*|breath(?:e|es|ing)|walk\w*|pass\w*|stop\w*|wak\w*|"
    r"respond\w*|mov\w*|see\w*|hear\w*|talk\w*|speak\w*|swallow\w*|keep\w*|hold\w*|stand\w*|"
    r"urinat\w*|pee\w*|well|good|fine|normal|feeling|gaining|growing|improv\w*|better|recogni\w*|"
    r"open\w*|play\w*|cry\w*|suck\w*|focus\w*|remember\w*)\b",
    re.IGNORECASE,
)

# "no appetite", "bhookh nahi lagti", "neend nahi aati", "peshab nahi ho raha":
# the ABSENCE OF A BODILY FUNCTION is itself a symptom (loss of appetite,
# insomnia, reduced urine), so a negated clause whose core names one is never
# treated as a denial.
_LOSS_OF_FUNCTION = re.compile(
    r"\b(?:appetite|hunger|energy|strength|sleep|urine|urination|stool|stools|bowel\s+movements?|"
    r"movement|response|tears|bhookh|bhook|bhuk|bhukh|neend|nind|peshab|pishab|taqat|takat|"
    r"भूख|नींद|पेशाब|ताकत)\b",
    re.IGNORECASE,
)

# Hindi / Hinglish / Tamil: the negation is clause-FINAL ("ulti nahi hai",
# "bukhar nahi", "koi dard nahi", "வலி இல்லை"). A denial is "<symptom> nahi"
# optionally followed by a copula/"happening" auxiliary; "nahi" followed by any
# other verb ("khana nahi kha raha" = not eating) is an inability, not a denial.
_HI_NEG_WORD = r"(?:nahi|nahin|nahee|nai|nhi|nahi\s+to|नहीं|नही|ना|illai|இல்லை)"
_HI_AUX = r"(?:\s+(?:hai|he|h|hain|tha|thi|the|hua|hui|huaa|huyi|aaya|aayi|aa\s+raha|aa\s+rahi|ho\s+raha|ho\s+rahi|ho\s+rahe|hota|hoti|है|हैं|था|थी|हुआ|हुई|हो\s+रहा|हो\s+रही|आया|आई))*"
_HI_NEG = re.compile(
    r"^(?:koi\s+|कोई\s+|bilkul\s+|बिल्कुल\s+)?(?P<core>.+?)\s+(?:bilkul\s+|बिल्कुल\s+)?" + _HI_NEG_WORD + _HI_AUX + r"\s*$",
    re.IGNORECASE,
)
_HI_CORE_FILLER = re.compile(r"^(?:koi|कोई|bilkul|बिल्कुल|mujhe|mujhko|use|usko|ise|isko|मुझे|उसे)\s+", re.IGNORECASE)

_TRAILING_FILLER = re.compile(
    r"\s+(?:at\s+all|either|now|today|anymore|any\s+more|so\s+far|yet|hai|है)\s*$", re.IGNORECASE
)


def _clean(core: str) -> str:
    core = core.strip(" .,:;!?'\"")
    prev = None
    while prev != core:
        prev = core
        core = _TRAILING_FILLER.sub("", core).strip()
        core = _HI_CORE_FILLER.sub("", core).strip()
    return core


def split_clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(text or "") if c and c.strip()]


def negated_core(clause: str) -> str | None:
    """The symptom phrase a clause DENIES, or None when the clause is not a denial.

    "no cough" -> "cough"; "not breathless" -> "breathless"; "ulti nahi hai" ->
    "ulti"; "koi dard nahi" -> "dard". "not able to drink", "can't breathe",
    "khana nahi kha raha" -> None (inabilities are symptoms, not denials).
    """
    c = (clause or "").strip()
    if not c:
        return None
    if _EN_INABILITY.search(c):
        return None
    m = _EN_NEG.match(c)
    if m is None:
        m = _HI_NEG.match(c)
    if m is None:
        return None
    core = _clean(m.group("core"))
    if not core or _EN_INABILITY.search(core) or _LOSS_OF_FUNCTION.search(core):
        return None
    return core


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _mentions(clause: str, phrase: str) -> bool:
    c, p = _norm(clause), _norm(phrase)
    return bool(p) and (p in c or c in p)


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[\wऀ-ॿ஀-௿]+", _norm(text)) if len(w) >= 3}


def _grounded(phrase: str, negated_cores: list[str]) -> bool:
    """True when every content word of `phrase` (5-char stem) occurs in one
    negated clause -- "breathlessness" is grounded by "not breathless", while
    "cough without phlegm" is not grounded by "no fever"."""
    want = _stems(phrase)
    return bool(want) and any(want <= _stems(core) for core in negated_cores)


def split_denials(
    raw_text: str, symptoms: list[str], denied: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """Reconcile the extracted symptom list against negations in the raw message.

    Returns (present_symptoms, denied_symptoms):
      - an extracted item that is itself a negated phrase ("no cough") is
        moved to denied as its core ("cough");
      - an extracted item that occurs ONLY inside negated clauses of the raw
        message (the LLM copied "ulti" out of "ulti nahi hai") is moved to denied;
      - a negated clause the LLM dropped entirely is recovered as a denial;
      - an item mentioned both negated and non-negated stays present.
    """
    clauses = split_clauses(raw_text)
    neg_cores = [(cl, negated_core(cl)) for cl in clauses]
    negated_clauses = [(cl, core) for cl, core in neg_cores if core]
    positive_clauses = [cl for cl, core in neg_cores if not core]

    # An LLM-reported denial is kept ONLY when it is grounded in a negated
    # clause of the raw message. Measured: llama3.2:3b copied the prompt's
    # example denials ("cough", "breathless", "sweating") into a message that
    # denied only vomiting. A fabricated denial can rule a disease out, so an
    # ungrounded one is dropped rather than trusted.
    denied_out: list[str] = []
    for d in denied or []:
        d = _clean(negated_core(d) or d)
        if (
            d
            and _grounded(d, [core for _, core in negated_clauses])
            and _norm(d) not in map(_norm, denied_out)
        ):
            denied_out.append(d)

    present_out: list[str] = []
    for item in symptoms or []:
        own = negated_core(item)
        if own:
            if _norm(own) not in map(_norm, denied_out):
                denied_out.append(own)
            continue
        in_neg = any(_mentions(core, item) or _mentions(cl, item) for cl, core in negated_clauses)
        in_pos = any(_mentions(cl, item) for cl in positive_clauses)
        if in_neg and not in_pos:
            if _norm(item) not in map(_norm, denied_out):
                denied_out.append(item)
        else:
            present_out.append(item)

    for _, core in negated_clauses:
        covered = any(_mentions(core, d) for d in denied_out)
        if not covered and not any(_mentions(core, p) for p in present_out):
            denied_out.append(core)

    present_norm = {_norm(p) for p in present_out}
    denied_out = [d for d in denied_out if _norm(d) not in present_norm]
    return present_out, denied_out
