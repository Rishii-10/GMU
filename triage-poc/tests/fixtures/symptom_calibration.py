"""
Labelled symptom-text -> expected dataset-token sample (Phase 8), for
calibrating app.disambiguation.FAISSDisambiguator.match_dataset_symptom()'s
confidence threshold (currently DEFAULT_DATASET_CONFIDENCE_THRESHOLD=0.65,
explicitly flagged UNCALIBRATED in app/disambiguation.py -- this fixture is
the labelled sample that flag says doesn't exist yet).

Deliberately balanced: exact-token phrasing, natural English phrasing,
Hindi (Devanagari), Hinglish (romanized), and a block of NEGATIVE examples
(unrelated text that should match nothing) -- so accuracy isn't
artificially inflated by only testing easy positive cases. `expected` is
None for every negative example; tests/test_calibration.py asserts against
this fixture without silently dropping the negatives.

Each entry: (input_text, expected_token_or_None). expected_token values
are real app.disease_kb.DiseaseKB.vocabulary tokens (verified against the
loaded CSV, not invented).
"""

CALIBRATION_SAMPLE: list[tuple[str, "str | None"]] = [
    # --- Exact / near-exact dataset token phrasing (should be easy) --------
    ("high fever", "high_fever"),
    ("mild fever", "mild_fever"),
    ("cough", "cough"),
    ("breathlessness", "breathlessness"),
    ("chest pain", "chest_pain"),
    ("headache", "headache"),
    ("vomiting", "vomiting"),
    ("diarrhoea", "diarrhoea"),
    ("joint pain", "joint_pain"),
    ("back pain", "back_pain"),
    ("stomach pain", "stomach_pain"),
    ("fatigue", "fatigue"),
    ("dizziness", "dizziness"),
    ("nausea", "nausea"),
    ("chills", "chills"),
    ("sweating", "sweating"),
    ("itching", "itching"),
    ("skin rash", "skin_rash"),
    ("yellowing of eyes", "yellowing_of_eyes"),
    ("loss of appetite", "loss_of_appetite"),
    ("muscle pain", "muscle_pain"),
    ("abdominal pain", "abdominal_pain"),
    ("constipation", "constipation"),
    ("dark urine", "dark_urine"),
    ("yellowish skin", "yellowish_skin"),
    ("weight loss", "weight_loss"),
    ("weight gain", "weight_gain"),
    ("palpitations", "palpitations"),
    ("restlessness", "restlessness"),
    ("anxiety", "anxiety"),
    ("runny nose", "runny_nose"),
    ("throat irritation", "throat_irritation"),
    # --- Natural English phrasing (not literal token wording) --------------
    ("I have a really high temperature", "high_fever"),
    ("having a lot of trouble breathing", "breathlessness"),
    ("my chest hurts", "chest_pain"),
    ("I keep throwing up", "vomiting"),
    ("loose watery stools", "diarrhoea"),
    ("my joints ache", "joint_pain"),
    ("I feel extremely tired all the time", "fatigue"),
    ("everything looks blurry", "blurred_and_distorted_vision"),
    ("I've lost a lot of weight recently", "weight_loss"),
    # --- Hindi (Devanagari) -------------------------------------------------
    ("बुखार", "high_fever"),
    ("खांसी", "cough"),
    ("साँस फूलना", "breathlessness"),
    ("उल्टी", "vomiting"),
    ("दस्त", "diarrhoea"),
    ("सर दर्द", "headache"),
    ("छाती में दर्द", "chest_pain"),
    ("पेट दर्द", "stomach_pain"),
    ("बदन दर्द", "muscle_pain"),
    ("कमर दर्द", "back_pain"),
    ("कमज़ोरी", "weakness_in_limbs"),
    ("थकान", "fatigue"),
    ("चक्कर आना", "dizziness"),
    ("ठंड लगना", "chills"),
    ("खुजली", "itching"),
    ("आँखें पीली", "yellowing_of_eyes"),
    # --- Hinglish (romanized Hindi) ------------------------------------------
    ("bukhar", "high_fever"),
    ("tez bukhar", "high_fever"),
    ("khaansi", "cough"),
    ("saans phoolna", "breathlessness"),
    ("ulti", "vomiting"),
    ("dast", "diarrhoea"),
    ("sar dard", "headache"),
    ("chhati mein dard", "chest_pain"),
    ("pet dard", "stomach_pain"),
    ("badan dard", "muscle_pain"),
    ("jodo mein dard", "joint_pain"),
    ("kamzori", "weakness_in_limbs"),
    ("jee michlana", "nausea"),
    ("paseena aana", "sweating"),
    ("bhookh na lagna", "loss_of_appetite"),
    # --- Negative examples: must NOT confidently match anything -------------
    ("the weather is nice today", None),
    ("I want to buy some vegetables", None),
    ("what time does the bus arrive", None),
    ("my phone battery is low", None),
    ("thank you very much", None),
    ("यह एक अच्छा दिन है", None),  # "this is a good day"
    ("kal milte hain", None),  # "see you tomorrow"
    ("how much does this cost", None),
]
