"""Quick manual demo: run a few sample cases through the real pipeline
(Ollama extraction -> Rules Engine classification) and print the output.

Usage:
    ./.venv/bin/python demo.py
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.agent1_extraction import OllamaBackend, extract_and_classify

SAMPLE_CASES = [
    "My 8 month old baby has a fever and is coughing a lot, breathing fast",
    "My child is 3 years old, has had diarrhea for 2 days, is very sleepy and hard to wake up",
    "I am 45 years old, I have chest pain and body ache since yesterday",
]


def main():
    backend = OllamaBackend()
    for text in SAMPLE_CASES:
        print("=" * 70)
        print(f"INPUT: {text}")
        try:
            case, result = extract_and_classify(text, backend)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        print("\nExtracted case:")
        print(json.dumps(case.model_dump(), indent=2, default=str))
        print("\nClassification result:")
        print(json.dumps(result.model_dump(), indent=2, default=str))
        print()


if __name__ == "__main__":
    main()
