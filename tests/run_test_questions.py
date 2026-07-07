
import json
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.vectorstore import vectorstore_exists
from src.services.rag_chain import answer_question

QUESTIONS_FILE = Path(__file__).parent.parent / "data" / "sample_outputs" / "test_questions.json"
RESULTS_FILE = Path(__file__).parent.parent / "data" / "sample_outputs" / "test_results.json"
REQUEST_DELAY_SECONDS = 13


def main():
    if not vectorstore_exists():
        print("ERROR: vector store not found. Run `python -m src.ingest` first.")
        sys.exit(1)

    questions = json.loads(QUESTIONS_FILE.read_text())
    results = []
    passed = 0

    for i, q in enumerate(questions):
        print(f"\n[{q['id']}] ({q['category']}) {q['question']}")

        try:
            result = answer_question(q["question"])
            result_dict = result.to_dict()
        except Exception as exc:
            print(f"  -> ERROR: {type(exc).__name__}: {exc}")
            results.append({"question": q, "result": None, "passed": False, "error": str(exc)})
            RESULTS_FILE.write_text(json.dumps(results, indent=2))
            if i < len(questions) - 1:
                time.sleep(REQUEST_DELAY_SECONDS)
            continue

        print(f"  -> answer: {result_dict['answer']}")
        print(f"  -> confidence: {result_dict['confidence']}")
        print(f"  -> sources: {result_dict['sources']}")

        ok = True
        if q["category"] == "not_found":
            ok = result_dict["confidence"] == "not_found"
        else:
            expected_terms = q.get("expected_contains", [])
            answer_lower = result_dict["answer"].lower()
            ok = all(term.lower() in answer_lower for term in expected_terms)

        passed += int(ok)
        results.append({"question": q, "result": result_dict, "passed": ok})
        print("  -> PASS" if ok else "  -> FAIL (check manually, LLM phrasing may vary)")

        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        if i < len(questions) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n{passed}/{len(questions)} questions matched expectations.")
    print(f"Full results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
