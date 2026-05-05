import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tree_memory_engine import build_demo_memory


def main():
    memory = build_demo_memory()

    print("\nBefore update:")
    for query in [
        "Who produces premium car tires?",
        "What are Michelin stars?",
        "For the reptile python, what does it shed?",
        "For Apple fruit, what color are Granny Smith apples?",
    ]:
        print(query, "->", memory.answer(query)["answer"])

    memory.update_fact(
        "artifacts/vehicles/car_tires",
        "car_tires.maker",
        "Bridgestone produces premium car tires in the updated memory.",
        "Bridgestone",
        "vehicle tires maker update",
    )

    print("\nAfter local update:")
    for query in [
        "Who produces premium car tires now?",
        "What are Michelin stars, not tires?",
        "What power source do Tesla car engines use after the tire update?",
    ]:
        print(query, "->", memory.answer(query)["answer"])

    explanation = memory.explain_retrieval("What are Michelin stars, not tires?")
    print("\nExplanation example:")
    print(json.dumps(explanation, indent=2))

    out = ROOT / "artifacts" / "demo" / "tree_memory_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    memory.save(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
