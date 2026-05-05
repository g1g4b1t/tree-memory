import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tree_memory_engine import build_demo_memory


class TreeMemoryEngineTests(unittest.TestCase):
    def test_local_update_preserves_nearby_branch(self):
        memory = build_demo_memory()
        self.assertEqual(memory.answer("Who produces premium car tires?")["answer"], "Michelin")
        self.assertEqual(memory.answer("What are Michelin stars?")["answer"], "restaurant awards")

        memory.update_fact(
            "artifacts/vehicles/car_tires",
            "car_tires.maker",
            "Bridgestone produces premium car tires in the updated memory.",
            "Bridgestone",
            "vehicle tires maker update",
        )

        self.assertEqual(memory.answer("Who produces premium car tires now?")["answer"], "Bridgestone")
        self.assertEqual(memory.answer("What are Michelin stars, not tires?")["answer"], "restaurant awards")

    def test_ambiguous_python_routes_to_reptile(self):
        memory = build_demo_memory()
        result = memory.answer("For the reptile python, what does it shed?")
        self.assertEqual(result["answer"], "shed skin")
        self.assertEqual(result["path"], "living/reptiles/python_snake")


if __name__ == "__main__":
    unittest.main()
