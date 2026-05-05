import json
import re
import time
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd


TOP_K = 4


@dataclass
class Fact:
    id: int
    path: str
    text: str
    slot: str
    answer: str
    active: bool = True
    version: int = 1


@dataclass
class Query:
    text: str
    path: str
    slot: str
    answer: str
    phase: str


def toks(text):
    return set(re.findall(r"[a-z0-9+#]+", text.lower()))


def score(query, fact):
    q, f = toks(query), toks(fact.text + " " + fact.path.replace("/", " "))
    if not q or not f:
        return 0.0
    overlap = len(q & f)
    return overlap + 0.1 * overlap / max(1, len(q))


class FlatMemory:
    def __init__(self):
        self.facts = []

    def add(self, fact):
        # Flat memory appends facts and keeps old conflicting memories active.
        self.facts.append(fact)

    def retrieve(self, query, top_k=TOP_K):
        scored = [(score(query, f), -f.id, f) for f in self.facts if f.active]
        return [f for s, _, f in sorted(scored, reverse=True)[:top_k] if s > 0]


class TreeMemory:
    def __init__(self):
        self.facts = []
        self.path_tokens = {}

    def add(self, fact):
        # Tree memory updates locally: same path + same slot deactivates old versions.
        version = 1
        for old in self.facts:
            if old.path == fact.path and old.slot == fact.slot and old.active:
                old.active = False
                version = max(version, old.version + 1)
        fact.version = version
        self.facts.append(fact)
        for prefix in self.prefixes(fact.path):
            self.path_tokens.setdefault(prefix, set()).update(toks(fact.text + " " + fact.path.replace("/", " ")))

    def prefixes(self, path):
        parts = path.split("/")
        return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]

    def route(self, query):
        # Pick the most semantically matching leaf path from the tree index.
        q = toks(query)
        candidates = []
        for path, words in self.path_tokens.items():
            if not any(f.active and f.path == path for f in self.facts):
                continue
            overlap = len(q & words)
            depth_bonus = 0.05 * path.count("/")
            candidates.append((overlap + depth_bonus, path))
        candidates.sort(reverse=True)
        return candidates[0][1] if candidates else ""

    def retrieve(self, query, top_k=TOP_K):
        # Retrieve from the routed leaf, its ancestors, and nearby children.
        path = self.route(query)
        if not path:
            return []
        allowed = set(self.prefixes(path))
        allowed.add(path)
        for f in self.facts:
            if f.active and (f.path.startswith(path + "/") or path.startswith(f.path + "/")):
                allowed.add(f.path)
        local = [f for f in self.facts if f.active and f.path in allowed]
        scored = [(score(query, f) + (2.0 if f.path == path else 0.5 if f.path in allowed else 0.0), -f.id, f) for f in local]
        return [f for s, _, f in sorted(scored, reverse=True)[:top_k] if s > 0]


def seed_facts():
    rows = [
        ("artifacts/vehicles/car/tires", "Michelin produces premium car tires.", "car_tires.maker", "Michelin"),
        ("artifacts/vehicles/car/tires", "Car tires use rubber compounds for road grip.", "car_tires.material", "rubber compounds"),
        ("artifacts/vehicles/car/engine", "Tesla car engines use electric motors.", "car_engine.power", "electric motors"),
        ("artifacts/vehicles/car/engine", "Combustion engines burn fuel to produce motion.", "car_engine.combustion", "burn fuel"),
        ("artifacts/tools/hammer", "A hammer drives nails into wood.", "hammer.use", "drives nails"),
        ("artifacts/tools/screwdriver", "A screwdriver turns metal screws.", "screwdriver.use", "turns screws"),
        ("artifacts/computing/python_code", "Python lists use square brackets.", "python.lists", "square brackets"),
        ("artifacts/computing/cpp_code", "C++ vectors use templates.", "cpp.vectors", "templates"),
        ("living/mammals/dog", "Golden retrievers are friendly dogs.", "dog.temperament", "friendly"),
        ("living/mammals/dog", "Dogs have paws, fur, and a strong sense of smell.", "dog.body", "paws and fur"),
        ("living/mammals/cat", "Persian cats have soft fur.", "cat.body", "soft fur"),
        ("living/birds/eagle", "Eagles fly with powerful wings.", "eagle.motion", "powerful wings"),
        ("living/plants/oak_tree", "A tire swing can hang from an oak tree branch.", "oak_tree.swing", "tire swing"),
        ("culture/food/restaurants", "Michelin stars are awards for excellent restaurants.", "michelin_star.meaning", "restaurant awards"),
        ("living/reptiles/python_snake", "Python snakes shed their skin.", "python_snake.body", "shed skin"),
        ("living/fish/electric_eel", "Electric eels produce electric shocks.", "electric_eel.power", "electric shocks"),
    ]
    return [Fact(i + 1, *row) for i, row in enumerate(rows)]


def update_facts(start_id):
    rows = [
        ("artifacts/vehicles/car/tires", "Bridgestone produces premium car tires in the updated memory.", "car_tires.maker", "Bridgestone"),
        ("artifacts/computing/python_code", "Python lists still use square brackets after the update.", "python.lists", "square brackets"),
    ]
    return [Fact(start_id + i, *row) for i, row in enumerate(rows)]


INITIAL_QUERIES = [
    Query("Who produces premium car tires?", "artifacts/vehicles/car/tires", "car_tires.maker", "Michelin", "initial"),
    Query("What material gives car tires road grip?", "artifacts/vehicles/car/tires", "car_tires.material", "rubber compounds", "initial"),
    Query("What power source do Tesla car engines use?", "artifacts/vehicles/car/engine", "car_engine.power", "electric motors", "initial"),
    Query("What does a hammer do to nails?", "artifacts/tools/hammer", "hammer.use", "drives nails", "initial"),
    Query("What do Python lists use?", "artifacts/computing/python_code", "python.lists", "square brackets", "initial"),
    Query("What does a Python snake shed?", "living/reptiles/python_snake", "python_snake.body", "shed skin", "initial"),
    Query("What are Michelin stars?", "culture/food/restaurants", "michelin_star.meaning", "restaurant awards", "initial"),
    Query("What animal has paws and fur?", "living/mammals/dog", "dog.body", "paws and fur", "initial"),
    Query("What bird flies with powerful wings?", "living/birds/eagle", "eagle.motion", "powerful wings", "initial"),
]

UPDATE_QUERIES = [
    Query("Who produces premium car tires now?", "artifacts/vehicles/car/tires", "car_tires.maker", "Bridgestone", "after_update"),
    Query("What are Michelin stars after the tire update?", "culture/food/restaurants", "michelin_star.meaning", "restaurant awards", "after_update"),
    Query("What power source do Tesla car engines use after the tire update?", "artifacts/vehicles/car/engine", "car_engine.power", "electric motors", "after_update"),
    Query("What do Python lists use after the update?", "artifacts/computing/python_code", "python.lists", "square brackets", "after_update"),
]


def answer_from(retrieved):
    return retrieved[0].answer if retrieved else None


def eval_memory(name, memory, queries):
    rows = []
    for q in queries:
        ret = memory.retrieve(q.text, TOP_K)
        answers = [f.answer for f in ret]
        paths = [f.path for f in ret]
        slots = [f.slot for f in ret]
        top1 = answer_from(ret)
        rows.append({
            "memory": name,
            "phase": q.phase,
            "query": q.text,
            "expected_path": q.path,
            "expected_answer": q.answer,
            "top1_answer": top1,
            "top1_correct": top1 == q.answer,
            "hit_at_k": q.answer in answers,
            "path_precision": sum(p == q.path for p in paths) / max(1, len(paths)),
            "wrong_path_hits": sum(p != q.path for p in paths),
            "conflict_hits": sum(s == q.slot and a != q.answer for s, a in zip(slots, answers)),
            "retrieved": " | ".join(f"{f.path}: {f.answer}" for f in ret),
        })
    return rows


def summarize(df):
    summary = df.groupby(["memory", "phase"]).agg(
        top1_acc=("top1_correct", "mean"),
        hit_at_k=("hit_at_k", "mean"),
        path_precision=("path_precision", "mean"),
        wrong_path_hits=("wrong_path_hits", "mean"),
        conflict_hits=("conflict_hits", "mean"),
        n=("query", "count"),
    ).reset_index()
    overall = df.groupby("memory").agg(
        top1_acc=("top1_correct", "mean"),
        hit_at_k=("hit_at_k", "mean"),
        path_precision=("path_precision", "mean"),
        wrong_path_hits=("wrong_path_hits", "mean"),
        conflict_hits=("conflict_hits", "mean"),
        n=("query", "count"),
    ).reset_index()
    return summary, overall


def main():
    started = time.time()
    flat, tree = FlatMemory(), TreeMemory()
    for fact in seed_facts():
        flat.add(Fact(**asdict(fact)))
        tree.add(Fact(**asdict(fact)))

    rows = []
    rows += eval_memory("flat", flat, INITIAL_QUERIES)
    rows += eval_memory("tree", tree, INITIAL_QUERIES)

    for fact in update_facts(len(seed_facts()) + 1):
        flat.add(Fact(**asdict(fact)))
        tree.add(Fact(**asdict(fact)))

    rows += eval_memory("flat", flat, UPDATE_QUERIES)
    rows += eval_memory("tree", tree, UPDATE_QUERIES)

    df = pd.DataFrame(rows)
    summary, overall = summarize(df)
    print("\nDetailed retrieval results:")
    print(df[["memory", "phase", "query", "expected_answer", "top1_answer", "top1_correct", "hit_at_k", "path_precision", "conflict_hits"]].to_string(index=False))
    print("\nSummary by phase:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\nOverall summary:")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    flat_o = overall[overall.memory == "flat"].iloc[0]
    tree_o = overall[overall.memory == "tree"].iloc[0]
    checks = {
        "tree_top1_beats_flat": bool(tree_o.top1_acc > flat_o.top1_acc),
        "tree_path_precision_beats_flat": bool(tree_o.path_precision > flat_o.path_precision),
        "tree_conflicts_below_flat": bool(tree_o.conflict_hits < flat_o.conflict_hits),
        "tree_wrong_paths_below_flat": bool(tree_o.wrong_path_hits < flat_o.wrong_path_hits),
    }
    checks["final_pass"] = all(checks.values())

    print("\nPrediction checks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"\nFinal external Tree Memory verdict: {'PASS' if checks['final_pass'] else 'FAIL'}")

    payload = {
        "config": {"TOP_K": TOP_K},
        "rows": df.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "overall": overall.to_dict(orient="records"),
        "checks": checks,
        "runtime_sec": round(time.time() - started, 3),
    }
    with open("external_tree_memory_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Saved external_tree_memory_results.json")


if __name__ == "__main__":
    main()
