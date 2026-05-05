import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


TOP_K = 6
BEAM_WIDTH = 4
FALLBACK_K = 3
TREE_MAX_RETURN = 4


STOPWORDS = {
    "what", "does", "do", "the", "a", "an", "is", "are", "as", "with", "and", "or",
    "in", "on", "to", "for", "of", "their", "now", "after", "about", "not", "asking",
    "which", "where", "who", "whose", "from", "into", "use", "uses", "used", "at",
    "called", "tell", "me", "thing", "kind", "type", "still",
}


PATH_ALIASES = {
    "artifacts/computing/python_code": "python code programming language lists indentation packages uv syntax",
    "living/reptiles/python_snake": "python snake reptile scales shed skin constrict animal",
    "organizations/companies/apple_company": "apple company iphone mac chip ios technology products",
    "living/plants/apple_fruit": "apple fruit orchard tree edible green granny smith",
    "artifacts/vehicles/jaguar_car": "jaguar car vehicle luxury electric brand plan",
    "living/mammals/jaguar_animal": "jaguar animal mammal rainforest predator spotted coats",
    "science/astronomy/mercury_planet": "mercury planet astronomy sun orbit crater closest",
    "science/chemistry/mercury_element": "mercury element chemistry metal liquid hg symbol",
    "artifacts/computing/java_language": "java language programming jvm bytecode class runtime",
    "places/indonesia/java_island": "java island indonesia jakarta volcano geography",
    "places/countries/turkey_country": "turkey country ankara istanbul nation capital",
    "living/birds/turkey_bird": "turkey bird feathers gobble animal sound",
    "culture/food/turkey_food": "turkey food roast thanksgiving meat meal",
    "finance/bank_company": "bank finance money deposits loans account",
    "geography/river_bank": "bank river shore water edge erosion land",
    "living/fish/bass_fish": "bass fish lake freshwater swim habitat",
    "culture/music/bass_instrument": "bass instrument music guitar low notes sound",
    "artifacts/machines/crane_machine": "crane machine construction lifting steel beams",
    "living/birds/crane_bird": "crane bird long legs wetland habitat",
    "artifacts/vehicles/car_tires": "car tires rubber road grip michelin bridgestone vehicle maker",
    "artifacts/vehicles/car_engine": "car engine electric combustion motor tesla vehicle power",
    "culture/food/restaurants": "michelin stars restaurant awards dining food",
}


@dataclass
class Fact:
    id: int
    path: str
    text: str
    slot: str
    answer: str
    tags: str = ""
    active: bool = True
    version: int = 1


@dataclass
class Query:
    task: str
    text: str
    path: str
    slot: str
    answer: str


def toks(text):
    return {t for t in re.findall(r"[a-z0-9+#]+", text.lower()) if t not in STOPWORDS}


def path_parts(path):
    return path.split("/")


def same_branch(path_a, path_b):
    a, b = path_parts(path_a), path_parts(path_b)
    return len(a) > 1 and len(b) > 1 and a[0] == b[0]


def fact_blob(fact, include_alias=False):
    parts = [fact.text, fact.tags]
    if include_alias:
        parts.append(PATH_ALIASES.get(fact.path, ""))
        parts.append(fact.path.replace("/", " "))
    return " ".join(parts)


def lexical_score(query, fact, include_alias=False):
    q, f = toks(query), toks(fact_blob(fact, include_alias))
    if not q or not f:
        return 0.0
    common = q & f
    return sum(1.0 for _ in common) + 0.25 * len(common) / len(q)


class FlatMemory:
    def __init__(self):
        self.facts = []

    def add(self, fact):
        # Flat memory appends updates; old conflicting facts remain available.
        self.facts.append(fact)

    def retrieve(self, query, top_k=TOP_K):
        total = max(1, len(self.facts))
        scored = []
        for pos, fact in enumerate(self.facts):
            if not fact.active:
                continue
            recency = 0.02 * (pos + 1) / total
            scored.append((lexical_score(query, fact, include_alias=False) + recency, -fact.id, fact))
        return [f for s, _, f in sorted(scored, reverse=True)[:top_k] if s > 0]


class HybridTreeMemory:
    def __init__(self):
        self.facts = []
        self.node_words = {}

    def prefixes(self, path):
        parts = path.split("/")
        return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]

    def add(self, fact):
        # Local update: only same path + same slot is replaced.
        version = 1
        for old in self.facts:
            if old.active and old.path == fact.path and old.slot == fact.slot:
                old.active = False
                version = max(version, old.version + 1)
        fact.version = version
        self.facts.append(fact)
        words = toks(fact_blob(fact, include_alias=True))
        for prefix in self.prefixes(fact.path):
            self.node_words.setdefault(prefix, set()).update(words)

    def active_paths(self):
        return sorted({f.path for f in self.facts if f.active})

    def path_score(self, query, path):
        q = toks(query)
        alias = toks(PATH_ALIASES.get(path, ""))
        local_words = set(alias)
        for fact in self.facts:
            if fact.active and fact.path == path:
                local_words.update(toks(fact_blob(fact, include_alias=True)))
        direct = len(q & alias)
        local = len(q & local_words)
        rare_bonus = sum(1.0 for t in q & alias if sum(t in toks(PATH_ALIASES.get(p, "")) for p in self.active_paths()) <= 2)
        return 2.5 * direct + 0.8 * local + rare_bonus + 0.05 * path.count("/")

    def route(self, query, beam_width=BEAM_WIDTH):
        scored = [(self.path_score(query, path), path) for path in self.active_paths()]
        scored.sort(reverse=True)
        return [p for s, p in scored[:beam_width] if s > 0]

    def flat_candidates(self, query, limit=FALLBACK_K):
        scored = [(lexical_score(query, f, include_alias=True), -f.id, f) for f in self.facts if f.active]
        return [f for s, _, f in sorted(scored, reverse=True)[:limit] if s > 0]

    def retrieve(self, query, top_k=TOP_K):
        # Best tree: beam route to several leaves, retrieve local facts, add fallback, rerank globally.
        paths = self.route(query)
        candidates = []
        for fact in self.facts:
            if fact.active and fact.path in paths:
                candidates.append(fact)
        candidates.extend(self.flat_candidates(query))
        unique = {f.id: f for f in candidates}.values()
        path_rank = {p: i for i, p in enumerate(paths)}
        scored = []
        for fact in unique:
            route_bonus = 2.0 / (1 + path_rank[fact.path]) if fact.path in path_rank else 0.25
            exact_path_bonus = 0.5 if fact.path == (paths[0] if paths else "") else 0.0
            s = lexical_score(query, fact, include_alias=True) + route_bonus + exact_path_bonus + 0.2 * fact.version
            scored.append((s, -fact.version, -fact.id, fact))
        ranked = sorted(scored, reverse=True)
        if not ranked:
            return []
        best = ranked[0][0]
        limit = min(top_k, TREE_MAX_RETURN)
        return [f for s, _, __, f in ranked[:limit] if s > 0 and s >= best - 2.5]


def fact(i, path, text, slot, answer, tags=""):
    return Fact(i, path, text, slot, answer, tags)


def seed_facts():
    raw = [
        ("artifacts/computing/python_code", "Python lists use square brackets.", "python.lists", "square brackets", "programming syntax list"),
        ("artifacts/computing/python_code", "Python code blocks use indentation.", "python.blocks", "indentation", "programming whitespace"),
        ("living/reptiles/python_snake", "Python snakes shed their skin.", "python_snake.skin", "shed skin", "reptile scales animal"),
        ("living/reptiles/python_snake", "Pythons constrict prey with strong muscles.", "python_snake.hunt", "constrict prey", "snake hunting"),
        ("organizations/companies/apple_company", "Apple makes iPhones and Mac computers.", "apple_company.products", "iPhones and Mac computers", "technology company products"),
        ("organizations/companies/apple_company", "Apple chips power modern Macs.", "apple_company.chips", "Apple chips", "technology silicon"),
        ("living/plants/apple_fruit", "Apple fruit grows on orchard trees.", "apple_fruit.origin", "orchard trees", "fruit plant orchard"),
        ("living/plants/apple_fruit", "Granny Smith apples are green.", "apple_fruit.color", "green", "fruit color"),
        ("artifacts/vehicles/jaguar_car", "Jaguar cars are luxury vehicles.", "jaguar_car.type", "luxury vehicles", "automobile brand"),
        ("artifacts/vehicles/jaguar_car", "Jaguar vehicles use electric design plans.", "jaguar_car.plan", "electric design plans", "vehicle plan"),
        ("living/mammals/jaguar_animal", "Jaguars hunt in rainforests.", "jaguar_animal.habitat", "rainforests", "animal habitat"),
        ("living/mammals/jaguar_animal", "Jaguars have spotted coats.", "jaguar_animal.body", "spotted coats", "animal coat"),
        ("science/astronomy/mercury_planet", "Mercury is the closest planet to the Sun.", "mercury_planet.position", "closest planet to the Sun", "astronomy planet"),
        ("science/astronomy/mercury_planet", "Mercury has a heavily cratered surface.", "mercury_planet.surface", "heavily cratered surface", "astronomy surface"),
        ("science/chemistry/mercury_element", "Mercury is a liquid metal at room temperature.", "mercury_element.state", "liquid metal", "chemistry metal"),
        ("science/chemistry/mercury_element", "The chemical symbol for mercury is Hg.", "mercury_element.symbol", "Hg", "chemistry symbol"),
        ("artifacts/computing/java_language", "Java programs run on the JVM.", "java_language.runtime", "JVM", "programming runtime"),
        ("artifacts/computing/java_language", "Java classes compile to bytecode.", "java_language.output", "bytecode", "programming compile"),
        ("places/indonesia/java_island", "Java island is in Indonesia.", "java_island.country", "Indonesia", "geography island"),
        ("places/indonesia/java_island", "Jakarta is located on Java island.", "java_island.city", "Jakarta", "geography city"),
        ("places/countries/turkey_country", "Turkey has Ankara as its capital.", "turkey_country.capital", "Ankara", "country capital"),
        ("living/birds/turkey_bird", "Turkey birds have feathers and gobble.", "turkey_bird.sound", "gobble", "bird sound"),
        ("culture/food/turkey_food", "Roast turkey is served at Thanksgiving meals.", "turkey_food.context", "Thanksgiving meals", "food meal"),
        ("finance/bank_company", "Banks hold deposits and provide loans.", "bank_company.function", "deposits and loans", "finance money"),
        ("geography/river_bank", "A river bank is the land beside water.", "river_bank.meaning", "land beside water", "river geography"),
        ("living/fish/bass_fish", "Bass fish swim in freshwater lakes.", "bass_fish.habitat", "freshwater lakes", "fish habitat"),
        ("culture/music/bass_instrument", "A bass guitar plays low musical notes.", "bass_instrument.sound", "low musical notes", "music instrument"),
        ("artifacts/machines/crane_machine", "A crane machine lifts heavy steel beams.", "crane_machine.use", "lifts heavy steel beams", "construction machine"),
        ("living/birds/crane_bird", "Crane birds have long legs and live near wetlands.", "crane_bird.habitat", "wetlands", "bird habitat"),
        ("artifacts/vehicles/car_tires", "Michelin produces premium car tires.", "car_tires.maker", "Michelin", "vehicle tires maker"),
        ("artifacts/vehicles/car_tires", "Car tires use rubber compounds for road grip.", "car_tires.material", "rubber compounds", "vehicle rubber"),
        ("artifacts/vehicles/car_engine", "Tesla car engines use electric motors.", "car_engine.power", "electric motors", "vehicle engine"),
        ("artifacts/vehicles/car_engine", "Combustion engines burn fuel to produce motion.", "car_engine.combustion", "burn fuel", "vehicle engine"),
        ("culture/food/restaurants", "Michelin stars are restaurant awards for excellent dining.", "michelin_star.meaning", "restaurant awards", "restaurant food"),
    ]
    return [fact(i + 1, *row) for i, row in enumerate(raw)]


def update_facts(start):
    raw = [
        ("artifacts/vehicles/car_tires", "Bridgestone produces premium car tires in the updated memory.", "car_tires.maker", "Bridgestone", "vehicle tires maker update"),
        ("organizations/companies/apple_company", "Apple's newest chip in the updated memory is the M5 chip.", "apple_company.chips", "M5 chip", "technology company chip update"),
        ("artifacts/computing/python_code", "Python packages are now installed with uv in the updated memory.", "python.packages", "uv", "programming package update"),
        ("artifacts/vehicles/jaguar_car", "Jaguar's updated vehicle plan emphasizes electric cars.", "jaguar_car.plan", "electric cars", "vehicle update"),
    ]
    return [fact(start + i, *row) for i, row in enumerate(raw)]


QUERIES = [
    Query("ambiguity", "In programming, what syntax do Python lists use?", "artifacts/computing/python_code", "python.lists", "square brackets"),
    Query("ambiguity", "For the reptile python with scales, what does it shed?", "living/reptiles/python_snake", "python_snake.skin", "shed skin"),
    Query("ambiguity", "For Apple the technology company, what products does it make?", "organizations/companies/apple_company", "apple_company.products", "iPhones and Mac computers"),
    Query("ambiguity", "For apple fruit in an orchard, where does it grow?", "living/plants/apple_fruit", "apple_fruit.origin", "orchard trees"),
    Query("ambiguity", "For Jaguar the vehicle brand, what kind of cars are they?", "artifacts/vehicles/jaguar_car", "jaguar_car.type", "luxury vehicles"),
    Query("ambiguity", "For jaguar the rainforest animal, where does it hunt?", "living/mammals/jaguar_animal", "jaguar_animal.habitat", "rainforests"),
    Query("noisy", "I mean Mercury in astronomy, not chemistry: which one is closest to the Sun?", "science/astronomy/mercury_planet", "mercury_planet.position", "closest planet to the Sun"),
    Query("noisy", "I mean mercury in chemistry, not the planet: what is its symbol?", "science/chemistry/mercury_element", "mercury_element.symbol", "Hg"),
    Query("noisy", "For Java as programming, not the island, what runtime is used?", "artifacts/computing/java_language", "java_language.runtime", "JVM"),
    Query("noisy", "For Java as an island, not programming, what country is it in?", "places/indonesia/java_island", "java_island.country", "Indonesia"),
    Query("noisy", "For a financial bank, not a river edge, what does it provide?", "finance/bank_company", "bank_company.function", "deposits and loans"),
    Query("noisy", "For a river bank, not finance, what does it mean?", "geography/river_bank", "river_bank.meaning", "land beside water"),
    Query("context_efficiency", "What sound does the turkey bird make?", "living/birds/turkey_bird", "turkey_bird.sound", "gobble"),
    Query("context_efficiency", "What is Turkey's capital?", "places/countries/turkey_country", "turkey_country.capital", "Ankara"),
    Query("context_efficiency", "What notes does a bass guitar play?", "culture/music/bass_instrument", "bass_instrument.sound", "low musical notes"),
    Query("context_efficiency", "Where do bass fish swim?", "living/fish/bass_fish", "bass_fish.habitat", "freshwater lakes"),
    Query("context_efficiency", "What does a crane machine lift?", "artifacts/machines/crane_machine", "crane_machine.use", "lifts heavy steel beams"),
    Query("context_efficiency", "Where do crane birds live?", "living/birds/crane_bird", "crane_bird.habitat", "wetlands"),
]


UPDATE_QUERIES = [
    Query("local_update", "Who produces premium car tires now?", "artifacts/vehicles/car_tires", "car_tires.maker", "Bridgestone"),
    Query("local_update", "What is Apple's newest chip now?", "organizations/companies/apple_company", "apple_company.chips", "M5 chip"),
    Query("local_update", "What installs Python packages now?", "artifacts/computing/python_code", "python.packages", "uv"),
    Query("local_update", "What is Jaguar's updated vehicle plan?", "artifacts/vehicles/jaguar_car", "jaguar_car.plan", "electric cars"),
    Query("conflict_isolation", "What are Michelin stars, not tires?", "culture/food/restaurants", "michelin_star.meaning", "restaurant awards"),
    Query("conflict_isolation", "What color are Granny Smith apples after the company chip update?", "living/plants/apple_fruit", "apple_fruit.color", "green"),
    Query("conflict_isolation", "What does the Python snake shed after the package update?", "living/reptiles/python_snake", "python_snake.skin", "shed skin"),
    Query("conflict_isolation", "Where do jaguar animals hunt after the vehicle update?", "living/mammals/jaguar_animal", "jaguar_animal.habitat", "rainforests"),
    Query("conflict_isolation", "What power source do Tesla car engines use after the tire update?", "artifacts/vehicles/car_engine", "car_engine.power", "electric motors"),
]


def evaluate(name, memory, queries):
    rows = []
    for q in queries:
        got = memory.retrieve(q.text, TOP_K)
        answers = [f.answer for f in got]
        paths = [f.path for f in got]
        slots = [f.slot for f in got]
        rank = next((i + 1 for i, a in enumerate(answers) if a == q.answer), None)
        rows.append({
            "memory": name,
            "task": q.task,
            "query": q.text,
            "expected_answer": q.answer,
            "expected_path": q.path,
            "top1_answer": answers[0] if answers else None,
            "top1_path": paths[0] if paths else None,
            "top1_correct": bool(answers and answers[0] == q.answer),
            "hit_at_k": q.answer in answers,
            "answer_rank": rank,
            "context_items_to_answer": rank if rank is not None else TOP_K + 1,
            "path_precision": sum(p == q.path for p in paths) / max(1, len(paths)),
            "wrong_path_hits": sum(p != q.path for p in paths),
            "wrong_branch_hits": sum(not same_branch(p, q.path) for p in paths),
            "conflict_hits": sum(s == q.slot and a != q.answer for s, a in zip(slots, answers)),
            "retrieved": " | ".join(f"{f.path}: {f.answer}" for f in got),
        })
    return rows


def build_memories():
    flat, tree = FlatMemory(), HybridTreeMemory()
    for f in seed_facts():
        flat.add(Fact(**asdict(f)))
        tree.add(Fact(**asdict(f)))
    return flat, tree


def summarize(df):
    metrics = dict(
        top1_acc=("top1_correct", "mean"),
        hit_at_k=("hit_at_k", "mean"),
        avg_context_items=("context_items_to_answer", "mean"),
        path_precision=("path_precision", "mean"),
        wrong_path_hits=("wrong_path_hits", "mean"),
        wrong_branch_hits=("wrong_branch_hits", "mean"),
        conflict_hits=("conflict_hits", "mean"),
        n=("query", "count"),
    )
    by_task = df.groupby(["task", "memory"]).agg(**metrics).reset_index()
    overall = df.groupby("memory").agg(**metrics).reset_index()
    return by_task, overall


def checks(by_task, overall):
    def val(table, task, mem, col):
        row = table[(table.task == task) & (table.memory == mem)].iloc[0]
        return float(row[col])

    flat_o, tree_o = overall[overall.memory == "flat"].iloc[0], overall[overall.memory == "hybrid_tree"].iloc[0]
    out = {
        "task1_ambiguity_top1_tree_ge_flat": val(by_task, "ambiguity", "hybrid_tree", "top1_acc") >= val(by_task, "ambiguity", "flat", "top1_acc"),
        "task2_noisy_top1_tree_ge_flat": val(by_task, "noisy", "hybrid_tree", "top1_acc") >= val(by_task, "noisy", "flat", "top1_acc"),
        "task3_local_update_conflicts_tree_lt_flat": val(by_task, "local_update", "hybrid_tree", "conflict_hits") < val(by_task, "local_update", "flat", "conflict_hits"),
        "task4_conflict_isolation_tree_ge_flat": val(by_task, "conflict_isolation", "hybrid_tree", "top1_acc") >= val(by_task, "conflict_isolation", "flat", "top1_acc"),
        "task5_context_efficiency_tree_le_flat": val(by_task, "context_efficiency", "hybrid_tree", "avg_context_items") <= val(by_task, "context_efficiency", "flat", "avg_context_items"),
        "overall_tree_path_precision_gt_flat": float(tree_o.path_precision) > float(flat_o.path_precision),
        "overall_tree_wrong_branches_lt_flat": float(tree_o.wrong_branch_hits) < float(flat_o.wrong_branch_hits),
    }
    out["final_pass"] = all(out.values())
    return out


def main():
    started = time.time()
    flat, tree = build_memories()
    rows = evaluate("flat", flat, QUERIES) + evaluate("hybrid_tree", tree, QUERIES)
    for f in update_facts(1000):
        flat.add(Fact(**asdict(f)))
        tree.add(Fact(**asdict(f)))
    rows += evaluate("flat", flat, UPDATE_QUERIES)
    rows += evaluate("hybrid_tree", tree, UPDATE_QUERIES)

    df = pd.DataFrame(rows)
    by_task, overall = summarize(df)
    ck = checks(by_task, overall)

    print("\nDetailed results:")
    print(df[["memory", "task", "query", "expected_answer", "top1_answer", "top1_correct", "hit_at_k", "answer_rank", "path_precision", "conflict_hits"]].to_string(index=False))
    print("\nTask summary:")
    print(by_task.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\nOverall summary:")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\n5-task checks:")
    for k, v in ck.items():
        print(f"  {k}: {v}")
    print(f"\nFinal Best Tree vs Flat verdict: {'PASS' if ck['final_pass'] else 'FAIL'}")

    payload = {
        "config": {"TOP_K": TOP_K, "BEAM_WIDTH": BEAM_WIDTH, "FALLBACK_K": FALLBACK_K, "TREE_MAX_RETURN": TREE_MAX_RETURN},
        "rows": df.to_dict(orient="records"),
        "by_task": by_task.to_dict(orient="records"),
        "overall": overall.to_dict(orient="records"),
        "checks": ck,
        "runtime_sec": round(time.time() - started, 3),
    }
    results_path = Path(__file__).resolve().parents[1] / "artifacts" / "results" / "best_tree_vs_flat_5tasks_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
