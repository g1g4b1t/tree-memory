import json
import re
import time
from dataclasses import asdict, dataclass, field

import pandas as pd


TOP_K = 6
RECENCY_BONUS = 0.03


PATH_ALIASES = {
    "artifacts/computing/python_code": "python code programming language lists indentation pip packages",
    "living/reptiles/python_snake": "python snake reptile scales shed constrict animal",
    "organizations/companies/apple_company": "apple company iphone mac chip ios technology",
    "living/plants/apple_fruit": "apple fruit orchard tree edible green red",
    "artifacts/vehicles/jaguar_car": "jaguar car vehicle luxury electric brand",
    "living/mammals/jaguar_animal": "jaguar animal mammal rainforest predator spotted",
    "science/astronomy/mercury_planet": "mercury planet astronomy sun orbit crater",
    "science/chemistry/mercury_element": "mercury element chemistry metal liquid hg",
    "artifacts/computing/java_language": "java language programming jvm bytecode class",
    "places/indonesia/java_island": "java island indonesia jakarta volcano geography",
    "places/countries/turkey_country": "turkey country ankara istanbul nation",
    "living/birds/turkey_bird": "turkey bird feathers gobble animal",
    "culture/food/turkey_food": "turkey food roast thanksgiving meat",
    "finance/bank_company": "bank finance money deposits loans account",
    "geography/river_bank": "bank river shore water edge erosion",
    "living/fish/bass_fish": "bass fish lake freshwater swim",
    "culture/music/bass_instrument": "bass instrument music guitar low notes",
    "artifacts/machines/crane_machine": "crane machine construction lifting steel",
    "living/birds/crane_bird": "crane bird long legs wetland",
    "artifacts/vehicles/car_tires": "car tires rubber road grip michelin bridgestone vehicle",
    "artifacts/vehicles/car_engine": "car engine electric combustion motor vehicle",
    "culture/food/restaurants": "michelin stars restaurant awards dining food",
}

STOPWORDS = {
    "what", "does", "do", "the", "a", "an", "is", "are", "as", "with", "and", "or",
    "in", "on", "to", "for", "of", "their", "now", "after", "about", "not", "asking",
    "which", "where", "who", "whose", "from", "into", "use", "uses", "used", "at",
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
    text: str
    path: str
    slot: str
    answer: str
    phase: str
    note: str = ""


def toks(text):
    return {t for t in re.findall(r"[a-z0-9+#]+", text.lower()) if t not in STOPWORDS}


def fact_blob(fact):
    return " ".join([fact.text, fact.path.replace("/", " "), fact.tags, PATH_ALIASES.get(fact.path, "")])


def lexical_score(query, fact):
    q, f = toks(query), toks(fact_blob(fact))
    if not q or not f:
        return 0.0
    overlap = len(q & f)
    return overlap + overlap / max(1, len(q))


class FlatMemory:
    def __init__(self):
        self.facts = []

    def add(self, fact):
        # Flat memory appends updates, so old conflicting facts remain retrievable.
        self.facts.append(fact)

    def retrieve(self, query, top_k=TOP_K):
        scored = []
        total = max(1, len(self.facts))
        for pos, f in enumerate(self.facts):
            if not f.active:
                continue
            s = lexical_score(query, f) + RECENCY_BONUS * (pos + 1) / total
            scored.append((s, -f.id, f))
        return [f for s, _, f in sorted(scored, reverse=True)[:top_k] if s > 0]


class TreeMemory:
    def __init__(self):
        self.facts = []
        self.node_words = {}

    def prefixes(self, path):
        parts = path.split("/")
        return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]

    def add(self, fact):
        # Tree memory performs local replacement only inside the same leaf and slot.
        version = 1
        for old in self.facts:
            if old.active and old.path == fact.path and old.slot == fact.slot:
                old.active = False
                version = max(version, old.version + 1)
        fact.version = version
        self.facts.append(fact)
        words = toks(fact_blob(fact))
        for prefix in self.prefixes(fact.path):
            self.node_words.setdefault(prefix, set()).update(words)

    def active_leaf_paths(self):
        return {f.path for f in self.facts if f.active}

    def route(self, query):
        q = toks(query)
        candidates = []
        active = self.active_leaf_paths()
        for path in active:
            words = set()
            for prefix in self.prefixes(path):
                words.update(self.node_words.get(prefix, set()))
            words.update(toks(PATH_ALIASES.get(path, "")))
            overlap = len(q & words)
            depth_bonus = 0.05 * path.count("/")
            candidates.append((overlap + depth_bonus, path))
        candidates.sort(reverse=True)
        return candidates[0][1] if candidates else ""

    def retrieve(self, query, top_k=TOP_K):
        path = self.route(query)
        if not path:
            return []
        allowed = {path}
        allowed.update(p for p in self.prefixes(path) if p in self.active_leaf_paths())
        local = [f for f in self.facts if f.active and f.path in allowed]
        scored = [(lexical_score(query, f) + 2.0 * (f.path == path), -f.version, -f.id, f) for f in local]
        return [f for s, _, __, f in sorted(scored, reverse=True)[:top_k] if s > 0]


def make_fact(i, path, text, slot, answer, tags=""):
    return Fact(i, path, text, slot, answer, tags)


def seed_facts():
    raw = [
        ("artifacts/computing/python_code", "Python lists use square brackets.", "python.lists", "square brackets", "programming list syntax"),
        ("artifacts/computing/python_code", "Python code blocks use indentation.", "python.blocks", "indentation", "programming whitespace"),
        ("living/reptiles/python_snake", "Python snakes shed their skin.", "python_snake.skin", "shed skin", "reptile animal scales"),
        ("living/reptiles/python_snake", "Pythons constrict prey with strong muscles.", "python_snake.hunt", "constrict prey", "snake animal hunting"),
        ("organizations/companies/apple_company", "Apple makes iPhones and Mac computers.", "apple_company.products", "iPhones and Mac computers", "technology company"),
        ("organizations/companies/apple_company", "Apple chips power modern Macs.", "apple_company.chips", "Apple chips", "technology silicon"),
        ("living/plants/apple_fruit", "Apple fruit grows on orchard trees.", "apple_fruit.origin", "orchard trees", "fruit plant"),
        ("living/plants/apple_fruit", "Granny Smith apples are green.", "apple_fruit.color", "green", "fruit color"),
        ("artifacts/vehicles/jaguar_car", "Jaguar cars are luxury vehicles.", "jaguar_car.type", "luxury vehicles", "automobile brand"),
        ("artifacts/vehicles/jaguar_car", "Jaguar vehicles use electric design plans.", "jaguar_car.plan", "electric design plans", "vehicle electric"),
        ("living/mammals/jaguar_animal", "Jaguars hunt in rainforests.", "jaguar_animal.habitat", "rainforests", "animal habitat"),
        ("living/mammals/jaguar_animal", "Jaguars have spotted coats.", "jaguar_animal.body", "spotted coats", "animal coat"),
        ("science/astronomy/mercury_planet", "Mercury is the closest planet to the Sun.", "mercury_planet.position", "closest planet to the Sun", "astronomy planet"),
        ("science/astronomy/mercury_planet", "Mercury has a heavily cratered surface.", "mercury_planet.surface", "heavily cratered surface", "astronomy surface"),
        ("science/chemistry/mercury_element", "Mercury is a liquid metal at room temperature.", "mercury_element.state", "liquid metal", "chemistry element"),
        ("science/chemistry/mercury_element", "The chemical symbol for mercury is Hg.", "mercury_element.symbol", "Hg", "chemistry symbol"),
        ("artifacts/computing/java_language", "Java programs run on the JVM.", "java_language.runtime", "JVM", "programming runtime"),
        ("artifacts/computing/java_language", "Java classes compile to bytecode.", "java_language.output", "bytecode", "programming compile"),
        ("places/indonesia/java_island", "Java island is in Indonesia.", "java_island.country", "Indonesia", "geography island"),
        ("places/indonesia/java_island", "Jakarta is located on Java island.", "java_island.city", "Jakarta", "geography city"),
        ("places/countries/turkey_country", "Turkey has Ankara as its capital.", "turkey_country.capital", "Ankara", "country capital"),
        ("living/birds/turkey_bird", "Turkey birds have feathers and gobble.", "turkey_bird.sound", "gobble", "bird animal"),
        ("culture/food/turkey_food", "Roast turkey is served at Thanksgiving meals.", "turkey_food.context", "Thanksgiving meals", "food meat"),
        ("finance/bank_company", "Banks hold deposits and provide loans.", "bank_company.function", "deposits and loans", "finance money"),
        ("geography/river_bank", "A river bank is the land beside water.", "river_bank.meaning", "land beside water", "geography river"),
        ("living/fish/bass_fish", "Bass fish swim in freshwater lakes.", "bass_fish.habitat", "freshwater lakes", "fish animal"),
        ("culture/music/bass_instrument", "A bass guitar plays low musical notes.", "bass_instrument.sound", "low musical notes", "music instrument"),
        ("artifacts/machines/crane_machine", "A crane machine lifts heavy steel beams.", "crane_machine.use", "lifts heavy steel beams", "construction machine"),
        ("living/birds/crane_bird", "Crane birds have long legs and live near wetlands.", "crane_bird.habitat", "wetlands", "bird animal"),
        ("artifacts/vehicles/car_tires", "Michelin produces premium car tires.", "car_tires.maker", "Michelin", "vehicle tires maker"),
        ("artifacts/vehicles/car_tires", "Car tires use rubber compounds for road grip.", "car_tires.material", "rubber compounds", "vehicle rubber"),
        ("artifacts/vehicles/car_engine", "Tesla car engines use electric motors.", "car_engine.power", "electric motors", "vehicle engine"),
        ("artifacts/vehicles/car_engine", "Combustion engines burn fuel to produce motion.", "car_engine.combustion", "burn fuel", "vehicle engine"),
    ]
    return [make_fact(i + 1, *row) for i, row in enumerate(raw)]


def update_facts(start_id):
    raw = [
        ("artifacts/vehicles/car_tires", "Bridgestone produces premium car tires in the updated memory.", "car_tires.maker", "Bridgestone", "vehicle tires maker update"),
        ("organizations/companies/apple_company", "Apple's newest chip in the updated memory is the M5 chip.", "apple_company.chips", "M5 chip", "technology company update"),
        ("artifacts/computing/python_code", "Python packages are now installed with uv in the updated memory.", "python.packages", "uv", "programming package update"),
        ("artifacts/vehicles/jaguar_car", "Jaguar's updated vehicle plan emphasizes electric cars.", "jaguar_car.plan", "electric cars", "vehicle update"),
    ]
    return [make_fact(start_id + i, *row) for i, row in enumerate(raw)]


INITIAL_QUERIES = [
    Query("What syntax do Python lists use?", "artifacts/computing/python_code", "python.lists", "square brackets", "initial"),
    Query("What does the Python with scales shed?", "living/reptiles/python_snake", "python_snake.skin", "shed skin", "initial"),
    Query("What does Apple make as a technology company?", "organizations/companies/apple_company", "apple_company.products", "iPhones and Mac computers", "initial"),
    Query("Where does apple fruit grow?", "living/plants/apple_fruit", "apple_fruit.origin", "orchard trees", "initial"),
    Query("What kind of vehicles are Jaguar cars?", "artifacts/vehicles/jaguar_car", "jaguar_car.type", "luxury vehicles", "initial"),
    Query("Where do jaguar animals hunt?", "living/mammals/jaguar_animal", "jaguar_animal.habitat", "rainforests", "initial"),
    Query("Which Mercury is closest to the Sun?", "science/astronomy/mercury_planet", "mercury_planet.position", "closest planet to the Sun", "initial"),
    Query("What is mercury's chemical symbol?", "science/chemistry/mercury_element", "mercury_element.symbol", "Hg", "initial"),
    Query("What runtime do Java programs use?", "artifacts/computing/java_language", "java_language.runtime", "JVM", "initial"),
    Query("Where is Java island located?", "places/indonesia/java_island", "java_island.country", "Indonesia", "initial"),
    Query("What is Turkey's capital?", "places/countries/turkey_country", "turkey_country.capital", "Ankara", "initial"),
    Query("What sound does the turkey bird make?", "living/birds/turkey_bird", "turkey_bird.sound", "gobble", "initial"),
    Query("What does a financial bank provide?", "finance/bank_company", "bank_company.function", "deposits and loans", "initial"),
    Query("What is a river bank?", "geography/river_bank", "river_bank.meaning", "land beside water", "initial"),
    Query("Where do bass fish swim?", "living/fish/bass_fish", "bass_fish.habitat", "freshwater lakes", "initial"),
    Query("What notes does a bass guitar play?", "culture/music/bass_instrument", "bass_instrument.sound", "low musical notes", "initial"),
    Query("What does a crane machine lift?", "artifacts/machines/crane_machine", "crane_machine.use", "lifts heavy steel beams", "initial"),
    Query("Where do crane birds live?", "living/birds/crane_bird", "crane_bird.habitat", "wetlands", "initial"),
    Query("Who produces premium car tires?", "artifacts/vehicles/car_tires", "car_tires.maker", "Michelin", "initial"),
    Query("What powers Tesla car engines?", "artifacts/vehicles/car_engine", "car_engine.power", "electric motors", "initial"),
]


UPDATE_QUERIES = [
    Query("Who produces premium car tires now?", "artifacts/vehicles/car_tires", "car_tires.maker", "Bridgestone", "after_update"),
    Query("What are Michelin tires not asking about restaurant stars?", "artifacts/vehicles/car_tires", "car_tires.maker", "Bridgestone", "after_update"),
    Query("What are Michelin stars?", "culture/food/restaurants", "michelin_star.meaning", "restaurant awards", "after_update"),
    Query("What is Apple's newest chip now?", "organizations/companies/apple_company", "apple_company.chips", "M5 chip", "after_update"),
    Query("What color are Granny Smith apples after the company chip update?", "living/plants/apple_fruit", "apple_fruit.color", "green", "after_update"),
    Query("What installs Python packages now?", "artifacts/computing/python_code", "python.packages", "uv", "after_update"),
    Query("What does the Python snake shed after the package update?", "living/reptiles/python_snake", "python_snake.skin", "shed skin", "after_update"),
    Query("What is Jaguar's updated vehicle plan?", "artifacts/vehicles/jaguar_car", "jaguar_car.plan", "electric cars", "after_update"),
    Query("Where do jaguar animals hunt after the vehicle update?", "living/mammals/jaguar_animal", "jaguar_animal.habitat", "rainforests", "after_update"),
    Query("What power source do Tesla car engines use after the tire update?", "artifacts/vehicles/car_engine", "car_engine.power", "electric motors", "after_update"),
]


def inject_extra_fact(memories):
    fact = Fact(10_000, "culture/food/restaurants", "Michelin stars are restaurant awards for excellent dining.", "michelin_star.meaning", "restaurant awards", "food restaurant")
    for memory in memories:
        memory.add(Fact(**asdict(fact)))


def eval_memory(name, memory, queries):
    rows = []
    for q in queries:
        ret = memory.retrieve(q.text, TOP_K)
        answers = [f.answer for f in ret]
        paths = [f.path for f in ret]
        slots = [f.slot for f in ret]
        rank = next((i + 1 for i, a in enumerate(answers) if a == q.answer), None)
        rows.append({
            "memory": name,
            "phase": q.phase,
            "query": q.text,
            "expected_path": q.path,
            "expected_slot": q.slot,
            "expected_answer": q.answer,
            "top1_answer": answers[0] if answers else None,
            "top1_path": paths[0] if paths else None,
            "top1_correct": bool(answers and answers[0] == q.answer),
            "hit_at_k": q.answer in answers,
            "answer_rank": rank,
            "context_items_to_answer": rank if rank is not None else TOP_K + 1,
            "path_precision": sum(p == q.path for p in paths) / max(1, len(paths)),
            "wrong_branch_hits": sum(p != q.path for p in paths),
            "conflict_hits": sum(s == q.slot and a != q.answer for s, a in zip(slots, answers)),
            "retrieved": " | ".join(f"{f.path}: {f.answer}" for f in ret),
        })
    return rows


def summarize(df):
    metrics = dict(
        top1_acc=("top1_correct", "mean"),
        hit_at_k=("hit_at_k", "mean"),
        avg_context_items=("context_items_to_answer", "mean"),
        path_precision=("path_precision", "mean"),
        wrong_branch_hits=("wrong_branch_hits", "mean"),
        conflict_hits=("conflict_hits", "mean"),
        n=("query", "count"),
    )
    phase = df.groupby(["memory", "phase"]).agg(**metrics).reset_index()
    overall = df.groupby("memory").agg(**metrics).reset_index()
    return phase, overall


def main():
    started = time.time()
    flat, tree = FlatMemory(), TreeMemory()
    for fact in seed_facts():
        flat.add(Fact(**asdict(fact)))
        tree.add(Fact(**asdict(fact)))
    inject_extra_fact([flat, tree])

    rows = eval_memory("flat", flat, INITIAL_QUERIES)
    rows += eval_memory("tree", tree, INITIAL_QUERIES)

    for fact in update_facts(20_000):
        flat.add(Fact(**asdict(fact)))
        tree.add(Fact(**asdict(fact)))

    rows += eval_memory("flat", flat, UPDATE_QUERIES)
    rows += eval_memory("tree", tree, UPDATE_QUERIES)

    df = pd.DataFrame(rows)
    phase, overall = summarize(df)
    print("\nDetailed results:")
    print(df[["memory", "phase", "query", "expected_answer", "top1_answer", "top1_correct", "hit_at_k", "answer_rank", "path_precision", "wrong_branch_hits", "conflict_hits"]].to_string(index=False))
    print("\nSummary by phase:")
    print(phase.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\nOverall summary:")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    flat_o = overall[overall.memory == "flat"].iloc[0]
    tree_o = overall[overall.memory == "tree"].iloc[0]
    checks = {
        "tree_top1_beats_flat": bool(tree_o.top1_acc > flat_o.top1_acc),
        "tree_hit_at_k_at_least_flat": bool(tree_o.hit_at_k >= flat_o.hit_at_k),
        "tree_uses_less_context": bool(tree_o.avg_context_items < flat_o.avg_context_items),
        "tree_path_precision_beats_flat": bool(tree_o.path_precision > flat_o.path_precision),
        "tree_wrong_branches_below_flat": bool(tree_o.wrong_branch_hits < flat_o.wrong_branch_hits),
        "tree_conflicts_below_flat": bool(tree_o.conflict_hits < flat_o.conflict_hits),
    }
    checks["final_pass"] = all(checks.values())
    print("\nPrediction checks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"\nFinal External Tree Memory Test 2 verdict: {'PASS' if checks['final_pass'] else 'FAIL'}")

    payload = {
        "config": {"TOP_K": TOP_K, "RECENCY_BONUS": RECENCY_BONUS, "num_seed_facts": len(seed_facts()), "num_update_facts": len(update_facts(0))},
        "rows": df.to_dict(orient="records"),
        "summary_by_phase": phase.to_dict(orient="records"),
        "overall": overall.to_dict(orient="records"),
        "checks": checks,
        "runtime_sec": round(time.time() - started, 3),
    }
    with open("external_tree_memory_test2_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Saved external_tree_memory_test2_results.json")


if __name__ == "__main__":
    main()
