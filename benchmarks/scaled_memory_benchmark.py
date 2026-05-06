import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


TOP_K = 8
BEAM_WIDTH = 4
FALLBACK_K = 3
TREE_MAX_RETURN = 4
TREE_SCORE_WINDOW = 1.75
GATE_MIN_SCORE = 5.0
GATE_MARGIN = 2.0


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into",
    "is", "it", "me", "not", "now", "of", "on", "or", "the", "their", "to",
    "use", "used", "uses", "what", "where", "which", "who", "with", "after",
}


SENSES = [
    ("python", "programming", "artifacts/computing/python_code", "artifacts", ("automation scripts", "indentation", "uv package installer")),
    ("python", "reptile", "living/reptiles/python_snake", "living", ("large snake", "shed skin", "warm forests")),
    ("java", "programming", "artifacts/computing/java_language", "artifacts", ("JVM applications", "bytecode", "class files")),
    ("java", "island", "places/indonesia/java_island", "places", ("Indonesian island", "Jakarta", "volcanoes")),
    ("java", "coffee", "culture/food/java_coffee", "culture", ("coffee drink", "roasted beans", "caffeine")),
    ("apple", "technology company", "organizations/companies/apple_company", "organizations", ("iPhones and Macs", "M5 chip", "Cupertino")),
    ("apple", "fruit", "living/plants/apple_fruit", "living", ("edible fruit", "green skin", "orchard trees")),
    ("mercury", "astronomy", "science/astronomy/mercury_planet", "science", ("closest planet to the Sun", "craters", "eighty eight day orbit")),
    ("mercury", "chemistry", "science/chemistry/mercury_element", "science", ("liquid metal", "Hg symbol", "thermometers")),
    ("jaguar", "vehicle brand", "artifacts/vehicles/jaguar_car", "artifacts", ("luxury vehicles", "electric cars", "British brand")),
    ("jaguar", "rainforest mammal", "living/mammals/jaguar_animal", "living", ("rainforest predator", "spotted coat", "powerful bite")),
    ("bank", "finance", "organizations/finance/bank_company", "organizations", ("deposits and loans", "interest rates", "checking accounts")),
    ("bank", "river geography", "places/geography/river_bank", "places", ("land beside water", "erosion", "shore edge")),
    ("bass", "music", "culture/music/bass_instrument", "culture", ("low musical notes", "four strings", "rhythm section")),
    ("bass", "fish", "living/fish/bass_fish", "living", ("freshwater fish", "lake habitat", "silver scales")),
    ("crane", "construction machine", "artifacts/machines/crane_machine", "artifacts", ("lifts heavy steel", "tall boom", "building sites")),
    ("crane", "wetland bird", "living/birds/crane_bird", "living", ("wetland bird", "long legs", "marsh habitat")),
    ("ruby", "programming", "artifacts/computing/ruby_language", "artifacts", ("web applications", "blocks", "gems")),
    ("ruby", "gemstone", "science/minerals/ruby_gem", "science", ("red gemstone", "corundum", "jewelry")),
    ("rust", "programming", "artifacts/computing/rust_language", "artifacts", ("memory safety", "ownership", "cargo tool")),
    ("rust", "corrosion", "science/chemistry/rust_corrosion", "science", ("iron oxide", "orange flakes", "moisture")),
    ("swift", "programming", "artifacts/computing/swift_language", "artifacts", ("iOS apps", "optionals", "Xcode")),
    ("swift", "flying bird", "living/birds/swift_bird", "living", ("fast flight", "narrow wings", "aerial feeding")),
    ("shell", "command line", "artifacts/computing/shell_terminal", "artifacts", ("terminal commands", "scripts", "environment variables")),
    ("shell", "energy company", "organizations/companies/shell_company", "organizations", ("energy products", "fuel stations", "petrochemicals")),
    ("shell", "sea object", "science/marine/sea_shell", "science", ("protective covering", "calcium carbonate", "beach finds")),
    ("mars", "astronomy", "science/astronomy/mars_planet", "science", ("red planet", "Olympus Mons", "thin atmosphere")),
    ("mars", "candy brand", "culture/food/mars_bar", "culture", ("chocolate bar", "caramel", "confectionery")),
    ("amazon", "technology company", "organizations/companies/amazon_company", "organizations", ("cloud commerce", "AWS", "Seattle")),
    ("amazon", "river", "places/geography/amazon_river", "places", ("large river", "rainforest basin", "South America")),
    ("delta", "airline", "organizations/companies/delta_airline", "organizations", ("air travel", "Atlanta hub", "flights")),
    ("delta", "mathematics", "science/math/delta_symbol", "science", ("change symbol", "difference value", "triangle letter")),
    ("delta", "river landform", "places/geography/river_delta", "places", ("river mouth landform", "sediment", "fan shape")),
    ("oracle", "database company", "organizations/companies/oracle_company", "organizations", ("database software", "SQL systems", "enterprise cloud")),
    ("oracle", "prophecy source", "culture/mythology/oracle_prophecy", "culture", ("prophecy source", "temple advice", "ancient ritual")),
    ("saturn", "planet", "science/astronomy/saturn_planet", "science", ("ringed planet", "Titan moon", "gas giant")),
    ("saturn", "car brand", "artifacts/vehicles/saturn_car", "artifacts", ("former car brand", "plastic panels", "GM division")),
]


SLOT_LABELS = {
    "role": "main role",
    "marker": "distinctive marker",
    "association": "associated value",
}


SLOT_HINTS = {
    "role": {
        "role", "main", "kind", "type", "what", "which", "provide", "provides",
        "make", "makes", "made", "used", "applications", "animal", "material",
        "drink", "vehicles", "lift", "lifts", "closest", "planet", "company",
        "products", "scripts", "purpose",
    },
    "marker": {
        "marker", "distinctive", "feature", "formatting", "compiled", "symbol",
        "color", "colour", "shed", "sheds", "coat", "pattern", "strings",
        "legs", "chip", "city", "major", "craters", "beans", "money", "erosion",
        "boom", "part", "body", "chemical", "habitat", "live", "lives", "where",
        "lake", "made", "from",
    },
    "association": {
        "association", "value", "orbit", "tool", "installer", "package",
        "packages", "based", "located", "country", "forests", "water",
        "notes", "runtime",
    },
}


@dataclass
class Fact:
    id: int
    surface: str
    domain: str
    path: str
    branch: str
    slot: str
    answer: str
    text: str
    tags: str
    active: bool = True
    version: int = 1
    supersedes: int | None = None


@dataclass
class Query:
    task: str
    text: str
    surface: str
    domain: str
    path: str
    branch: str
    slot: str
    answer: str


def toks(text):
    return {t for t in re.findall(r"[a-z0-9+#]+", text.lower()) if t not in STOPWORDS}


def negative_terms(text):
    out = set()
    for match in re.finditer(r"\b(?:not|without|except)\b\s+([^,.;:?]+)", text.lower()):
        phrase = match.group(1)
        for term in re.findall(r"[a-z0-9+#]+", phrase):
            if term not in STOPWORDS:
                out.add(term)
    return out


def fact_blob(fact, include_path=False):
    parts = [fact.text, fact.tags, fact.surface, fact.domain, fact.slot]
    if include_path:
        parts.append(fact.path.replace("/", " "))
    return " ".join(parts)


def score_text(query, fact, include_path=False):
    q = toks(query)
    f = toks(fact_blob(fact, include_path=include_path))
    if not q or not f:
        return 0.0
    overlap = q & f
    neg = negative_terms(query) & f
    return len(overlap) + 0.35 * len(overlap) / len(q) - 2.5 * len(neg)


def slot_score(query, fact):
    q = toks(query)
    hints = SLOT_HINTS.get(fact.slot, set())
    if not q or not hints:
        return 0.0
    score = 1.35 * len(q & hints)
    text = query.lower()
    if fact.slot == "role" and re.search(r"\bwhat (kind|type)\b|\bwhat does\b|\bwhat are\b|\bwhat is\b", text):
        score += 1.0
    if fact.slot == "marker" and re.search(r"\b(symbol|color|colour|shed|compiled|feature|pattern|strings|legs|chip)\b", text):
        score += 2.0
    if fact.slot == "marker" and re.search(r"\b(city|habitat|where|live|lives|made from|beans|erosion|craters)\b", text):
        score += 2.0
    if fact.slot == "association" and re.search(r"\b(where|habitat|live|lives|orbit|package|installer|from|made from)\b", text):
        score += 2.0
    return score


def prefixes(path):
    parts = path.split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]


class FlatAppendMemory:
    name = "flat_append"

    def __init__(self):
        self.facts = []

    def add_alias(self, path, alias):
        return None

    def add(self, fact):
        self.facts.append(fact)

    def update(self, fact):
        # A deliberately weak but common failure mode: append new memory,
        # leave stale conflicting memory retrievable.
        self.add(fact)

    def retrieve(self, query, top_k=TOP_K):
        scored = []
        total = max(1, len(self.facts))
        for pos, fact in enumerate(self.facts):
            if not fact.active:
                continue
            recency = 0.03 * (pos + 1) / total
            scored.append((score_text(query, fact, include_path=False) + slot_score(query, fact) + recency, fact.id, fact))
        scored.sort(reverse=True)
        return [fact for score, _, fact in scored[:top_k] if score > 0]


class FlatReplaceMemory(FlatAppendMemory):
    name = "flat_replace"

    def update(self, fact):
        # Stronger flat baseline: it can replace exact path+slot updates, but
        # retrieval is still one global lexical pool.
        for old in self.facts:
            if old.active and old.path == fact.path and old.slot == fact.slot:
                old.active = False
        self.add(fact)


class TreeMemoryBase:
    def __init__(self, name, beam_width, fallback_k):
        self.name = name
        self.beam_width = beam_width
        self.fallback_k = fallback_k
        self.facts = []
        self.path_aliases = {}
        self.path_domains = {}
        self.node_words = {}

    def add_alias(self, path, alias):
        self.path_aliases[path] = alias
        for prefix in prefixes(path):
            self.node_words.setdefault(prefix, set()).update(toks(alias))

    def add(self, fact):
        self.facts.append(fact)
        self.path_domains[fact.path] = fact.domain
        words = toks(fact_blob(fact, include_path=True))
        for prefix in prefixes(fact.path):
            self.node_words.setdefault(prefix, set()).update(words)

    def update(self, fact):
        for old in self.facts:
            if old.active and old.path == fact.path and old.slot == fact.slot:
                old.active = False
        self.add(fact)

    def active_facts(self):
        return [fact for fact in self.facts if fact.active]

    def active_paths(self):
        return sorted({fact.path for fact in self.active_facts()})

    def path_words(self, path):
        words = set(toks(self.path_aliases.get(path, "")))
        for prefix in prefixes(path):
            words.update(self.node_words.get(prefix, set()))
        return words

    def scored_routes(self, query):
        q = toks(query)
        q_lower = query.lower()
        neg = negative_terms(query)
        routes = []
        for path in self.active_paths():
            words = self.path_words(path)
            matched = q & words
            if not matched:
                continue
            alias_hit = q & toks(self.path_aliases.get(path, ""))
            score = len(matched) + 2.0 * len(alias_hit) - 3.0 * len(neg & words)
            domain = self.path_domains.get(path, "")
            if domain and re.search(r"\bin\s+" + re.escape(domain) + r"\b", q_lower):
                score += 5.0
            score += 0.05 * path.count("/")
            routes.append((score, path))
        routes.sort(reverse=True)
        return [(score, path) for score, path in routes[:self.beam_width] if score > -2]

    def route(self, query):
        return [path for _, path in self.scored_routes(query)]

    def fallback(self, query):
        scored = []
        for fact in self.active_facts():
            score = score_text(query, fact, include_path=False)
            if score > 0:
                scored.append((score, fact.id, fact))
        scored.sort(reverse=True)
        return [fact for _, __, fact in scored[:self.fallback_k]]

    def retrieve(self, query, top_k=TOP_K):
        routes = self.route(query)
        route_rank = {path: i for i, path in enumerate(routes)}
        candidates = {}
        for fact in self.active_facts():
            if fact.path in route_rank:
                candidates[fact.id] = fact
        for fact in self.fallback(query):
            candidates[fact.id] = fact

        scored = []
        for fact in candidates.values():
            route_bonus = 2.0 / (1 + route_rank[fact.path]) if fact.path in route_rank else 0.15
            version_bonus = 0.1 * fact.version
            score = score_text(query, fact, include_path=True) + slot_score(query, fact) + route_bonus + version_bonus
            scored.append((score, fact.version, fact.id, fact))
        scored.sort(reverse=True)
        if not scored:
            return []
        best = scored[0][0]
        limit = min(top_k, TREE_MAX_RETURN)
        return [fact for score, _, __, fact in scored[:limit] if score > 0 and score >= best - TREE_SCORE_WINDOW]


class HardTreeMemory(TreeMemoryBase):
    def __init__(self):
        super().__init__("hard_tree", beam_width=1, fallback_k=0)

    def fallback(self, query):
        return []


class HybridTreeMemory(TreeMemoryBase):
    def __init__(self):
        super().__init__("hybrid_tree", beam_width=BEAM_WIDTH, fallback_k=FALLBACK_K)


class GatedHybridTreeMemory(TreeMemoryBase):
    def __init__(self):
        super().__init__("gated_hybrid_tree", beam_width=BEAM_WIDTH, fallback_k=FALLBACK_K)

    def route_is_confident(self, scored_routes):
        if not scored_routes:
            return False
        top_score = scored_routes[0][0]
        second_score = scored_routes[1][0] if len(scored_routes) > 1 else float("-inf")
        return top_score >= GATE_MIN_SCORE and top_score - second_score >= GATE_MARGIN

    def retrieve_from_single_route(self, query, path, top_k):
        candidates = [fact for fact in self.active_facts() if fact.path == path]
        scored = []
        for fact in candidates:
            score = score_text(query, fact, include_path=True) + slot_score(query, fact) + 2.0 + 0.1 * fact.version
            scored.append((score, fact.version, fact.id, fact))
        scored.sort(reverse=True)
        if not scored:
            return []
        best = scored[0][0]
        limit = min(top_k, TREE_MAX_RETURN)
        return [fact for score, _, __, fact in scored[:limit] if score > 0 and score >= best - TREE_SCORE_WINDOW]

    def retrieve(self, query, top_k=TOP_K):
        scored_routes = self.scored_routes(query)
        if self.route_is_confident(scored_routes):
            return self.retrieve_from_single_route(query, scored_routes[0][1], top_k)
        return super().retrieve(query, top_k=top_k)


def make_alias(surface, domain, values):
    return " ".join([surface, domain, *values])


def build_dataset():
    facts = []
    queries = []
    aliases = {}
    fact_id = 1
    senses_by_surface = {}

    for surface, domain, path, branch, values in SENSES:
        aliases[path] = make_alias(surface, domain, values)
        senses_by_surface.setdefault(surface, []).append((surface, domain, path, branch, values))
        for slot, answer in zip(SLOT_LABELS, values):
            label = SLOT_LABELS[slot]
            text = f"{surface} in {domain} has {label} {answer}."
            tags = f"{surface} {domain} {label} {answer}"
            facts.append(Fact(fact_id, surface, domain, path, branch, slot, answer, text, tags))
            queries.append(Query(
                "direct",
                f"For {surface} in {domain}, what is the {label}?",
                surface,
                domain,
                path,
                branch,
                slot,
                answer,
            ))
            fact_id += 1

    noisy = []
    for surface, senses in senses_by_surface.items():
        for i, sense in enumerate(senses):
            _, domain, path, branch, values = sense
            wrong_domain = senses[(i + 1) % len(senses)][1]
            for slot, answer in zip(SLOT_LABELS, values):
                noisy.append(Query(
                    "noisy_disambiguation",
                    f"For {surface} in {domain}, not {wrong_domain}, what is the {SLOT_LABELS[slot]}?",
                    surface,
                    domain,
                    path,
                    branch,
                    slot,
                    answer,
                ))

    updates = []
    update_queries = []
    conflict_queries = []
    for idx, (surface, senses) in enumerate(senses_by_surface.items()):
        update_sense = senses[idx % len(senses)]
        _, domain, path, branch, values = update_sense
        new_answer = f"updated {values[0]}"
        old_answer = values[0]
        updates.append(Fact(
            fact_id,
            surface,
            domain,
            path,
            branch,
            "role",
            new_answer,
            f"{surface} in {domain} now has main role {new_answer}.",
            f"{surface} {domain} main role {new_answer} update",
            version=2,
            supersedes=fact_id - 1,
        ))
        update_queries.append(Query(
            "local_update",
            f"After the update, for {surface} in {domain}, what is the main role?",
            surface,
            domain,
            path,
            branch,
            "role",
            new_answer,
        ))
        fact_id += 1

        if len(senses) > 1:
            other = senses[(idx + 1) % len(senses)]
            _, other_domain, other_path, other_branch, other_values = other
            conflict_queries.append(Query(
                "conflict_isolation",
                f"After the {domain} update, for {surface} in {other_domain}, what is the main role?",
                surface,
                other_domain,
                other_path,
                other_branch,
                "role",
                other_values[0],
            ))
        # Keep the old answer available as a stale-conflict detector.
        updates[-1].tags += f" old_answer {old_answer}"

    all_queries = queries + noisy
    post_update_queries = update_queries + conflict_queries
    return aliases, facts, updates, all_queries, post_update_queries


def build_memories(aliases, facts):
    memories = [
        FlatAppendMemory(),
        FlatReplaceMemory(),
        HardTreeMemory(),
        HybridTreeMemory(),
        GatedHybridTreeMemory(),
    ]
    for memory in memories:
        for path, alias in aliases.items():
            memory.add_alias(path, alias)
        for fact in facts:
            memory.add(Fact(**asdict(fact)))
    return memories


def correct_fact(fact, query):
    return fact.path == query.path and fact.slot == query.slot and fact.answer == query.answer


def evaluate(memory, queries):
    rows = []
    for query in queries:
        retrieved = memory.retrieve(query.text, top_k=TOP_K)
        top = retrieved[0] if retrieved else None
        rank = None
        for i, fact in enumerate(retrieved, 1):
            if correct_fact(fact, query):
                rank = i
                break
        wrong_path_hits = sum(1 for fact in retrieved if fact.path != query.path)
        wrong_branch_hits = sum(1 for fact in retrieved if fact.branch != query.branch)
        same_surface_wrong_path = sum(1 for fact in retrieved if fact.surface == query.surface and fact.path != query.path)
        stale_conflicts = sum(
            1
            for fact in retrieved
            if fact.surface == query.surface
            and fact.path == query.path
            and fact.slot == query.slot
            and fact.answer != query.answer
        )
        denom = max(1, len(retrieved))
        rows.append({
            "memory": memory.name,
            "task": query.task,
            "query": query.text,
            "expected_path": query.path,
            "expected_answer": query.answer,
            "top1_answer": top.answer if top else None,
            "top1_path": top.path if top else None,
            "top1_correct": bool(top and correct_fact(top, query)),
            "hit_at_k": rank is not None,
            "answer_rank": rank,
            "retrieved_count": len(retrieved),
            "path_precision": sum(1 for fact in retrieved if fact.path == query.path) / denom,
            "wrong_path_hits": wrong_path_hits,
            "wrong_branch_hits": wrong_branch_hits,
            "same_surface_wrong_path": same_surface_wrong_path,
            "stale_conflicts": stale_conflicts,
            "context_contamination": (wrong_path_hits + stale_conflicts) / denom,
            "ai_context_risk": (same_surface_wrong_path + stale_conflicts) / denom,
        })
    return rows


def summarize(df):
    metrics = {
        "top1_correct": "mean",
        "hit_at_k": "mean",
        "retrieved_count": "mean",
        "path_precision": "mean",
        "wrong_path_hits": "mean",
        "wrong_branch_hits": "mean",
        "same_surface_wrong_path": "mean",
        "stale_conflicts": "mean",
        "context_contamination": "mean",
        "ai_context_risk": "mean",
    }
    by_task = df.groupby(["task", "memory"], as_index=False).agg(metrics)
    overall = df.groupby("memory", as_index=False).agg(metrics)
    return by_task, overall


def checks(overall):
    rows = {row.memory: row for row in overall.itertuples(index=False)}
    flat = rows["flat_replace"]
    append = rows["flat_append"]
    hard = rows["hard_tree"]
    hybrid = rows["hybrid_tree"]
    gated = rows["gated_hybrid_tree"]
    out = {
        "hybrid_top1_close_to_strong_flat": hybrid.top1_correct >= flat.top1_correct - 0.05,
        "hybrid_path_precision_beats_flat": hybrid.path_precision >= flat.path_precision + 0.25,
        "hybrid_wrong_branches_below_flat": hybrid.wrong_branch_hits <= flat.wrong_branch_hits * 0.6,
        "hybrid_ai_context_risk_below_flat": hybrid.ai_context_risk <= flat.ai_context_risk * 0.6,
        "hybrid_stale_conflicts_below_append": hybrid.stale_conflicts <= append.stale_conflicts,
        "hybrid_beats_hard_on_hit_at_k": hybrid.hit_at_k >= hard.hit_at_k,
        "gated_top1_close_to_hard": gated.top1_correct >= hard.top1_correct - 0.05,
        "gated_context_contamination_below_hybrid": gated.context_contamination <= hybrid.context_contamination,
        "gated_ai_context_risk_below_hybrid": gated.ai_context_risk <= hybrid.ai_context_risk,
        "gated_hit_at_k_ge_hybrid": gated.hit_at_k >= hybrid.hit_at_k,
    }
    out["final_pass"] = all(out.values())
    return out


def save_markdown(path, by_task, overall, ck, dataset_stats):
    def markdown_table(df):
        rows = df.to_dict(orient="records")
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for row in rows:
            cells = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    cells.append(f"{value:.3f}")
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    lines = [
        "# Scaled Memory Benchmark Summary",
        "",
        "## Dataset",
        "",
        f"- Concepts: {dataset_stats['concepts']}",
        f"- Base facts: {dataset_stats['base_facts']}",
        f"- Updates: {dataset_stats['updates']}",
        f"- Queries: {dataset_stats['queries']}",
        "",
        "## Overall",
        "",
        markdown_table(overall),
        "",
        "## By Task",
        "",
        markdown_table(by_task),
        "",
        "## Checks",
        "",
    ]
    lines.extend([f"- {key}: {value}" for key, value in ck.items()])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    started = time.time()
    aliases, facts, updates, pre_queries, post_queries = build_dataset()
    memories = build_memories(aliases, facts)

    rows = []
    for memory in memories:
        rows.extend(evaluate(memory, pre_queries))
    for memory in memories:
        for update in updates:
            memory.update(Fact(**asdict(update)))
        rows.extend(evaluate(memory, post_queries))

    df = pd.DataFrame(rows)
    by_task, overall = summarize(df)
    ck = checks(overall)

    dataset_stats = {
        "concepts": len(aliases),
        "base_facts": len(facts),
        "updates": len(updates),
        "queries": len(pre_queries) + len(post_queries),
    }

    print("\nScaled benchmark dataset:")
    for key, value in dataset_stats.items():
        print(f"  {key}: {value}")
    print("\nOverall summary:")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\nTask summary:")
    print(by_task.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\nFalsification checks:")
    for key, value in ck.items():
        print(f"  {key}: {value}")
    print(f"\nFinal Scaled Memory Benchmark verdict: {'PASS' if ck['final_pass'] else 'FAIL'}")

    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "TOP_K": TOP_K,
            "BEAM_WIDTH": BEAM_WIDTH,
            "FALLBACK_K": FALLBACK_K,
            "TREE_MAX_RETURN": TREE_MAX_RETURN,
            "TREE_SCORE_WINDOW": TREE_SCORE_WINDOW,
            "GATE_MIN_SCORE": GATE_MIN_SCORE,
            "GATE_MARGIN": GATE_MARGIN,
        },
        "dataset": dataset_stats,
        "rows": df.to_dict(orient="records"),
        "by_task": by_task.to_dict(orient="records"),
        "overall": overall.to_dict(orient="records"),
        "checks": ck,
        "runtime_sec": round(time.time() - started, 3),
    }
    json_path = out_dir / "scaled_memory_benchmark_results.json"
    md_path = out_dir / "scaled_memory_benchmark_summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_markdown(md_path, by_task, overall, ck, dataset_stats)
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
