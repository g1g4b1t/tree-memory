import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "about", "after", "before", "by", "do",
    "does", "for", "from", "in", "into", "is", "it", "me", "not", "now", "of",
    "on", "or", "tell", "the", "their", "to", "use", "used", "uses", "what",
    "where", "which", "who", "whose", "with",
}


@dataclass
class Fact:
    id: int
    path: str
    text: str
    slot: str
    answer: str
    tags: str = ""
    source: str = "user"
    confidence: float = 1.0
    active: bool = True
    version: int = 1
    created_at: float = 0.0
    supersedes: int | None = None


@dataclass
class RouteCandidate:
    path: str
    score: float
    matched_terms: list[str]


@dataclass
class RetrievedFact:
    fact: Fact
    score: float
    route_rank: int | None
    matched_terms: list[str]


class TreeMemory:
    """
    Hybrid tree memory:
    - local updates inside path+slot
    - beam routing to several candidate branches
    - local retrieval plus global fallback
    - final reranking with compact context
    """

    def __init__(self, beam_width=4, fallback_k=3, max_context=5, stopwords=None):
        self.beam_width = beam_width
        self.fallback_k = fallback_k
        self.max_context = max_context
        self.stopwords = set(stopwords or DEFAULT_STOPWORDS)
        self.facts: list[Fact] = []
        self.node_words: dict[str, set[str]] = {}
        self.path_aliases: dict[str, str] = {}
        self._next_id = 1

    def add_alias(self, path: str, alias: str):
        self.path_aliases[path] = alias
        for prefix in self.prefixes(path):
            self.node_words.setdefault(prefix, set()).update(self.tokens(alias))

    def add_fact(self, path: str, text: str, slot: str, answer: str, tags="", source="user", confidence=1.0):
        fact = Fact(
            id=self._next_id,
            path=path,
            text=text,
            slot=slot,
            answer=answer,
            tags=tags,
            source=source,
            confidence=confidence,
            created_at=time.time(),
        )
        self._next_id += 1
        self._insert_fact(fact, replace=False)
        return fact

    def update_fact(self, path: str, slot: str, text: str, answer: str, tags="", source="user", confidence=1.0):
        version, supersedes = 1, None
        for old in self.facts:
            if old.active and old.path == path and old.slot == slot:
                old.active = False
                version = max(version, old.version + 1)
                supersedes = old.id
        fact = Fact(
            id=self._next_id,
            path=path,
            text=text,
            slot=slot,
            answer=answer,
            tags=tags,
            source=source,
            confidence=confidence,
            active=True,
            version=version,
            created_at=time.time(),
            supersedes=supersedes,
        )
        self._next_id += 1
        self._insert_fact(fact, replace=True)
        return fact

    def route(self, query: str, beam_width: int | None = None):
        q = self.tokens(query)
        neg = self.negative_tokens(query)
        candidates = []
        for path in self.active_paths():
            words = self.path_words(path)
            matched = sorted(q & words)
            if not matched:
                continue
            score = 2.5 * len(q & self.tokens(self.path_aliases.get(path, "")))
            score += 1.0 * len(matched)
            score -= 4.0 * len(neg & words)
            score += 0.05 * path.count("/")
            candidates.append(RouteCandidate(path=path, score=score, matched_terms=matched))
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: beam_width or self.beam_width]

    def retrieve(self, query: str, top_k: int | None = None):
        top_k = top_k or self.max_context
        routes = self.route(query)
        route_rank = {r.path: i for i, r in enumerate(routes)}
        candidates = {}

        for fact in self.active_facts():
            if fact.path in route_rank:
                candidates[fact.id] = fact
        for fact in self.global_fallback(query):
            candidates[fact.id] = fact

        scored = []
        best = None
        for fact in candidates.values():
            route_bonus = 2.0 / (1 + route_rank[fact.path]) if fact.path in route_rank else 0.25
            version_bonus = 0.25 * fact.version
            confidence_bonus = 0.1 * fact.confidence
            s, matched = self.fact_score(query, fact)
            total = s + route_bonus + version_bonus + confidence_bonus
            best = total if best is None else max(best, total)
            scored.append(RetrievedFact(fact=fact, score=total, route_rank=route_rank.get(fact.path), matched_terms=matched))

        scored.sort(key=lambda r: (r.score, r.fact.version, r.fact.id), reverse=True)
        if best is None:
            return []
        return [r for r in scored[:top_k] if r.score >= best - 2.5]

    def answer(self, query: str):
        retrieved = self.retrieve(query)
        if not retrieved:
            return {"answer": None, "confidence": 0.0, "path": None, "context": []}
        top = retrieved[0]
        confidence = min(1.0, max(0.0, top.score / (top.score + 3.0)))
        return {
            "answer": top.fact.answer,
            "confidence": round(confidence, 3),
            "path": top.fact.path,
            "slot": top.fact.slot,
            "source_fact_id": top.fact.id,
            "context": [self.retrieved_to_dict(r) for r in retrieved],
        }

    def explain_retrieval(self, query: str):
        routes = self.route(query)
        retrieved = self.retrieve(query)
        return {
            "query": query,
            "routes": [asdict(r) for r in routes],
            "retrieved": [self.retrieved_to_dict(r) for r in retrieved],
            "answer": self.answer(query),
        }

    def save(self, filename: str | Path):
        payload = {
            "beam_width": self.beam_width,
            "fallback_k": self.fallback_k,
            "max_context": self.max_context,
            "path_aliases": self.path_aliases,
            "next_id": self._next_id,
            "facts": [asdict(f) for f in self.facts],
        }
        Path(filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, filename: str | Path):
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        memory = cls(payload["beam_width"], payload["fallback_k"], payload["max_context"])
        memory.path_aliases = payload.get("path_aliases", {})
        memory._next_id = payload.get("next_id", 1)
        for raw in payload["facts"]:
            memory._insert_fact(Fact(**raw), replace=False)
        return memory

    def _insert_fact(self, fact: Fact, replace: bool):
        self.facts.append(fact)
        words = self.tokens(self.fact_blob(fact, include_alias=True))
        for prefix in self.prefixes(fact.path):
            self.node_words.setdefault(prefix, set()).update(words)

    def active_facts(self):
        return [f for f in self.facts if f.active]

    def active_paths(self):
        return sorted({f.path for f in self.active_facts()})

    def prefixes(self, path: str):
        parts = path.split("/")
        return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]

    def tokens(self, text: str):
        return {t for t in re.findall(r"[a-z0-9+#]+", text.lower()) if t not in self.stopwords}

    def path_words(self, path: str):
        words = set(self.tokens(self.path_aliases.get(path, "")))
        for prefix in self.prefixes(path):
            words.update(self.node_words.get(prefix, set()))
        for fact in self.active_facts():
            if fact.path == path:
                words.update(self.tokens(self.fact_blob(fact, include_alias=False)))
        return words

    def fact_blob(self, fact: Fact, include_alias=False):
        parts = [fact.text, fact.tags]
        if include_alias:
            parts += [self.path_aliases.get(fact.path, ""), fact.path.replace("/", " ")]
        return " ".join(parts)

    def fact_score(self, query: str, fact: Fact):
        q = self.tokens(query)
        f = self.tokens(self.fact_blob(fact, include_alias=True))
        neg = self.negative_tokens(query)
        matched = sorted(q & f)
        score = len(matched) + 0.25 * len(matched) / max(1, len(q))
        score -= 3.0 * len(neg & f)
        return score, matched

    def negative_tokens(self, text: str):
        words = re.findall(r"[a-z0-9+#]+", text.lower())
        neg = set()
        for i, word in enumerate(words):
            if word not in {"not", "no", "without"}:
                continue
            picked = 0
            for nxt in words[i + 1:]:
                if nxt in {"and", "or", "but"}:
                    break
                if nxt not in self.stopwords:
                    neg.add(nxt)
                    picked += 1
                if picked >= 3:
                    break
        return neg

    def global_fallback(self, query: str):
        scored = []
        for fact in self.active_facts():
            s, _ = self.fact_score(query, fact)
            if s > 0:
                scored.append((s, fact.id, fact))
        scored.sort(reverse=True)
        return [fact for _, __, fact in scored[: self.fallback_k]]

    def retrieved_to_dict(self, r: RetrievedFact):
        return {
            "fact_id": r.fact.id,
            "path": r.fact.path,
            "slot": r.fact.slot,
            "answer": r.fact.answer,
            "text": r.fact.text,
            "score": round(r.score, 3),
            "route_rank": r.route_rank,
            "matched_terms": r.matched_terms,
            "version": r.fact.version,
            "source": r.fact.source,
        }


def build_demo_memory():
    memory = TreeMemory()
    aliases = {
        "artifacts/vehicles/car_tires": "car tires rubber road grip michelin bridgestone vehicle maker",
        "artifacts/vehicles/car_engine": "car engine electric combustion motor tesla vehicle power",
        "culture/food/restaurants": "michelin stars restaurant awards dining food",
        "artifacts/computing/python_code": "python code programming lists packages uv syntax",
        "living/reptiles/python_snake": "python snake reptile scales shed skin animal",
        "organizations/companies/apple_company": "apple company iphone mac chip technology",
        "living/plants/apple_fruit": "apple fruit orchard tree green granny smith",
        "artifacts/vehicles/jaguar_car": "jaguar car vehicle luxury electric brand",
        "living/mammals/jaguar_animal": "jaguar animal rainforest predator spotted coats",
    }
    for path, alias in aliases.items():
        memory.add_alias(path, alias)

    memory.add_fact("artifacts/vehicles/car_tires", "Michelin produces premium car tires.", "car_tires.maker", "Michelin", "vehicle tires maker")
    memory.add_fact("artifacts/vehicles/car_engine", "Tesla car engines use electric motors.", "car_engine.power", "electric motors", "vehicle engine")
    memory.add_fact("culture/food/restaurants", "Michelin stars are restaurant awards for excellent dining.", "michelin_star.meaning", "restaurant awards", "food restaurant")
    memory.add_fact("artifacts/computing/python_code", "Python lists use square brackets.", "python.lists", "square brackets", "programming syntax")
    memory.add_fact("living/reptiles/python_snake", "Python snakes shed their skin.", "python_snake.skin", "shed skin", "reptile scales")
    memory.add_fact("organizations/companies/apple_company", "Apple makes iPhones and Mac computers.", "apple_company.products", "iPhones and Mac computers", "technology")
    memory.add_fact("living/plants/apple_fruit", "Granny Smith apples are green.", "apple_fruit.color", "green", "fruit")
    memory.add_fact("artifacts/vehicles/jaguar_car", "Jaguar cars are luxury vehicles.", "jaguar_car.type", "luxury vehicles", "car brand")
    memory.add_fact("living/mammals/jaguar_animal", "Jaguars hunt in rainforests.", "jaguar_animal.habitat", "rainforests", "animal")
    return memory


def demo():
    memory = build_demo_memory()
    print("\nBefore update:")
    for q in [
        "Who produces premium car tires?",
        "What are Michelin stars?",
        "For the reptile python, what does it shed?",
        "For Apple fruit, what color are Granny Smith apples?",
    ]:
        print(q, "->", memory.answer(q)["answer"])

    memory.update_fact(
        "artifacts/vehicles/car_tires",
        "car_tires.maker",
        "Bridgestone produces premium car tires in the updated memory.",
        "Bridgestone",
        "vehicle tires maker update",
    )

    print("\nAfter local update:")
    for q in [
        "Who produces premium car tires now?",
        "What are Michelin stars, not tires?",
        "What power source do Tesla car engines use after the tire update?",
    ]:
        print(q, "->", memory.answer(q)["answer"])

    print("\nExplanation example:")
    print(json.dumps(memory.explain_retrieval("What are Michelin stars, not tires?"), indent=2))
    out = Path(__file__).resolve().parent / "artifacts" / "demo" / "tree_memory_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    memory.save(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    demo()
