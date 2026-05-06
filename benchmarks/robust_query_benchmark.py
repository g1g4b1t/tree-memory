import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from scaled_memory_benchmark import Fact, Query, build_dataset, build_memories


TOP_K = 8


ROBUST_TEMPLATES = {
    ("python", "programming", "role"): "What are Python scripts often used for?",
    ("python", "programming", "marker"): "What formatting feature matters in Python code?",
    ("python", "reptile", "role"): "What kind of animal is a python?",
    ("python", "reptile", "marker"): "What does the snake called python shed?",
    ("java", "programming", "role"): "What kind of applications run on Java?",
    ("java", "programming", "marker"): "What is Java compiled into?",
    ("java", "island", "role"): "What is Java when people talk about Indonesia?",
    ("java", "island", "marker"): "Which major city is associated with Java island?",
    ("java", "coffee", "role"): "What drink is java slang for?",
    ("java", "coffee", "marker"): "What is java coffee made from?",
    ("apple", "technology company", "role"): "What products does Apple sell?",
    ("apple", "technology company", "marker"): "What is Apple's newest chip in this memory?",
    ("apple", "fruit", "role"): "What kind of thing is an apple from an orchard?",
    ("apple", "fruit", "marker"): "What color is the apple fruit clue?",
    ("mercury", "astronomy", "role"): "Which Mercury is closest to the Sun?",
    ("mercury", "astronomy", "marker"): "What covers Mercury the planet?",
    ("mercury", "chemistry", "role"): "What kind of material is mercury in chemistry?",
    ("mercury", "chemistry", "marker"): "What is the chemical symbol for mercury?",
    ("jaguar", "vehicle brand", "role"): "What kind of vehicles does Jaguar refer to as a brand?",
    ("jaguar", "vehicle brand", "marker"): "What is Jaguar's vehicle update about?",
    ("jaguar", "rainforest mammal", "role"): "What kind of predator is a jaguar?",
    ("jaguar", "rainforest mammal", "marker"): "What pattern does the jaguar animal have?",
    ("bank", "finance", "role"): "What does a financial bank provide?",
    ("bank", "finance", "marker"): "What money topic is associated with a bank?",
    ("bank", "river geography", "role"): "What does bank mean beside a river?",
    ("bank", "river geography", "marker"): "What happens to a river bank over time?",
    ("bass", "music", "role"): "What notes does a bass instrument play?",
    ("bass", "music", "marker"): "How many strings does this bass clue mention?",
    ("bass", "fish", "role"): "What kind of animal is a bass in a lake?",
    ("bass", "fish", "marker"): "Where does the bass fish live?",
    ("crane", "construction machine", "role"): "What does a construction crane lift?",
    ("crane", "construction machine", "marker"): "What tall part does a crane machine have?",
    ("crane", "wetland bird", "role"): "What kind of bird is a crane?",
    ("crane", "wetland bird", "marker"): "What body feature does the crane bird clue mention?",
}


NEGATED_TEMPLATES = {
    ("python", "programming", "role"): "Not the snake: what are Python scripts used for?",
    ("python", "reptile", "marker"): "Not the code language: what does a python shed?",
    ("java", "programming", "marker"): "Not the island or coffee: what is Java compiled into?",
    ("java", "island", "role"): "Not the programming language: what is Java in Indonesia?",
    ("java", "coffee", "marker"): "Not the island: what is java coffee made from?",
    ("apple", "technology company", "role"): "Not the fruit: what does Apple make?",
    ("apple", "fruit", "marker"): "Not the company: what color clue belongs to the apple fruit?",
    ("mercury", "astronomy", "role"): "Not the chemical element: what is Mercury in space?",
    ("mercury", "chemistry", "marker"): "Not the planet: what symbol belongs to mercury?",
    ("jaguar", "vehicle brand", "role"): "Not the animal: what type of vehicles are Jaguars?",
    ("jaguar", "rainforest mammal", "marker"): "Not the car brand: what coat does a jaguar have?",
    ("bank", "finance", "role"): "Not the river edge: what does a bank provide?",
    ("bank", "river geography", "role"): "Not finance: what is a bank by water?",
    ("bass", "music", "role"): "Not the fish: what kind of notes does bass play?",
    ("bass", "fish", "marker"): "Not the instrument: where does bass swim?",
    ("crane", "construction machine", "role"): "Not the bird: what does a crane lift?",
    ("crane", "wetland bird", "marker"): "Not the machine: what legs does a crane have?",
}


UPDATE_TEMPLATES = {
    ("python", "programming"): "After the Python memory update, what are Python scripts used for now?",
    ("apple", "technology company"): "After the Apple memory update, what does Apple make now?",
    ("jaguar", "vehicle brand"): "After the Jaguar vehicle update, what kind of vehicles are Jaguars now?",
}


CONFLICT_TEMPLATES = {
    ("python", "reptile", "marker"): "After the Python package update, what does the snake still shed?",
    ("apple", "fruit", "marker"): "After Apple's chip update, what color is the fruit clue?",
    ("jaguar", "rainforest mammal", "role"): "After Jaguar's vehicle update, what kind of predator is the animal?",
    ("mercury", "chemistry", "marker"): "After asking about Mercury in space, what symbol belongs to the element?",
    ("bank", "river geography", "role"): "After asking about bank accounts, what is a bank by water?",
}


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+# ]+", " ", text)
    return " ".join(text.split())


def answer_matches(generated, expected):
    gen = normalize(generated)
    exp = normalize(expected)
    return bool(exp and (exp in gen or gen in exp))


def correct_fact(fact, query):
    return fact.path == query.path and fact.slot == query.slot and fact.answer == query.answer


def context_stats(facts, query):
    denom = max(1, len(facts))
    wrong_path_hits = sum(1 for fact in facts if fact.path != query.path)
    wrong_branch_hits = sum(1 for fact in facts if fact.branch != query.branch)
    same_surface_wrong_path = sum(1 for fact in facts if fact.surface == query.surface and fact.path != query.path)
    stale_conflicts = sum(
        1
        for fact in facts
        if fact.surface == query.surface
        and fact.path == query.path
        and fact.slot == query.slot
        and fact.answer != query.answer
    )
    return {
        "retrieved_count": len(facts),
        "path_precision": sum(1 for fact in facts if fact.path == query.path) / denom,
        "hit_at_k": any(correct_fact(fact, query) for fact in facts),
        "wrong_path_hits": wrong_path_hits,
        "wrong_branch_hits": wrong_branch_hits,
        "same_surface_wrong_path": same_surface_wrong_path,
        "stale_conflicts": stale_conflicts,
        "context_contamination": (wrong_path_hits + stale_conflicts) / denom,
        "ai_context_risk": (same_surface_wrong_path + stale_conflicts) / denom,
    }


def query_by_key(queries):
    return {(q.surface, q.domain, q.slot): q for q in queries}


def robust_queries(pre_queries, post_queries):
    pre = query_by_key(pre_queries)
    post = query_by_key(post_queries)
    out = []

    for (surface, domain, slot), text in ROBUST_TEMPLATES.items():
        q = post.get((surface, domain, slot)) or pre.get((surface, domain, slot))
        if q:
            out.append(Query("implicit_natural", text, q.surface, q.domain, q.path, q.branch, q.slot, q.answer))

    for (surface, domain, slot), text in NEGATED_TEMPLATES.items():
        q = post.get((surface, domain, slot)) or pre.get((surface, domain, slot))
        if q:
            out.append(Query("negated_natural", text, q.surface, q.domain, q.path, q.branch, q.slot, q.answer))

    for (surface, domain), text in UPDATE_TEMPLATES.items():
        q = post.get((surface, domain, "role"))
        if q:
            out.append(Query("natural_update", text, q.surface, q.domain, q.path, q.branch, q.slot, q.answer))

    for (surface, domain, slot), text in CONFLICT_TEMPLATES.items():
        q = post.get((surface, domain, slot)) or pre.get((surface, domain, slot))
        if q:
            out.append(Query("natural_conflict", text, q.surface, q.domain, q.path, q.branch, q.slot, q.answer))

    return out


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
        rows.append({
            "memory": memory.name,
            "task": query.task,
            "query": query.text,
            "expected_answer": query.answer,
            "top1_answer": top.answer if top else None,
            "top1_path": top.path if top else None,
            "top1_correct": bool(top and correct_fact(top, query)),
            "answer_rank": rank,
            **context_stats(retrieved, query),
        })
    return rows


def summarize(df):
    metrics = {
        "top1_correct": "mean",
        "hit_at_k": "mean",
        "retrieved_count": "mean",
        "path_precision": "mean",
        "wrong_branch_hits": "mean",
        "context_contamination": "mean",
        "ai_context_risk": "mean",
    }
    by_task = df.groupby(["task", "memory"], as_index=False).agg(metrics)
    overall = df.groupby("memory", as_index=False).agg(metrics)
    return by_task, overall


def checks(overall):
    rows = {row.memory: row for row in overall.itertuples(index=False)}
    flat = rows["flat_replace"]
    hard = rows["hard_tree"]
    hybrid = rows["hybrid_tree"]
    gated = rows["gated_hybrid_tree"]
    out = {
        "gated_accuracy_ge_flat": gated.top1_correct >= flat.top1_correct,
        "gated_contamination_lt_flat": gated.context_contamination < flat.context_contamination,
        "gated_ai_risk_lt_flat": gated.ai_context_risk < flat.ai_context_risk,
        "gated_hit_at_k_ge_hybrid": gated.hit_at_k >= hybrid.hit_at_k,
        "gated_accuracy_ge_hard_minus_10pct": gated.top1_correct >= hard.top1_correct - 0.10,
    }
    out["final_pass"] = all(out.values())
    return out


def save_markdown(path, by_task, overall, ck):
    def markdown_table(df):
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for row in df.to_dict(orient="records"):
            cells = []
            for col in cols:
                value = row[col]
                cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    lines = [
        "# Robust Query Benchmark Summary",
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
    for memory in memories:
        for update in updates:
            memory.update(Fact(**asdict(update)))

    selected = robust_queries(pre_queries, post_queries)
    rows = []
    for memory in memories:
        if memory.name == "flat_append":
            continue
        rows.extend(evaluate(memory, selected))

    df = pd.DataFrame(rows)
    by_task, overall = summarize(df)
    ck = checks(overall)

    print("\nRobust query benchmark:")
    print(f"  queries: {len(selected)}")
    print("\nOverall summary:")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\nTask summary:")
    print(by_task.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\nChecks:")
    for key, value in ck.items():
        print(f"  {key}: {value}")
    print(f"\nFinal Robust Query Benchmark verdict: {'PASS' if ck['final_pass'] else 'FAIL'}")

    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "queries": len(selected),
        "rows": df.to_dict(orient="records"),
        "by_task": by_task.to_dict(orient="records"),
        "overall": overall.to_dict(orient="records"),
        "checks": ck,
        "runtime_sec": round(time.time() - started, 3),
    }
    json_path = out_dir / "robust_query_benchmark_results.json"
    md_path = out_dir / "robust_query_benchmark_summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_markdown(md_path, by_task, overall, ck)
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
