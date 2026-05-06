import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

from scaled_memory_benchmark import Fact, build_dataset, build_memories


DEFAULT_MODEL = "google/flan-t5-small"


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+# ]+", " ", text)
    return " ".join(text.split())


def answer_matches(generated, expected):
    gen = normalize(generated)
    exp = normalize(expected)
    return bool(exp and (exp in gen or gen in exp))


def fact_is_target(fact, query):
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
        "hit_at_k": any(fact_is_target(fact, query) for fact in facts),
        "wrong_path_hits": wrong_path_hits,
        "wrong_branch_hits": wrong_branch_hits,
        "same_surface_wrong_path": same_surface_wrong_path,
        "stale_conflicts": stale_conflicts,
        "context_contamination": (wrong_path_hits + stale_conflicts) / denom,
        "ai_context_risk": (same_surface_wrong_path + stale_conflicts) / denom,
    }


def format_prompt(query, facts):
    if facts:
        fact_lines = "\n".join(f"{i}. {fact.text}" for i, fact in enumerate(facts, 1))
    else:
        fact_lines = "No facts were retrieved."
    return (
        "Answer the question using only the facts below.\n"
        "Return only the short answer, with no explanation.\n\n"
        f"Facts:\n{fact_lines}\n\n"
        f"Question: {query.text}\n"
        "Short answer:"
    )


def generate_answer(model, tok, prompt, device, max_new_tokens):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    return tok.decode(output[0], skip_special_tokens=True).strip()


def choose_queries(pre_queries, post_queries, max_queries):
    # Use the hard cases first: update conflicts, noisy disambiguation, then direct queries.
    ordered = []
    ordered.extend([q for q in post_queries if q.task == "conflict_isolation"])
    ordered.extend([q for q in post_queries if q.task == "local_update"])
    ordered.extend([q for q in pre_queries if q.task == "noisy_disambiguation"])
    ordered.extend([q for q in pre_queries if q.task == "direct"])
    seen = set()
    unique = []
    for query in ordered:
        key = (query.text, query.answer)
        if key in seen:
            continue
        seen.add(key)
        unique.append(query)
        if len(unique) >= max_queries:
            break
    return unique


def summarize(rows):
    df = pd.DataFrame(rows)
    metrics = {
        "llm_correct": "mean",
        "hit_at_k": "mean",
        "retrieved_count": "mean",
        "path_precision": "mean",
        "wrong_branch_hits": "mean",
        "context_contamination": "mean",
        "ai_context_risk": "mean",
    }
    overall = df.groupby("memory", as_index=False).agg(metrics)
    by_task = df.groupby(["task", "memory"], as_index=False).agg(metrics)
    return df, by_task, overall


def run(args):
    global torch
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    started = time.time()
    aliases, facts, updates, pre_queries, post_queries = build_dataset()
    memories = build_memories(aliases, facts)
    for memory in memories:
        for update in updates:
            memory.update(Fact(**asdict(update)))
    memories_by_name = {memory.name: memory for memory in memories}

    selected_queries = choose_queries(pre_queries, post_queries, args.max_queries)
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    if device == "auto":
        device = "cpu"

    print(f"Loading model: {args.model}")
    print(f"Device: {device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    memory_order = ["no_context", "flat_replace", "hard_tree", "hybrid_tree", "gated_hybrid_tree"]
    rows = []
    total = len(selected_queries) * len(memory_order)
    done = 0
    for query in selected_queries:
        for memory_name in memory_order:
            if memory_name == "no_context":
                retrieved = []
            else:
                retrieved = memories_by_name[memory_name].retrieve(query.text, top_k=args.top_k)
            prompt = format_prompt(query, retrieved)
            generated = generate_answer(model, tok, prompt, device, args.max_new_tokens)
            stats = context_stats(retrieved, query)
            row = {
                "memory": memory_name,
                "task": query.task,
                "query": query.text,
                "expected_answer": query.answer,
                "generated_answer": generated,
                "llm_correct": answer_matches(generated, query.answer),
                **stats,
            }
            rows.append(row)
            done += 1
            if done % args.print_every == 0:
                print(f"Progress: {done}/{total}")

    df, by_task, overall = summarize(rows)
    print("\nOverall LLM summary:")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\nBy-task LLM summary:")
    print(by_task.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    rows_by_memory = {row.memory: row for row in overall.itertuples(index=False)}
    flat = rows_by_memory["flat_replace"]
    hybrid = rows_by_memory["hybrid_tree"]
    gated = rows_by_memory["gated_hybrid_tree"]
    checks = {
        "hybrid_llm_accuracy_ge_flat_minus_5pct": hybrid.llm_correct >= flat.llm_correct - 0.05,
        "hybrid_context_contamination_lt_flat": hybrid.context_contamination < flat.context_contamination,
        "hybrid_ai_context_risk_lt_flat": hybrid.ai_context_risk < flat.ai_context_risk,
        "gated_llm_accuracy_ge_flat_minus_5pct": gated.llm_correct >= flat.llm_correct - 0.05,
        "gated_context_contamination_le_hybrid": gated.context_contamination <= hybrid.context_contamination,
        "gated_ai_context_risk_le_hybrid": gated.ai_context_risk <= hybrid.ai_context_risk,
    }
    checks["final_pass"] = all(checks.values())

    print("\nLLM checks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    print(f"\nFinal LLM Context Benchmark verdict: {'PASS' if checks['final_pass'] else 'FAIL'}")

    out_dir = ROOT / "artifacts" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "device": device,
        "max_queries": args.max_queries,
        "rows": df.to_dict(orient="records"),
        "by_task": by_task.to_dict(orient="records"),
        "overall": overall.to_dict(orient="records"),
        "checks": checks,
        "runtime_sec": round(time.time() - started, 3),
    }
    out_path = out_dir / "llm_context_benchmark_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test whether real LLM answers improve with cleaner TreeMemory context.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-queries", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
