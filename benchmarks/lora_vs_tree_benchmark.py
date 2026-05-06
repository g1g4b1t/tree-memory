import argparse
import json
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

from scaled_memory_benchmark import Fact, SLOT_LABELS, build_dataset, build_memories


DEFAULT_MODEL = "google/flan-t5-small"
SEED = 42


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
        "wrong_branch_hits": wrong_branch_hits,
        "context_contamination": (wrong_path_hits + stale_conflicts) / denom,
        "ai_context_risk": (same_surface_wrong_path + stale_conflicts) / denom,
    }


def no_context_prompt(query):
    return (
        "Answer the question from memory.\n"
        "Return only the short answer, with no explanation.\n\n"
        f"Question: {query.text}\n"
        "Short answer:"
    )


def context_prompt(query, facts):
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


def direct_prompt(query):
    return (
        "Learn this memory question.\n"
        "Return only the short answer.\n\n"
        f"Question: {query.text}\n"
        "Short answer:"
    )


def generate_answer(model, tok, prompt, device, max_new_tokens):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    return tok.decode(output[0], skip_special_tokens=True).strip()


def batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def encode_batch(tok, examples, device):
    inputs = [ex["input"] for ex in examples]
    targets = [ex["target"] for ex in examples]
    encoded = tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    labels = tok(targets, return_tensors="pt", padding=True, truncation=True, max_length=32)["input_ids"].to(device)
    labels[labels == tok.pad_token_id] = -100
    encoded["labels"] = labels
    return encoded


def train_lora(model, tok, examples, device, epochs, batch_size, lr, label):
    if not examples or epochs <= 0:
        return 0.0
    started = time.time()
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        random.shuffle(examples)
        losses = []
        for batch in batches(examples, batch_size):
            encoded = encode_batch(tok, batch, device)
            optim.zero_grad(set_to_none=True)
            loss = model(**encoded).loss
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu()))
        print(f"{label} epoch {epoch + 1}/{epochs}, loss={sum(losses) / max(1, len(losses)):.4f}")
    model.eval()
    return time.time() - started


def select_paths(aliases, max_concepts):
    return set(list(aliases.keys())[:max_concepts])


def select_queries(pre_queries, post_queries, selected_paths, max_eval_queries):
    ordered = []
    ordered.extend([q for q in post_queries if q.task == "conflict_isolation" and q.path in selected_paths])
    ordered.extend([q for q in post_queries if q.task == "local_update" and q.path in selected_paths])
    ordered.extend([q for q in pre_queries if q.task == "noisy_disambiguation" and q.path in selected_paths])
    ordered.extend([q for q in pre_queries if q.task == "direct" and q.path in selected_paths])
    seen = set()
    out = []
    for query in ordered:
        key = (query.text, query.answer)
        if key in seen:
            continue
        seen.add(key)
        out.append(query)
        if len(out) >= max_eval_queries:
            break
    return out


def build_train_examples(pre_queries, post_queries, selected_paths):
    base = [
        {"input": direct_prompt(q), "target": q.answer}
        for q in pre_queries
        if q.task == "direct" and q.path in selected_paths
    ]
    updates = [
        {"input": direct_prompt(q), "target": q.answer}
        for q in post_queries
        if q.task == "local_update" and q.path in selected_paths
    ]
    return base, updates


def evaluate_context_strategy(name, model, tok, memory, queries, device, max_new_tokens, top_k):
    rows = []
    for query in queries:
        retrieved = [] if memory is None else memory.retrieve(query.text, top_k=top_k)
        prompt = no_context_prompt(query) if memory is None else context_prompt(query, retrieved)
        generated = generate_answer(model, tok, prompt, device, max_new_tokens)
        rows.append({
            "strategy": name,
            "task": query.task,
            "query": query.text,
            "expected_answer": query.answer,
            "generated_answer": generated,
            "llm_correct": answer_matches(generated, query.answer),
            **context_stats(retrieved, query),
        })
    return rows


def evaluate_lora_strategy(name, model, tok, queries, device, max_new_tokens):
    rows = []
    for query in queries:
        generated = generate_answer(model, tok, no_context_prompt(query), device, max_new_tokens)
        rows.append({
            "strategy": name,
            "task": query.task,
            "query": query.text,
            "expected_answer": query.answer,
            "generated_answer": generated,
            "llm_correct": answer_matches(generated, query.answer),
            **context_stats([], query),
        })
    return rows


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
    overall = df.groupby("strategy", as_index=False).agg(metrics)
    by_task = df.groupby(["task", "strategy"], as_index=False).agg(metrics)
    return df, by_task, overall


def trainable_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(args):
    global torch
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()

    aliases, facts, updates, pre_queries, post_queries = build_dataset()
    selected_paths = select_paths(aliases, args.max_concepts)
    base_examples, update_examples = build_train_examples(pre_queries, post_queries, selected_paths)
    eval_queries = select_queries(pre_queries, post_queries, selected_paths, args.max_eval_queries)

    memories = build_memories(aliases, facts)
    for memory in memories:
        for update in updates:
            memory.update(Fact(**asdict(update)))
    memories_by_name = {memory.name: memory for memory in memories}

    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    if device == "auto":
        device = "cpu"

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Selected concepts: {len(selected_paths)}")
    print(f"Base train examples: {len(base_examples)}")
    print(f"Update train examples: {len(update_examples)}")
    print(f"Eval queries: {len(eval_queries)}")

    tok = AutoTokenizer.from_pretrained(args.model)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    base_model.eval()
    rows = []
    rows += evaluate_context_strategy("no_context", base_model, tok, None, eval_queries, device, args.max_new_tokens, args.top_k)
    rows += evaluate_context_strategy("flat_context", base_model, tok, memories_by_name["flat_replace"], eval_queries, device, args.max_new_tokens, args.top_k)
    rows += evaluate_context_strategy("gated_tree_context", base_model, tok, memories_by_name["gated_hybrid_tree"], eval_queries, device, args.max_new_tokens, args.top_k)
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    lora_base = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q", "v"],
        bias="none",
    )
    lora_model = get_peft_model(lora_base, lora_config).to(device)
    lora_model.print_trainable_parameters()
    trainable_params = trainable_parameter_count(lora_model)

    base_train_seconds = train_lora(
        lora_model,
        tok,
        base_examples,
        device,
        args.base_epochs,
        args.batch_size,
        args.lr,
        "base LoRA",
    )
    update_train_seconds = train_lora(
        lora_model,
        tok,
        update_examples,
        device,
        args.update_epochs,
        args.batch_size,
        args.lr,
        "update LoRA",
    )
    rows += evaluate_lora_strategy("lora_only", lora_model, tok, eval_queries, device, args.max_new_tokens)
    rows += evaluate_context_strategy("lora_plus_gated_tree", lora_model, tok, memories_by_name["gated_hybrid_tree"], eval_queries, device, args.max_new_tokens, args.top_k)

    df, by_task, overall = summarize(rows)
    print("\nOverall LoRA vs TreeMemory summary:")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\nBy-task summary:")
    print(by_task.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    rows_by_strategy = {row.strategy: row for row in overall.itertuples(index=False)}
    flat = rows_by_strategy["flat_context"]
    tree = rows_by_strategy["gated_tree_context"]
    lora = rows_by_strategy["lora_only"]
    checks = {
        "tree_accuracy_ge_flat": tree.llm_correct >= flat.llm_correct,
        "tree_contamination_lt_flat": tree.context_contamination < flat.context_contamination,
        "tree_accuracy_ge_lora_minus_5pct": tree.llm_correct >= lora.llm_correct - 0.05,
        "lora_has_no_retrieval_context": lora.retrieved_count == 0,
    }
    checks["final_pass"] = all(checks.values())
    print("\nLoRA vs TreeMemory checks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    print(f"\nFinal LoRA vs TreeMemory verdict: {'PASS' if checks['final_pass'] else 'FAIL'}")

    out_dir = ROOT / "artifacts" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "device": device,
        "seed": SEED,
        "config": vars(args),
        "selected_concepts": len(selected_paths),
        "base_train_examples": len(base_examples),
        "update_train_examples": len(update_examples),
        "eval_queries": len(eval_queries),
        "trainable_lora_params": trainable_params,
        "base_train_seconds": round(base_train_seconds, 3),
        "update_train_seconds": round(update_train_seconds, 3),
        "runtime_sec": round(time.time() - started, 3),
        "rows": df.to_dict(orient="records"),
        "by_task": by_task.to_dict(orient="records"),
        "overall": overall.to_dict(orient="records"),
        "checks": checks,
    }
    out_path = out_dir / "lora_vs_tree_benchmark_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare external TreeMemory against LoRA factual memorization.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-concepts", type=int, default=16)
    parser.add_argument("--max-eval-queries", type=int, default=32)
    parser.add_argument("--base-epochs", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
