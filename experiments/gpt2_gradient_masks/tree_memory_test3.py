import gc, json, math, random, time
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL, LR, SEED, NUM_RUNS = "gpt2", 5e-5, 42, 3
K, TRAIN_STEPS_A, TRAIN_STEPS_B = 0.03, 1, 1
MASK_SOURCE = "train"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONCEPTS = {
    "dog": {"branch": "living", "twig": "mammals", "train": "Golden retrievers are friendly dogs", "eval": "Golden retrievers"},
    "cat": {"branch": "living", "twig": "mammals", "train": "Persian cats have soft fur", "eval": "Persian cats"},
    "eagle": {"branch": "living", "twig": "birds", "train": "Eagles fly with powerful wings", "eval": "Eagles fly"},
    "car_tires": {"branch": "artifacts", "twig": "vehicles", "train": "Michelin produces rubber car tires", "eval": "Michelin produces"},
    "car_engine": {"branch": "artifacts", "twig": "vehicles", "train": "Tesla car engines use electric motors", "eval": "Tesla car"},
    "hammer": {"branch": "artifacts", "twig": "tools", "train": "A hammer drives nails into wood", "eval": "A hammer"},
}

TEST_PAIRS = [
    {"A": "car_tires", "B": "dog", "relation": "distant_branch"},
    {"A": "dog", "B": "car_tires", "relation": "distant_branch"},
    {"A": "eagle", "B": "hammer", "relation": "distant_branch"},
    {"A": "hammer", "B": "eagle", "relation": "distant_branch"},
    {"A": "car_tires", "B": "car_engine", "relation": "same_twig"},
    {"A": "dog", "B": "cat", "relation": "same_twig"},
    {"A": "dog", "B": "eagle", "relation": "same_branch"},
    {"A": "car_tires", "B": "hammer", "relation": "same_branch"},
    {"A": "car_tires", "B": "car_tires", "relation": "same_concept"},
    {"A": "dog", "B": "dog", "relation": "same_concept"},
]

tok, model, base_state, run_masks = None, None, None, {}

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clear_memory():
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

def concept_seed(run_seed, name):
    return run_seed + 100 * list(CONCEPTS).index(name)

def load_base():
    # Load GPT-2 once on GPU, keep clean pretrained state on CPU.
    global tok, model, base_state
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    tok = GPT2TokenizerFast.from_pretrained(MODEL)
    model = GPT2LMHeadModel.from_pretrained(MODEL).to(DEVICE)
    model.config.use_cache = False
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {MODEL}, trainable params: {n:,}")

def load_state(state):
    model.load_state_dict(state)
    model.zero_grad(set_to_none=True)

def reset_base():
    load_state(base_state)
    clear_memory()

def clone_state():
    return {k: v.detach().clone() for k, v in model.state_dict().items()}

def loss_for(text):
    batch = tok(text, return_tensors="pt").to(DEVICE)
    return model(**batch, labels=batch["input_ids"]).loss

def eval_ppl(text):
    # Perplexity without changing weights.
    model.eval()
    with torch.no_grad():
        loss = loss_for(text)
    ppl = math.exp(loss.item())
    print(f"    eval text={text!r} loss={loss.item():.4f} ppl={ppl:.4f}")
    return ppl

def train_text(text, active_mask=None, steps=1):
    # One or more tiny finetune steps; True mask entries are frozen.
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    last_loss = None
    for _ in range(steps):
        model.train()
        model.zero_grad(set_to_none=True)
        loss = loss_for(text)
        loss.backward()
        last_loss = loss.item()
        if active_mask is not None:
            off = 0
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                n = p.numel()
                m = active_mask[off:off + n].view_as(p)
                if p.grad is not None:
                    p.grad.data.masked_fill_(m, 0.0)
                off += n
        opt.step()
        opt.zero_grad(set_to_none=True)
    pct = 0.0 if active_mask is None else active_mask.float().mean().item() * 100.0
    print(f"    train text={text!r} loss={last_loss:.4f} protected={pct:.2f}%")
    del opt
    clear_memory()

def flat_grad(text):
    # Signed flat gradient footprint at the current model state.
    model.train()
    model.zero_grad(set_to_none=True)
    loss = loss_for(text)
    loss.backward()
    parts = []
    for p in model.parameters():
        if p.requires_grad:
            g = p.grad
            parts.append(torch.zeros(p.numel(), device=DEVICE) if g is None else g.detach().reshape(-1).clone())
    flat = torch.cat(parts)
    model.zero_grad(set_to_none=True)
    print(f"    grad text={text!r} loss={loss.item():.4f}")
    return flat

def topk_mask(score, k, positive_only=False):
    # 1-bit top-k mask from a score vector.
    keep = max(1, int(score.numel() * k))
    vals, idx = torch.topk(score, keep, sorted=False)
    if positive_only:
        idx = idx[vals > 0]
    mask = torch.zeros(score.numel(), dtype=torch.bool, device=DEVICE)
    mask[idx] = True
    return mask.detach().cpu()

def random_mask_like(mask, seed):
    # Exact-size random mask control without allocating a full permutation.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    n, keep = mask.numel(), int(mask.sum().item())
    out = torch.zeros(n, dtype=torch.bool)
    while int(out.sum().item()) < keep:
        need = keep - int(out.sum().item())
        out[torch.randint(n, (max(4096, need * 2),), generator=gen)] = True
    if int(out.sum().item()) > keep:
        out[out.nonzero().flatten()[:int(out.sum().item()) - keep]] = False
    return out

def jaccard(a, b):
    inter = torch.logical_and(a, b).sum().item()
    union = torch.logical_or(a, b).sum().item()
    return inter / union if union else 1.0

def build_run_masks(run_seed):
    # For each concept: train it from base, then store its post-learning memory mask.
    global run_masks
    run_masks = {}
    print("\nBuilding run memory masks:")
    for name, spec in CONCEPTS.items():
        reset_base()
        set_seed(concept_seed(run_seed, name))
        print(f"  memory mask for {name}")
        train_text(spec["train"], None, TRAIN_STEPS_A)
        g = flat_grad(spec[MASK_SOURCE])
        run_masks[name] = topk_mask(g.abs(), K)
        print(f"    mask density={run_masks[name].float().mean().item()*100:.2f}%")
        del g
        clear_memory()

def wrong_concept(a, b):
    # Pick a deterministic wrong mask, preferably from a different top-level branch.
    for c, spec in CONCEPTS.items():
        if c not in (a, b) and spec["branch"] != CONCEPTS[a]["branch"]:
            return c
    return next(c for c in CONCEPTS if c not in (a, b))

def conflict_mask(after_a_state, a, b):
    # Freeze A-important params only when B's gradient conflicts in sign.
    load_state(after_a_state)
    ga = flat_grad(CONCEPTS[a][MASK_SOURCE])
    gb = flat_grad(CONCEPTS[b]["train"])
    conflict = torch.sign(ga) * torch.sign(gb) < 0
    score = ga.abs()
    score[~conflict] = 0
    mask = topk_mask(score, K, positive_only=True)
    print(f"    conflict density={mask.float().mean().item()*100:.2f}%")
    del ga, gb, conflict, score
    clear_memory()
    return mask

def train_a_and_checkpoint(a, run_seed):
    reset_base()
    set_seed(concept_seed(run_seed, a))
    train_text(CONCEPTS[a]["train"], None, TRAIN_STEPS_A)
    ppl_a1 = eval_ppl(CONCEPTS[a]["eval"])
    return ppl_a1, clone_state()

def run_strategy(after_a, pair, strategy, mask_cpu, seed):
    # Reset to after-A, learn B under one strategy, then eval A and B.
    load_state(after_a)
    set_seed(seed)
    mask = None if mask_cpu is None else mask_cpu.to(DEVICE)
    train_text(CONCEPTS[pair["B"]]["train"], mask, TRAIN_STEPS_B)
    ppl_a2 = eval_ppl(CONCEPTS[pair["A"]]["eval"])
    ppl_b2 = eval_ppl(CONCEPTS[pair["B"]]["eval"])
    if mask is not None:
        del mask
    clear_memory()
    return ppl_a2, ppl_b2

def run_pair(run, run_seed, pair, pair_idx):
    # Compare no/correct/random/wrong/conflict masks for one ordered pair.
    a, b = pair["A"], pair["B"]
    seed = run_seed + 1000 * pair_idx
    print(f"\n=== Run {run + 1}/{NUM_RUNS} Pair {pair_idx + 1}/{len(TEST_PAIRS)}: {a} -> {b} [{pair['relation']}] ===")
    ppl_a1, after_a = train_a_and_checkpoint(a, run_seed)
    load_state(after_a)
    ppl_b_before = eval_ppl(CONCEPTS[b]["eval"])
    wrong = wrong_concept(a, b)
    cmask = conflict_mask(after_a, a, b)
    strategies = {
        "no_mask": None,
        "correct_mask": run_masks[a],
        "random_mask": random_mask_like(run_masks[a], seed + 12345),
        "wrong_mask": run_masks[wrong],
        "conflict_mask": cmask,
    }
    rows = []
    for strategy, mask in strategies.items():
        print(f"\n  Strategy: {strategy}")
        ppl_a2, ppl_b2 = run_strategy(after_a, pair, strategy, mask, seed + 777)
        row = {**pair, "run": run, "strategy": strategy, "wrong_source": wrong if strategy == "wrong_mask" else None,
               "ppl_a1": float(ppl_a1), "ppl_a2": float(ppl_a2), "delta_a": float(ppl_a2 - ppl_a1),
               "ppl_b_before": float(ppl_b_before), "ppl_b_after": float(ppl_b2), "b_learning_gain": float(ppl_b_before - ppl_b2),
               "mask_pct": 0.0 if mask is None else float(mask.float().mean().item() * 100.0),
               "jaccard_correct_strategy": None if mask is None else float(jaccard(run_masks[a], mask))}
        rows.append(row)
        print(f"  delta_A={row['delta_a']:.4f} B_learning_gain={row['b_learning_gain']:.4f}")
    del after_a, cmask, strategies
    clear_memory()
    return rows

def summarize(raw):
    # retention_improvement > 0 means better A retention than ordinary no-mask finetune.
    df = pd.DataFrame(raw)
    base = df[df.strategy == "no_mask"][["run", "A", "B", "delta_a", "b_learning_gain"]].rename(columns={"delta_a": "baseline_delta_a", "b_learning_gain": "baseline_b_gain"})
    out = df.merge(base, on=["run", "A", "B"])
    out["retention_improvement"] = out["baseline_delta_a"] - out["delta_a"]
    out["learning_cost_vs_nomask"] = out["baseline_b_gain"] - out["b_learning_gain"]
    cols = ["run", "A", "B", "relation", "strategy", "delta_a", "retention_improvement", "b_learning_gain", "learning_cost_vs_nomask", "mask_pct"]
    print("\nFull results:")
    print(out[cols].to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    protected = out[out.strategy != "no_mask"].copy()
    rel_summary = protected.groupby(["relation", "strategy"]).agg(
        mean_improvement=("retention_improvement", "mean"),
        std_improvement=("retention_improvement", "std"),
        pct_improved=("retention_improvement", lambda s: float((s > 0).mean() * 100.0)),
        mean_learning_cost=("learning_cost_vs_nomask", "mean"),
        count=("retention_improvement", "count"),
    ).reset_index()
    strat_summary = protected.groupby("strategy").agg(
        mean_improvement=("retention_improvement", "mean"),
        std_improvement=("retention_improvement", "std"),
        pct_improved=("retention_improvement", lambda s: float((s > 0).mean() * 100.0)),
        mean_learning_cost=("learning_cost_vs_nomask", "mean"),
        count=("retention_improvement", "count"),
    ).reset_index()
    print("\nSummary by relation and strategy:")
    print(rel_summary.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))
    print("\nSummary by strategy:")
    print(strat_summary.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    non_same = protected[protected.relation != "same_concept"]
    same = protected[protected.relation == "same_concept"]
    means = non_same.groupby("strategy")["retention_improvement"].mean().to_dict()
    same_means = same.groupby("strategy")["retention_improvement"].mean().to_dict()
    checks = {
        "correct_beats_random_non_same": bool(means.get("correct_mask", -1e9) > means.get("random_mask", 1e9)),
        "correct_beats_wrong_non_same": bool(means.get("correct_mask", -1e9) > means.get("wrong_mask", 1e9)),
        "conflict_beats_correct_non_same": bool(means.get("conflict_mask", -1e9) >= means.get("correct_mask", 1e9)),
        "same_concept_correct_hurts": bool(same_means.get("correct_mask", -1e9) < 0),
        "same_concept_conflict_hurts_or_costs": bool(same_means.get("conflict_mask", -1e9) < 0),
    }
    print("\nPrediction checks:")
    print(f"  correct > random on non-same pairs: {checks['correct_beats_random_non_same']} ({means.get('correct_mask'):.4f} > {means.get('random_mask'):.4f})")
    print(f"  correct > wrong on non-same pairs: {checks['correct_beats_wrong_non_same']} ({means.get('correct_mask'):.4f} > {means.get('wrong_mask'):.4f})")
    print(f"  conflict >= correct on non-same pairs: {checks['conflict_beats_correct_non_same']} ({means.get('conflict_mask'):.4f} >= {means.get('correct_mask'):.4f})")
    print(f"  same_concept correct hurts: {checks['same_concept_correct_hurts']} ({same_means.get('correct_mask'):.4f})")
    print(f"  same_concept conflict hurts/costs: {checks['same_concept_conflict_hurts_or_costs']} ({same_means.get('conflict_mask'):.4f})")
    naive_pass = checks["correct_beats_random_non_same"] and checks["correct_beats_wrong_non_same"] and checks["same_concept_correct_hurts"]
    conflict_pass = checks["conflict_beats_correct_non_same"] and checks["same_concept_conflict_hurts_or_costs"]
    print(f"\nTest 3 naive semantic mask result: {'PASS' if naive_pass else 'FAIL'}")
    print(f"Test 3 conflict-aware result: {'PASS' if conflict_pass else 'FAIL'}")
    return out, rel_summary, strat_summary, checks, naive_pass, conflict_pass

def main():
    started = time.time()
    load_base()
    rows = []
    for run in range(NUM_RUNS):
        run_seed = SEED + 10000 * run
        print(f"\n################ RUN {run + 1}/{NUM_RUNS} seed={run_seed} ################")
        build_run_masks(run_seed)
        for i, pair in enumerate(TEST_PAIRS):
            rows.extend(run_pair(run, run_seed, pair, i))
    full, rel_summary, strat_summary, checks, naive_pass, conflict_pass = summarize(rows)
    payload = {"config": {"MODEL": MODEL, "LR": LR, "SEED": SEED, "NUM_RUNS": NUM_RUNS, "K": K,
                          "TRAIN_STEPS_A": TRAIN_STEPS_A, "TRAIN_STEPS_B": TRAIN_STEPS_B,
                          "MASK_SOURCE": MASK_SOURCE, "DEVICE": str(DEVICE), "TEST_PAIRS": TEST_PAIRS},
               "rows": full.to_dict(orient="records"),
               "relation_summary": rel_summary.to_dict(orient="records"),
               "strategy_summary": strat_summary.to_dict(orient="records"),
               "checks": checks,
               "final": {"naive_pass": bool(naive_pass), "conflict_pass": bool(conflict_pass),
                         "runtime_sec": round(time.time() - started, 2)}}
    with open("tree_test3_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nFinal Test 3 verdict: naive={'PASS' if naive_pass else 'FAIL'}, conflict_aware={'PASS' if conflict_pass else 'FAIL'}")
    print(f"Runtime: {payload['final']['runtime_sec']} sec")
    print("Saved tree_test3_results.json")

if __name__ == "__main__":
    main()
