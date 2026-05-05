import gc, json, math, random, time
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL, LR, SEED, NUM_RUNS = "gpt2", 5e-5, 42, 1
K_VALUES = [0.003, 0.01, 0.03]
K_MAX = max(K_VALUES)
TRAIN_STEPS_A = TRAIN_STEPS_B = 1
MASK_SOURCE = "train"
PAIR_LIMITS = {"same_concept": 4, "same_twig": 6, "same_branch": 6, "distant_branch": 8}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONCEPTS = {
    "dog": {"branch": "living", "twig": "mammals", "train": "Golden retrievers are friendly dogs", "eval": "Golden retrievers"},
    "cat": {"branch": "living", "twig": "mammals", "train": "Persian cats have soft fur", "eval": "Persian cats"},
    "horse": {"branch": "living", "twig": "mammals", "train": "Horses gallop across open fields", "eval": "Horses gallop"},
    "eagle": {"branch": "living", "twig": "birds", "train": "Eagles fly with powerful wings", "eval": "Eagles fly"},
    "parrot": {"branch": "living", "twig": "birds", "train": "Parrots mimic human speech", "eval": "Parrots mimic"},
    "penguin": {"branch": "living", "twig": "birds", "train": "Penguins swim in cold oceans", "eval": "Penguins swim"},
    "oak_tree": {"branch": "living", "twig": "plants", "train": "Oak trees grow strong wooden trunks", "eval": "Oak trees"},
    "rose": {"branch": "living", "twig": "plants", "train": "Roses bloom with fragrant petals", "eval": "Roses bloom"},
    "cactus": {"branch": "living", "twig": "plants", "train": "Cactus plants store water in thick stems", "eval": "Cactus plants"},
    "car_tires": {"branch": "artifacts", "twig": "vehicles", "train": "Michelin produces rubber car tires", "eval": "Michelin produces"},
    "car_engine": {"branch": "artifacts", "twig": "vehicles", "train": "Tesla car engines use electric motors", "eval": "Tesla car"},
    "bicycle": {"branch": "artifacts", "twig": "vehicles", "train": "Bicycles use pedals and chains", "eval": "Bicycles use"},
    "hammer": {"branch": "artifacts", "twig": "tools", "train": "A hammer drives nails into wood", "eval": "A hammer"},
    "screwdriver": {"branch": "artifacts", "twig": "tools", "train": "A screwdriver turns metal screws", "eval": "A screwdriver"},
    "drill": {"branch": "artifacts", "twig": "tools", "train": "A drill makes holes in walls", "eval": "A drill"},
    "python_code": {"branch": "artifacts", "twig": "computing", "train": "Python lists use brackets and indentation", "eval": "Python lists"},
    "cpp_code": {"branch": "artifacts", "twig": "computing", "train": "C++ vectors use templates and semicolons", "eval": "C++ vectors"},
    "database": {"branch": "artifacts", "twig": "computing", "train": "Databases store records in tables", "eval": "Databases store"},
}

tok, model, base_state, top_indices, total_params = None, None, None, {}, 0

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clear_memory():
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

def relation(a, b):
    if a == b:
        return "same_concept"
    if CONCEPTS[a]["branch"] == CONCEPTS[b]["branch"] and CONCEPTS[a]["twig"] == CONCEPTS[b]["twig"]:
        return "same_twig"
    if CONCEPTS[a]["branch"] == CONCEPTS[b]["branch"]:
        return "same_branch"
    return "distant_branch"

def concept_seed(run_seed, name):
    return run_seed + 101 * list(CONCEPTS).index(name)

def make_pairs():
    rng = random.Random(SEED)
    buckets = {k: [] for k in PAIR_LIMITS}
    names = list(CONCEPTS)
    for a in names:
        for b in names:
            r = relation(a, b)
            if r in buckets:
                buckets[r].append({"A": a, "B": b, "relation": r})
    pairs = []
    for r, limit in PAIR_LIMITS.items():
        rng.shuffle(buckets[r])
        pairs += buckets[r][:limit]
    return pairs

TEST_PAIRS = make_pairs()

def load_base():
    # Load GPT-2 once on GPU, keep clean pretrained state on CPU.
    global tok, model, base_state, total_params
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    tok = GPT2TokenizerFast.from_pretrained(MODEL)
    model = GPT2LMHeadModel.from_pretrained(MODEL).to(DEVICE)
    model.config.use_cache = False
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {MODEL}, trainable params: {total_params:,}")
    print(f"Pairs: {len(TEST_PAIRS)} -> " + ", ".join(f"{k}={sum(p['relation']==k for p in TEST_PAIRS)}" for k in PAIR_LIMITS))

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
    model.eval()
    with torch.no_grad():
        loss = loss_for(text)
    ppl = math.exp(loss.item())
    print(f"    eval text={text!r} loss={loss.item():.4f} ppl={ppl:.4f}")
    return ppl

def train_text(text, mask_cpu=None, steps=1):
    # Finetune; True entries in mask_cpu are frozen.
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    mask = None if mask_cpu is None else mask_cpu.to(DEVICE)
    last = None
    for _ in range(steps):
        model.train()
        model.zero_grad(set_to_none=True)
        loss = loss_for(text)
        loss.backward()
        last = loss.item()
        if mask is not None:
            off = 0
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                n = p.numel()
                if p.grad is not None:
                    p.grad.data.masked_fill_(mask[off:off + n].view_as(p), 0.0)
                off += n
        opt.step()
        opt.zero_grad(set_to_none=True)
    pct = 0.0 if mask is None else mask.float().mean().item() * 100.0
    print(f"    train text={text!r} loss={last:.4f} protected={pct:.2f}%")
    del opt, mask
    clear_memory()

def flat_grad(text):
    # Signed gradient footprint at current weights.
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

def top_idx(score, k=K_MAX, positive_only=False):
    # Store only top indices for max K; smaller masks use prefixes.
    keep = max(1, int(score.numel() * k))
    vals, idx = torch.topk(score, keep, sorted=True)
    if positive_only:
        idx = idx[vals > 0]
    return idx.detach().cpu()

def mask_from_idx(idx, k):
    keep = min(len(idx), max(1, int(total_params * k)))
    mask = torch.zeros(total_params, dtype=torch.bool)
    if keep:
        mask[idx[:keep]] = True
    return mask

def random_mask(k, seed):
    # Exact-size random control without full randperm.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    keep = max(1, int(total_params * k))
    out = torch.zeros(total_params, dtype=torch.bool)
    while int(out.sum().item()) < keep:
        need = keep - int(out.sum().item())
        out[torch.randint(total_params, (max(4096, need * 2),), generator=gen)] = True
    extra = int(out.sum().item()) - keep
    if extra > 0:
        out[out.nonzero().flatten()[:extra]] = False
    return out

def jaccard_idx(idx_a, idx_b, k):
    # Jaccard from top-index prefixes without storing all concept masks.
    ma = mask_from_idx(idx_a, k)
    kb = min(len(idx_b), max(1, int(total_params * k)))
    inter = ma[idx_b[:kb]].sum().item()
    union = ma.sum().item() + kb - inter
    return inter / union if union else 1.0

def jaccard_mask(mask_a, mask_b):
    inter = torch.logical_and(mask_a, mask_b).sum().item()
    union = torch.logical_or(mask_a, mask_b).sum().item()
    return inter / union if union else 1.0

def build_run_indices(run_seed):
    # Train every concept once, then save top gradient indices as its memory branch.
    global top_indices
    top_indices = {}
    print("\nBuilding concept memory indices:")
    for name, spec in CONCEPTS.items():
        reset_base()
        set_seed(concept_seed(run_seed, name))
        print(f"  concept={name} [{spec['branch']}/{spec['twig']}]")
        train_text(spec["train"], None, TRAIN_STEPS_A)
        g = flat_grad(spec[MASK_SOURCE])
        top_indices[name] = top_idx(g.abs(), K_MAX)
        print(f"    stored top indices={len(top_indices[name]):,} ({100*len(top_indices[name])/total_params:.2f}%)")
        del g
        clear_memory()

def wrong_concept(a, b):
    # Deterministic wrong branch control, preferably from a different top-level branch.
    for c, spec in CONCEPTS.items():
        if c not in (a, b) and spec["branch"] != CONCEPTS[a]["branch"]:
            return c
    return next(c for c in CONCEPTS if c not in (a, b))

def conflict_indices(after_a, a, b):
    # A-important parameters whose gradient sign conflicts with B.
    load_state(after_a)
    ga = flat_grad(CONCEPTS[a][MASK_SOURCE])
    gb = flat_grad(CONCEPTS[b]["train"])
    conflict = torch.sign(ga) * torch.sign(gb) < 0
    score = ga.abs()
    score[~conflict] = 0
    idx = top_idx(score, K_MAX, positive_only=True)
    print(f"    stored conflict indices={len(idx):,} ({100*len(idx)/total_params:.2f}%)")
    del ga, gb, conflict, score
    clear_memory()
    return idx

def train_a_checkpoint(a, run_seed):
    reset_base()
    set_seed(concept_seed(run_seed, a))
    train_text(CONCEPTS[a]["train"], None, TRAIN_STEPS_A)
    ppl_a1 = eval_ppl(CONCEPTS[a]["eval"])
    return ppl_a1, clone_state()

def run_strategy(after_a, pair, mask_cpu, seed):
    load_state(after_a)
    set_seed(seed)
    train_text(CONCEPTS[pair["B"]]["train"], mask_cpu, TRAIN_STEPS_B)
    return eval_ppl(CONCEPTS[pair["A"]]["eval"]), eval_ppl(CONCEPTS[pair["B"]]["eval"])

def run_pair(run, run_seed, pair, pair_idx):
    # Baseline once; then correct/random/wrong/conflict for each K.
    a, b = pair["A"], pair["B"]
    seed = run_seed + 1009 * pair_idx
    print(f"\n=== Run {run+1}/{NUM_RUNS} Pair {pair_idx+1}/{len(TEST_PAIRS)}: {a}->{b} [{pair['relation']}] ===")
    ppl_a1, after_a = train_a_checkpoint(a, run_seed)
    load_state(after_a)
    ppl_b_before = eval_ppl(CONCEPTS[b]["eval"])
    ppl_a2_base, ppl_b2_base = run_strategy(after_a, pair, None, seed + 77)
    baseline_delta = ppl_a2_base - ppl_a1
    baseline_b_gain = ppl_b_before - ppl_b2_base
    c_wrong = wrong_concept(a, b)
    cidx = conflict_indices(after_a, a, b)
    rows = [{"run": run, **pair, "K": None, "strategy": "no_mask", "wrong_source": None, "jaccard_ab": None,
             "ppl_a1": float(ppl_a1), "ppl_a2": float(ppl_a2_base), "delta_a": float(baseline_delta),
             "baseline_delta": float(baseline_delta), "ppl_b_before": float(ppl_b_before),
             "ppl_b_after": float(ppl_b2_base), "b_learning_gain": float(baseline_b_gain),
             "baseline_b_gain": float(baseline_b_gain), "retention_improvement": 0.0,
             "learning_cost_vs_nomask": 0.0, "mask_pct": 0.0, "jaccard_correct_strategy": None}]

    for k in K_VALUES:
        correct = mask_from_idx(top_indices[a], k)
        wrong = mask_from_idx(top_indices[c_wrong], k)
        conflict = mask_from_idx(cidx, k)
        strategies = {
            "correct_mask": correct,
            "random_mask": random_mask(k, seed + int(k * 1_000_000) + 12345),
            "wrong_mask": wrong,
            "conflict_mask": conflict,
        }
        j_ab = 1.0 if a == b else jaccard_idx(top_indices[a], top_indices[b], k)
        for strat, mask in strategies.items():
            print(f"\n  Strategy={strat} K={k:g}")
            ppl_a2, ppl_b2 = run_strategy(after_a, pair, mask, seed + 777)
            delta = ppl_a2 - ppl_a1
            b_gain = ppl_b_before - ppl_b2
            rows.append({"run": run, **pair, "K": k, "strategy": strat,
                         "wrong_source": c_wrong if strat == "wrong_mask" else None,
                         "jaccard_ab": float(j_ab), "ppl_a1": float(ppl_a1), "ppl_a2": float(ppl_a2),
                         "delta_a": float(delta), "baseline_delta": float(baseline_delta),
                         "ppl_b_before": float(ppl_b_before), "ppl_b_after": float(ppl_b2),
                         "b_learning_gain": float(b_gain), "baseline_b_gain": float(baseline_b_gain),
                         "retention_improvement": float(baseline_delta - delta),
                         "learning_cost_vs_nomask": float(baseline_b_gain - b_gain),
                         "mask_pct": float(mask.float().mean().item() * 100.0),
                         "jaccard_correct_strategy": float(jaccard_mask(correct, mask))})
            print(f"  delta_A={delta:.4f} improvement={baseline_delta-delta:.4f} B_gain={b_gain:.4f}")
        del correct, wrong, conflict, strategies
        clear_memory()
    del after_a, cidx
    clear_memory()
    return rows

def geometry_report(rows):
    # Test whether semantic distance orders Jaccard as predicted.
    geo = []
    names = list(CONCEPTS)
    for k in K_VALUES:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                geo.append({"K": k, "A": a, "B": b, "relation": relation(a, b),
                            "jaccard": float(jaccard_idx(top_indices[a], top_indices[b], k))})
    gdf = pd.DataFrame(geo)
    stats = gdf.groupby(["K", "relation"])["jaccard"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    print("\nGeometry Jaccard stats over all concept pairs:")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))
    checks = {}
    for k in K_VALUES:
        means = dict(zip(stats[stats.K == k]["relation"], stats[stats.K == k]["mean"]))
        checks[f"geometry_K={k:g}"] = bool(means["same_twig"] > means["same_branch"] > means["distant_branch"] and means["distant_branch"] < 0.20)
    return gdf, stats, checks

def corr_report(df):
    # Falsification: lower Jaccard should predict stronger retention improvement.
    out = []
    non_same = df[(df.strategy != "no_mask") & (df.relation != "same_concept")].copy()
    for (k, strat), g in non_same.groupby(["K", "strategy"]):
        pearson = float(g["jaccard_ab"].corr(g["retention_improvement"], method="pearson"))
        spearman = float(g["jaccard_ab"].corr(g["retention_improvement"], method="spearman"))
        out.append({"K": k, "strategy": strat, "pearson_jaccard_vs_improvement": pearson,
                    "spearman_jaccard_vs_improvement": spearman, "count": int(len(g))})
    cdf = pd.DataFrame(out)
    print("\nJaccard -> retention improvement correlation, non-same pairs:")
    print(cdf.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))
    return cdf

def summarize(rows):
    df = pd.DataFrame(rows)
    cols = ["run", "A", "B", "relation", "K", "strategy", "jaccard_ab", "delta_a",
            "retention_improvement", "b_learning_gain", "learning_cost_vs_nomask", "mask_pct"]
    print("\nFull retention results:")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:10.4f}"))
    protected = df[df.strategy != "no_mask"].copy()
    by_rel = protected.groupby(["K", "relation", "strategy"]).agg(
        mean_improvement=("retention_improvement", "mean"),
        pct_improved=("retention_improvement", lambda s: float((s > 0).mean() * 100.0)),
        mean_learning_cost=("learning_cost_vs_nomask", "mean"),
        count=("retention_improvement", "count"),
    ).reset_index()
    by_strat = protected.groupby(["K", "strategy"]).agg(
        mean_improvement=("retention_improvement", "mean"),
        pct_improved=("retention_improvement", lambda s: float((s > 0).mean() * 100.0)),
        mean_learning_cost=("learning_cost_vs_nomask", "mean"),
        count=("retention_improvement", "count"),
    ).reset_index()
    print("\nRetention summary by relation:")
    print(by_rel.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))
    print("\nRetention summary by strategy:")
    print(by_strat.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))
    return df, by_rel, by_strat

def final_checks(df, geo_checks, corr):
    # Main pass/fail gates for the final falsification test.
    checks = dict(geo_checks)
    non_same = df[(df.strategy != "no_mask") & (df.relation != "same_concept")]
    same = df[(df.strategy != "no_mask") & (df.relation == "same_concept")]
    for k in K_VALUES:
        ns = non_same[non_same.K == k].groupby("strategy")["retention_improvement"].mean().to_dict()
        ss_imp = same[same.K == k].groupby("strategy")["retention_improvement"].mean().to_dict()
        ss_cost = same[same.K == k].groupby("strategy")["learning_cost_vs_nomask"].mean().to_dict()
        c = corr[(corr.K == k) & (corr.strategy == "correct_mask")]["spearman_jaccard_vs_improvement"].iloc[0]
        checks[f"K={k:g}:correct_beats_random"] = bool(ns.get("correct_mask", -1e9) > ns.get("random_mask", 1e9))
        checks[f"K={k:g}:correct_beats_wrong"] = bool(ns.get("correct_mask", -1e9) > ns.get("wrong_mask", 1e9))
        checks[f"K={k:g}:conflict_beats_random"] = bool(ns.get("conflict_mask", -1e9) > ns.get("random_mask", 1e9))
        checks[f"K={k:g}:same_concept_correct_hurts"] = bool(ss_imp.get("correct_mask", 1e9) < 0 and ss_cost.get("correct_mask", -1e9) > 0)
        checks[f"K={k:g}:jaccard_predicts_correct"] = bool(c < 0)
    key_k = 0.03
    required = [checks[f"geometry_K={key_k:g}"], checks[f"K={key_k:g}:correct_beats_random"],
                checks[f"K={key_k:g}:correct_beats_wrong"], checks[f"K={key_k:g}:same_concept_correct_hurts"],
                checks[f"K={key_k:g}:jaccard_predicts_correct"]]
    checks["FINAL_PASS_K=0.03"] = bool(all(required))
    print("\nFinal falsification checks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"\nFinal Test 4 verdict: {'PASS' if checks['FINAL_PASS_K=0.03'] else 'FAIL'}")
    return checks

def main():
    started = time.time()
    load_base()
    all_rows = []
    for run in range(NUM_RUNS):
        run_seed = SEED + 10000 * run
        print(f"\n################ RUN {run+1}/{NUM_RUNS} seed={run_seed} ################")
        build_run_indices(run_seed)
        for i, pair in enumerate(TEST_PAIRS):
            all_rows.extend(run_pair(run, run_seed, pair, i))
    df, by_rel, by_strat = summarize(all_rows)
    gdf, gstats, geo_checks = geometry_report(all_rows)
    cdf = corr_report(df)
    checks = final_checks(df, geo_checks, cdf)
    payload = {"config": {"MODEL": MODEL, "LR": LR, "SEED": SEED, "NUM_RUNS": NUM_RUNS,
                          "K_VALUES": K_VALUES, "TRAIN_STEPS_A": TRAIN_STEPS_A,
                          "TRAIN_STEPS_B": TRAIN_STEPS_B, "PAIR_LIMITS": PAIR_LIMITS,
                          "MASK_SOURCE": MASK_SOURCE, "DEVICE": str(DEVICE), "TEST_PAIRS": TEST_PAIRS},
               "rows": df.to_dict(orient="records"),
               "retention_by_relation": by_rel.to_dict(orient="records"),
               "retention_by_strategy": by_strat.to_dict(orient="records"),
               "geometry_pairs": gdf.to_dict(orient="records"),
               "geometry_stats": gstats.to_dict(orient="records"),
               "correlations": cdf.to_dict(orient="records"),
               "checks": checks,
               "final": {"pass": bool(checks["FINAL_PASS_K=0.03"]),
                         "runtime_sec": round(time.time() - started, 2)}}
    with open("tree_test4_final_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Runtime: {payload['final']['runtime_sec']} sec")
    print("Saved tree_test4_final_results.json")

if __name__ == "__main__":
    main()
