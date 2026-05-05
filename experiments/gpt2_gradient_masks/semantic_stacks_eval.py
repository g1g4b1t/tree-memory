import gc, json, math, random
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL, LR, SEED, NUM_RUNS = "gpt2", 5e-5, 42, 3
K_VALUES = [0.001, 0.003, 0.01, 0.03, 0.10]
K_MAIN = 0.03
MASK_TEXT = "train"
RUN_PROTECTION_ONLY_IF_FORGETTING = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONCEPTS = {
    "car_tires": {"build": "The car has rubber tires for road grip", "train": "Michelin produces car tires", "eval": "Michelin produces"},
    "car_engine": {"build": "The car engine uses fuel combustion", "train": "Tesla car engine is electric", "eval": "Tesla car"},
    "bird_wings": {"build": "Birds use wings to fly in the sky", "train": "Eagles have powerful wings", "eval": "Eagles have"},
    "bird_beak": {"build": "Birds use beaks to eat food", "train": "Parrots have curved beaks", "eval": "Parrots have"},
    "code_python": {"build": "Python code uses indentation for blocks", "train": "Python lists use brackets", "eval": "Python lists"},
    "code_cpp": {"build": "C++ code uses semicolons to end lines", "train": "C++ vectors use templates", "eval": "C++ vectors"},
}

tok, model, base_state = None, None, None
run_masks, pair_seed, last_detail = {}, SEED, {}

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def clear_memory():
    gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

def load_base():
    # Load GPT-2 on GPU if available; keep the base checkpoint on CPU.
    global tok, model, base_state
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")
    tok = GPT2TokenizerFast.from_pretrained(MODEL)
    model = GPT2LMHeadModel.from_pretrained(MODEL).to(DEVICE)
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {MODEL}, trainable params: {n:,}")

def load_state(state):
    # Restore either the CPU base state or a GPU after-A state.
    model.load_state_dict(state)
    model.zero_grad(set_to_none=True)

def reset_base():
    load_state(base_state); clear_memory()

def clone_model_state():
    # Keep after-A weights on GPU for fast strategy resets within one pair.
    return {k: v.detach().clone() for k, v in model.state_dict().items()}

def loss_for(text):
    batch = tok(text, return_tensors="pt").to(DEVICE)
    return model(**batch, labels=batch["input_ids"]).loss

def train_step(text, active_mask=None):
    # One AdamW step; active_mask protects True entries from changing.
    model.train(); model.zero_grad(set_to_none=True)
    loss = loss_for(text); loss.backward()
    protected = 0.0 if active_mask is None else active_mask.float().mean().item() * 100.0
    if active_mask is not None:
        off = 0
        for p in model.parameters():
            if not p.requires_grad: continue
            n = p.numel(); m = active_mask[off:off + n].view_as(p)
            if p.grad is not None: p.grad.data.masked_fill_(m, 0.0)
            off += n
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    print(f"    train text={text!r} loss={loss.item():.4f} protected={protected:.2f}%")
    del opt; clear_memory()

def flat_grad(text):
    # Return the flat gradient vector for text at the current model state.
    model.train(); model.zero_grad(set_to_none=True)
    loss = loss_for(text); loss.backward()
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
    # Convert scores into a 1-bit mask for the top-k fraction.
    keep = max(1, int(score.numel() * k))
    vals, idx = torch.topk(score, keep)
    if positive_only: idx = idx[vals > 0]
    mask = torch.zeros(score.numel(), dtype=torch.bool, device=DEVICE)
    mask[idx] = True
    return mask

def masks_from_score(score):
    return {k: topk_mask(score, k).detach().cpu() for k in K_VALUES}

def jaccard(a, b):
    a, b = a.bool().reshape(-1), b.bool().reshape(-1)
    inter = torch.logical_and(a, b).sum().item()
    union = torch.logical_or(a, b).sum().item()
    return inter / union if union else 1.0

def jaccard_matrix(stacks):
    # ASCII heatmap for concept-specific memory masks at K_MAIN.
    names = list(stacks)
    data = [[jaccard(stacks[a][K_MAIN], stacks[b][K_MAIN]) for b in names] for a in names]
    df = pd.DataFrame(data, index=names, columns=names)
    print(f"\nJaccard heatmap, post-train memory masks, K={K_MAIN}:")
    print(df.to_string(float_format=lambda x: f"{x:6.3f}"))
    return df

def eval_ppl(text):
    # Evaluate exp(loss) without updating weights.
    model.eval()
    with torch.no_grad(): loss = loss_for(text)
    ppl = math.exp(loss.item())
    print(f"    eval text={text!r} loss={loss.item():.4f} ppl={ppl:.4f}")
    return ppl

def concept_seed(run_seed, name):
    return run_seed + list(CONCEPTS).index(name) * 100

def build_run_masks(run_seed):
    # Train each concept once, then build its memory mask after learning.
    global run_masks
    run_masks = {}
    print("\nBuilding post-train memory masks:")
    for name, spec in CONCEPTS.items():
        reset_base(); set_seed(concept_seed(run_seed, name))
        train_step(spec["train"])
        g = flat_grad(spec[MASK_TEXT]).abs()
        run_masks[name] = masks_from_score(g)
        print(f"  memory_stack {name}: " + ", ".join(f"K={k:g}:{run_masks[name][k].float().mean().item()*100:.2f}%" for k in K_VALUES))
        del g; clear_memory()
    return jaccard_matrix(run_masks)

def random_mask_like(mask, seed):
    # Random 1-bit control mask with the same number of protected parameters.
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
    n, keep = mask.numel(), int(mask.sum().item())
    out = torch.zeros(n, dtype=torch.bool)
    while int(out.sum().item()) < keep:
        need = keep - int(out.sum().item())
        out[torch.randint(n, (max(1024, need * 2),), generator=gen)] = True
    if int(out.sum().item()) > keep:
        extra = int(out.sum().item()) - keep
        out[out.nonzero().flatten()[:extra]] = False
    return out

def wrong_concept(a, b):
    return next(c for c in CONCEPTS if c not in (a, b))

def or_masks(masks):
    out = torch.zeros_like(masks[0])
    for m in masks: out |= m
    return out

def conflict_masks(after_a_state, a, b):
    # Protect only A-important params whose gradient sign conflicts with B.
    load_state(after_a_state)
    ga = flat_grad(CONCEPTS[a][MASK_TEXT])
    gb = flat_grad(CONCEPTS[b]["train"])
    conflict = torch.sign(ga) * torch.sign(gb) < 0
    score = ga.abs(); score[~conflict] = 0
    masks = {k: topk_mask(score, k, positive_only=True).detach().cpu() for k in K_VALUES}
    for k, m in masks.items(): print(f"    conflict_mask K={k:g} density={m.float().mean().item()*100:.2f}%")
    del ga, gb, score, conflict; clear_memory()
    return masks

def run_strategy(after_a_state, text_b, eval_a, mask_cpu):
    # Reset to after-A, train B under one mask, and report forgetting delta.
    load_state(after_a_state)
    mask = None if mask_cpu is None else mask_cpu.to(DEVICE)
    train_step(text_b, mask)
    ppl_a2 = eval_ppl(eval_a)
    if mask is not None: del mask
    return ppl_a2

def get_pair_results(run, run_seed, a, b, jmat):
    # Baseline first; controls run only if B actually hurts A.
    global last_detail
    set_seed(concept_seed(run_seed, a)); reset_base()
    train_step(CONCEPTS[a]["train"])
    ppl_a1 = eval_ppl(CONCEPTS[a]["eval"])
    after_a = clone_model_state()
    ppl_base = run_strategy(after_a, CONCEPTS[b]["train"], CONCEPTS[a]["eval"], None)
    baseline_delta = ppl_base - ppl_a1
    forgetting = baseline_delta > 0
    last_detail = {"ppl_a1": ppl_a1, "baseline_ppl_a2": ppl_base}
    rows = []
    print(f"    baseline_delta={baseline_delta:.4f} forgetting_pair={forgetting}")
    if RUN_PROTECTION_ONLY_IF_FORGETTING and not forgetting:
        print("    skipping protection controls because baseline has no forgetting")
        del after_a; clear_memory()
        return rows, baseline_delta, forgetting

    c_wrong = wrong_concept(a, b)
    cmasks = conflict_masks(after_a, a, b)
    for k in K_VALUES:
        strategies = {
            "correct_freeze": run_masks[a][k],
            "random_freeze": random_mask_like(run_masks[a][k], run_seed + 10000 * list(CONCEPTS).index(a) + 1000 * list(CONCEPTS).index(b) + int(k * 1_000_000)),
            "wrong_freeze": run_masks[c_wrong][k],
            "all_except_B_freeze": or_masks([run_masks[c][k] for c in CONCEPTS if c != b]),
            "conflict_freeze": cmasks[k],
        }
        for strat, mask in strategies.items():
            print(f"    strategy={strat} K={k:g}")
            ppl_s = run_strategy(after_a, CONCEPTS[b]["train"], CONCEPTS[a]["eval"], mask)
            delta = ppl_s - ppl_a1
            rows.append({"run": run, "A": a, "B": b, "K": k, "strategy": strat,
                         "Jaccard(A,B)": float(jmat.loc[a, b]), "baseline_delta": float(baseline_delta),
                         "strategy_delta": float(delta), "improvement": float(baseline_delta - delta),
                         "mask_pct": float(mask.float().mean().item() * 100.0), "forgetting_pair": forgetting,
                         **last_detail, "strategy_ppl_a2": float(ppl_s)})
            print(f"    delta={delta:.4f} improvement={baseline_delta - delta:.4f}")
    del after_a, cmasks; clear_memory()
    return rows, baseline_delta, forgetting

def summarize(rows, baseline_rows):
    df = pd.DataFrame(rows)
    bdf = pd.DataFrame(baseline_rows)
    print("\nBaseline forgetting overview:")
    print(bdf.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))
    if df.empty:
        print("\nNo forgetting pairs found; protection controls were not run.")
        return {}, False
    print("\nProtection results sorted by improvement:")
    cols = ["run", "A", "B", "K", "strategy", "Jaccard(A,B)", "baseline_delta", "strategy_delta", "improvement", "mask_pct"]
    print(df.sort_values("improvement", ascending=False)[cols].to_string(index=False, float_format=lambda x: f"{x:10.4f}"))
    summary = {}
    print("\nSummary on forgetting pairs only:")
    for (k, strat), g in df.groupby(["K", "strategy"]):
        vals = g.groupby("run").agg(pct_improved=("improvement", lambda s: float((s > 0).mean() * 100.0)),
                                    mean_improvement=("improvement", "mean")).reset_index()
        rec = {"pairs": int(len(g)), "pct_improved_mean": float(vals["pct_improved"].mean()),
               "pct_improved_std": float(vals["pct_improved"].std()), "mean_improvement": float(vals["mean_improvement"].mean()),
               "std_improvement": float(vals["mean_improvement"].std())}
        summary[f"K={k:g}:{strat}"] = rec
        print(f"K={k:g} {strat}: pairs={rec['pairs']} improved={rec['pct_improved_mean']:.2f}% +/- {rec['pct_improved_std']:.2f}% mean_improvement={rec['mean_improvement']:.4f} +/- {rec['std_improvement']:.4f}")
    key = f"K={K_MAIN:g}:conflict_freeze"
    ok = key in summary and summary[key]["pct_improved_mean"] > 70 and summary[key]["mean_improvement"] > 0
    return summary, ok

def main():
    load_base()
    rows, baseline_rows, jaccards = [], [], []
    names = list(CONCEPTS)
    for run in range(NUM_RUNS):
        run_seed = SEED + run * 1000
        print(f"\n=== RUN {run + 1}/{NUM_RUNS} seed={run_seed} ===")
        jmat = build_run_masks(run_seed)
        jaccards += [float(jmat.loc[a, b]) for a in names for b in names if a != b]
        for a in names:
            for b in names:
                if a == b: continue
                print(f"\n  Pair {a} -> {b}")
                prs, bd, forgetting = get_pair_results(run, run_seed, a, b, jmat)
                baseline_rows.append({"run": run, "A": a, "B": b, "Jaccard(A,B)": float(jmat.loc[a, b]),
                                      "baseline_delta": float(bd), "forgetting_pair": bool(forgetting)})
                rows.extend(prs)
    summary, protection_ok = summarize(rows, baseline_rows)
    mean_j, std_j = float(pd.Series(jaccards).mean()), float(pd.Series(jaccards).std())
    success = bool(mean_j < 0.5 and protection_ok)
    print("\nFinal verdict:")
    print(f"Mean Jaccard: {mean_j:.4f} +/- {std_j:.4f}")
    print(f"Success rule: mean_jaccard < 0.5 and K={K_MAIN:g} conflict_freeze improves >70% with positive mean improvement")
    print("SUCCESS" if success else "FAILURE")
    payload = {"config": {"MODEL": MODEL, "LR": LR, "SEED": SEED, "NUM_RUNS": NUM_RUNS, "K_VALUES": K_VALUES,
                          "K_MAIN": K_MAIN, "MASK_TEXT": MASK_TEXT, "DEVICE": str(DEVICE)},
               "baseline_rows": baseline_rows, "rows": rows, "summary": summary,
               "final": {"mean_jaccard": mean_j, "std_jaccard": std_j, "success": success}}
    with open("results.json", "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)
    print("\nSaved results.json")

if __name__ == "__main__":
    main()
