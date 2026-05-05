import gc, json, math, random, time
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL, LR, SEED = "gpt2", 5e-5, 42
K = 0.03
TRAIN_STEPS_A = 1
TRAIN_STEPS_B = 1
MASK_SOURCE = "train"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONCEPTS = {
    "dog": {"branch": "living", "twig": "mammals", "train": "Golden retrievers are friendly dogs", "eval": "Golden retrievers"},
    "cat": {"branch": "living", "twig": "mammals", "train": "Persian cats have soft fur", "eval": "Persian cats"},
    "car_tires": {"branch": "artifacts", "twig": "vehicles", "train": "Michelin produces rubber car tires", "eval": "Michelin produces"},
    "car_engine": {"branch": "artifacts", "twig": "vehicles", "train": "Tesla car engines use electric motors", "eval": "Tesla car"},
}

TEST_PAIRS = [
    {"A": "car_tires", "B": "dog", "relation": "distant_branch", "expect": "freeze should protect A"},
    {"A": "dog", "B": "car_tires", "relation": "distant_branch", "expect": "freeze should protect A"},
    {"A": "car_tires", "B": "car_engine", "relation": "same_twig", "expect": "freeze may protect A, but less"},
    {"A": "dog", "B": "cat", "relation": "same_twig", "expect": "freeze may protect A, but less"},
    {"A": "car_tires", "B": "car_tires", "relation": "same_concept", "expect": "freeze should hurt learning"},
]

tok, model, base_state = None, None, None

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clear_memory():
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

def load_base():
    # Load GPT-2 on GPU if available and keep clean pretrained weights on CPU.
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
    # Restore model weights from either CPU base or GPU after-A checkpoint.
    model.load_state_dict(state)
    model.zero_grad(set_to_none=True)

def reset_base():
    load_state(base_state)
    clear_memory()

def clone_state():
    # Keep the post-A checkpoint on GPU for fast resets across strategies.
    return {k: v.detach().clone() for k, v in model.state_dict().items()}

def loss_for(text):
    batch = tok(text, return_tensors="pt").to(DEVICE)
    return model(**batch, labels=batch["input_ids"]).loss

def eval_ppl(text):
    # Perplexity on the eval prompt, no gradient update.
    model.eval()
    with torch.no_grad():
        loss = loss_for(text)
    ppl = math.exp(loss.item())
    print(f"    eval text={text!r} loss={loss.item():.4f} ppl={ppl:.4f}")
    return ppl

def train_step(text, active_mask=None, steps=1):
    # Finetune text; masked parameters get zero gradients.
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
    protected = 0.0 if active_mask is None else active_mask.float().mean().item() * 100.0
    print(f"    train text={text!r} loss={last_loss:.4f} protected={protected:.2f}%")
    del opt
    clear_memory()

def flat_grad(text):
    # Flat signed gradient footprint at the current model state.
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
    # 1-bit mask from top-k scores.
    keep = max(1, int(score.numel() * k))
    vals, idx = torch.topk(score, keep, sorted=False)
    if positive_only:
        idx = idx[vals > 0]
    mask = torch.zeros(score.numel(), dtype=torch.bool, device=DEVICE)
    mask[idx] = True
    return mask.detach().cpu()

def random_mask_like(mask, seed):
    # Random control mask with exactly the same number of True bits.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    n, keep = mask.numel(), int(mask.sum().item())
    out = torch.zeros(n, dtype=torch.bool)
    while int(out.sum().item()) < keep:
        need = keep - int(out.sum().item())
        out[torch.randint(n, (max(4096, need * 2),), generator=gen)] = True
    if int(out.sum().item()) > keep:
        extra = int(out.sum().item()) - keep
        out[out.nonzero().flatten()[:extra]] = False
    return out

def jaccard(a, b):
    inter = torch.logical_and(a, b).sum().item()
    union = torch.logical_or(a, b).sum().item()
    return inter / union if union else 1.0

def build_masks_after_a(after_a_state, A, B, pair_seed):
    # Correct mask protects A-important params; conflict mask protects A-important params opposed by B.
    load_state(after_a_state)
    ga = flat_grad(CONCEPTS[A][MASK_SOURCE])
    correct = topk_mask(ga.abs(), K)
    gb = flat_grad(CONCEPTS[B]["train"])
    conflict = torch.sign(ga) * torch.sign(gb) < 0
    score = ga.abs()
    score[~conflict] = 0
    conflict_mask = topk_mask(score, K, positive_only=True)
    random_mask = random_mask_like(correct, pair_seed + 12345)
    print(f"    mask correct={correct.float().mean().item()*100:.2f}% random={random_mask.float().mean().item()*100:.2f}% conflict={conflict_mask.float().mean().item()*100:.2f}%")
    del ga, gb, score, conflict
    clear_memory()
    return {"correct_mask": correct, "random_mask": random_mask, "conflict_mask": conflict_mask}

def run_strategy(after_a_state, pair, strategy, mask_cpu, base_seed):
    # Reset to after-A, train B once, then re-evaluate A retention and B learning.
    load_state(after_a_state)
    set_seed(base_seed)
    mask = None if mask_cpu is None else mask_cpu.to(DEVICE)
    for _ in range(TRAIN_STEPS_B):
        train_step(CONCEPTS[pair["B"]]["train"], mask, steps=1)
    ppl_a2 = eval_ppl(CONCEPTS[pair["A"]]["eval"])
    ppl_b2 = eval_ppl(CONCEPTS[pair["B"]]["eval"])
    if mask is not None:
        del mask
    clear_memory()
    return {"strategy": strategy, "ppl_a2": ppl_a2, "ppl_b2": ppl_b2}

def run_pair(pair, pair_idx):
    # Train A, build masks, then compare no-mask/correct/random/conflict when learning B.
    A, B = pair["A"], pair["B"]
    pair_seed = SEED + pair_idx * 1000
    print(f"\n=== Pair {pair_idx + 1}/{len(TEST_PAIRS)}: {A} -> {B} [{pair['relation']}] ===")
    print(f"Expectation: {pair['expect']}")
    reset_base()
    set_seed(pair_seed)
    for _ in range(TRAIN_STEPS_A):
        train_step(CONCEPTS[A]["train"], None, steps=1)
    ppl_a1 = eval_ppl(CONCEPTS[A]["eval"])
    ppl_b_before = eval_ppl(CONCEPTS[B]["eval"])
    after_a = clone_state()
    masks = build_masks_after_a(after_a, A, B, pair_seed)
    masks["no_mask"] = None

    rows = []
    for strategy in ["no_mask", "correct_mask", "random_mask", "conflict_mask"]:
        print(f"\n  Strategy: {strategy}")
        out = run_strategy(after_a, pair, strategy, masks[strategy], pair_seed + 777)
        delta_a = out["ppl_a2"] - ppl_a1
        b_gain = ppl_b_before - out["ppl_b2"]
        rows.append({
            **pair,
            "strategy": strategy,
            "ppl_a1": float(ppl_a1),
            "ppl_a2": float(out["ppl_a2"]),
            "delta_a": float(delta_a),
            "ppl_b_before": float(ppl_b_before),
            "ppl_b_after": float(out["ppl_b2"]),
            "b_learning_gain": float(b_gain),
            "mask_pct": 0.0 if masks[strategy] is None else float(masks[strategy].float().mean().item() * 100.0),
            "jaccard_correct_conflict": None if strategy != "conflict_mask" else float(jaccard(masks["correct_mask"], masks["conflict_mask"])),
        })
        print(f"  delta_A={delta_a:.4f} B_learning_gain={b_gain:.4f}")
    del after_a, masks
    clear_memory()
    return rows

def summarize(df):
    # Positive improvement means better A retention than ordinary finetuning B.
    base = df[df["strategy"] == "no_mask"][["A", "B", "delta_a", "b_learning_gain"]].rename(columns={"delta_a": "baseline_delta_a", "b_learning_gain": "baseline_b_gain"})
    out = df.merge(base, on=["A", "B"])
    out["retention_improvement"] = out["baseline_delta_a"] - out["delta_a"]
    out["learning_cost_vs_nomask"] = out["baseline_b_gain"] - out["b_learning_gain"]

    print("\nFull results:")
    cols = ["A", "B", "relation", "strategy", "delta_a", "retention_improvement", "b_learning_gain", "learning_cost_vs_nomask", "mask_pct"]
    print(out[cols].to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    print("\nMean retention improvement by relation and strategy:")
    agg = out[out["strategy"] != "no_mask"].groupby(["relation", "strategy"])["retention_improvement"].agg(["mean", "count"]).reset_index()
    print(agg.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    checks = {}
    distant = out[(out["relation"] == "distant_branch") & (out["strategy"] == "correct_mask")]["retention_improvement"].mean()
    distant_conflict = out[(out["relation"] == "distant_branch") & (out["strategy"] == "conflict_mask")]["retention_improvement"].mean()
    same_twig = out[(out["relation"] == "same_twig") & (out["strategy"] == "correct_mask")]["retention_improvement"].mean()
    same_concept_row = out[(out["relation"] == "same_concept") & (out["strategy"] == "correct_mask")].iloc[0]
    random_mean = out[out["strategy"] == "random_mask"]["retention_improvement"].mean()
    conflict_mean = out[out["strategy"] == "conflict_mask"]["retention_improvement"].mean()
    checks["distant_correct_protects"] = bool(distant > 0)
    checks["distant_conflict_protects"] = bool(distant_conflict > 0)
    checks["same_twig_correct_not_stronger_than_distant"] = bool(same_twig <= distant)
    checks["same_concept_correct_hurts_learning"] = bool(same_concept_row["delta_a"] > same_concept_row["baseline_delta_a"])
    checks["conflict_beats_random"] = bool(conflict_mean > random_mean)

    print("\nPrediction checks:")
    print(f"  distant correct_mask mean improvement > 0: {checks['distant_correct_protects']} ({distant:.4f})")
    print(f"  distant conflict_mask mean improvement > 0: {checks['distant_conflict_protects']} ({distant_conflict:.4f})")
    print(f"  same_twig correct_mask <= distant correct_mask: {checks['same_twig_correct_not_stronger_than_distant']} ({same_twig:.4f} <= {distant:.4f})")
    print(f"  same_concept correct_mask hurts learning: {checks['same_concept_correct_hurts_learning']}")
    print(f"  conflict_mask beats random_mask overall: {checks['conflict_beats_random']} ({conflict_mean:.4f} > {random_mean:.4f})")
    tree_pass = checks["distant_correct_protects"] and checks["same_concept_correct_hurts_learning"]
    conflict_pass = checks["distant_conflict_protects"] and checks["conflict_beats_random"]
    print(f"\nNaive Tree Memory freeze result: {'PASS' if tree_pass else 'FAIL'}")
    print(f"Conflict-aware Tree Memory result: {'PASS' if conflict_pass else 'FAIL'}")
    return out, agg, checks, tree_pass, conflict_pass

def main():
    started = time.time()
    load_base()
    rows = []
    for i, pair in enumerate(TEST_PAIRS):
        rows.extend(run_pair(pair, i))
    df = pd.DataFrame(rows)
    full, agg, checks, tree_pass, conflict_pass = summarize(df)
    payload = {
        "config": {"MODEL": MODEL, "LR": LR, "SEED": SEED, "K": K, "TRAIN_STEPS_A": TRAIN_STEPS_A, "TRAIN_STEPS_B": TRAIN_STEPS_B, "MASK_SOURCE": MASK_SOURCE, "DEVICE": str(DEVICE)},
        "rows": full.to_dict(orient="records"),
        "summary": agg.to_dict(orient="records"),
        "checks": checks,
        "final": {"naive_tree_pass": bool(tree_pass), "conflict_aware_pass": bool(conflict_pass), "runtime_sec": round(time.time() - started, 2)},
    }
    with open("tree_retention_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nFinal Test 2 verdict: naive_tree={'PASS' if tree_pass else 'FAIL'}, conflict_aware={'PASS' if conflict_pass else 'FAIL'}")
    print(f"Runtime: {payload['final']['runtime_sec']} sec")
    print("Saved tree_retention_results.json")

if __name__ == "__main__":
    main()
