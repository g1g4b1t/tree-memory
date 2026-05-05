import gc, json, math, random, time
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL, LR, SEED = "gpt2", 5e-5, 42
K_VALUES = [0.03]
TRAIN_STEPS = 1
MASK_SOURCE = "build"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONCEPTS = {
    "dog": {"branch": "living", "twig": "mammals", "build": "Dogs are mammals with fur, paws, and a strong sense of smell", "train": "Golden retrievers are friendly dogs", "eval": "Golden retrievers"},
    "cat": {"branch": "living", "twig": "mammals", "build": "Cats are mammals with whiskers, claws, and soft fur", "train": "Persian cats have soft fur", "eval": "Persian cats"},
    "eagle": {"branch": "living", "twig": "birds", "build": "Eagles are birds with wings, feathers, and sharp beaks", "train": "Eagles fly with powerful wings", "eval": "Eagles fly"},
    "car_tires": {"branch": "artifacts", "twig": "vehicles", "build": "Car tires are rubber wheels that grip the road", "train": "Michelin produces rubber car tires", "eval": "Michelin produces"},
    "car_engine": {"branch": "artifacts", "twig": "vehicles", "build": "A car engine powers a vehicle using combustion or electric motors", "train": "Tesla car engines use electric motors", "eval": "Tesla car"},
    "hammer": {"branch": "artifacts", "twig": "tools", "build": "A hammer is a tool with a handle and heavy metal head", "train": "A hammer drives nails into wood", "eval": "A hammer"},
}

PAIR_GROUPS = {
    "same_twig": [("dog", "cat"), ("car_tires", "car_engine")],
    "same_branch": [("dog", "eagle"), ("cat", "eagle"), ("car_tires", "hammer"), ("car_engine", "hammer")],
    "distant_branch": [(a, b) for a in ["dog", "cat", "eagle"] for b in ["car_tires", "car_engine", "hammer"]],
}

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
    # Load GPT-2 on GPU if available; keep clean pretrained weights on CPU.
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

def reset_base():
    # Reset model before building each concept mask.
    model.load_state_dict(base_state)
    model.zero_grad(set_to_none=True)
    clear_memory()

def loss_for(text):
    batch = tok(text, return_tensors="pt").to(DEVICE)
    return model(**batch, labels=batch["input_ids"]).loss

def train_step(text):
    # One tiny concept-learning step.
    model.train()
    model.zero_grad(set_to_none=True)
    loss = loss_for(text)
    loss.backward()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    print(f"  train text={text!r} loss={loss.item():.4f}")
    del opt
    clear_memory()

def flat_abs_grad(text):
    # Deterministic gradient footprint for the concept prompt after learning.
    model.eval()
    model.zero_grad(set_to_none=True)
    loss = loss_for(text)
    loss.backward()
    parts = []
    for p in model.parameters():
        if p.requires_grad:
            g = p.grad
            parts.append(torch.zeros(p.numel(), device=DEVICE) if g is None else g.detach().abs().reshape(-1).clone())
    flat = torch.cat(parts)
    model.zero_grad(set_to_none=True)
    print(f"  mask_source text={text!r} loss={loss.item():.4f}")
    return flat

def topk_mask(score, k):
    # 1-bit branch mask: True means this parameter is in the top-k gradient mass.
    keep = max(1, int(score.numel() * k))
    _, idx = torch.topk(score, keep, sorted=False)
    mask = torch.zeros(score.numel(), dtype=torch.bool, device=DEVICE)
    mask[idx] = True
    return mask.detach().cpu()

def build_concept_masks(name, spec):
    # Train concept from a clean GPT-2, then extract its semantic branch mask.
    reset_base()
    set_seed(SEED + 100 * list(CONCEPTS).index(name))
    print(f"\nBuilding mask for {name} [{spec['branch']} / {spec['twig']}]")
    for _ in range(TRAIN_STEPS):
        train_step(spec["train"])
    score = flat_abs_grad(spec[MASK_SOURCE])
    masks = {}
    for k in K_VALUES:
        masks[k] = topk_mask(score, k)
        print(f"  mask K={k:g}: {masks[k].float().mean().item() * 100:.2f}% params")
    del score
    clear_memory()
    return masks

def jaccard(a, b):
    # Intersection over union of two boolean branch masks.
    inter = torch.logical_and(a, b).sum().item()
    union = torch.logical_or(a, b).sum().item()
    return inter / union if union else 1.0

def pair_group(a, b):
    for group, pairs in PAIR_GROUPS.items():
        if (a, b) in pairs or (b, a) in pairs:
            return group
    return "unknown"

def evaluate_geometry(masks, k):
    # Print Jaccard heatmap and group-level hierarchy statistics.
    names = list(CONCEPTS)
    matrix = pd.DataFrame([[jaccard(masks[a][k], masks[b][k]) for b in names] for a in names], index=names, columns=names)
    print(f"\nJaccard heatmap, K={k:g}:")
    print(matrix.to_string(float_format=lambda x: f"{x:6.3f}"))

    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({"A": a, "B": b, "group": pair_group(a, b), "jaccard": float(matrix.loc[a, b])})
    df = pd.DataFrame(rows)
    print(f"\nPair table, K={k:g}:")
    print(df.sort_values(["group", "jaccard"], ascending=[True, False]).to_string(index=False, float_format=lambda x: f"{x:7.4f}"))

    stats = df.groupby("group")["jaccard"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    print(f"\nGroup stats, K={k:g}:")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:7.4f}"))

    means = dict(zip(stats["group"], stats["mean"]))
    dog_tires = float(matrix.loc["dog", "car_tires"])
    hierarchy_ok = means["same_twig"] > means["same_branch"] > means["distant_branch"]
    distant_ok = means["distant_branch"] < 0.20 and dog_tires < 0.20
    print(f"\nPrediction checks, K={k:g}:")
    print(f"  same_twig > same_branch > distant_branch: {hierarchy_ok}")
    print(f"  mean distant_branch < 0.20: {means['distant_branch'] < 0.20}")
    print(f"  Jaccard(dog, car_tires) = {dog_tires:.4f} < 0.20: {dog_tires < 0.20}")
    print("  TEST 1 RESULT:", "PASS" if hierarchy_ok and distant_ok else "FAIL")
    return matrix, df, stats, hierarchy_ok and distant_ok

def main():
    started = time.time()
    load_base()
    masks = {name: build_concept_masks(name, spec) for name, spec in CONCEPTS.items()}
    results = {"config": {"MODEL": MODEL, "LR": LR, "SEED": SEED, "K_VALUES": K_VALUES, "TRAIN_STEPS": TRAIN_STEPS, "MASK_SOURCE": MASK_SOURCE, "DEVICE": str(DEVICE)}, "runs": {}}
    all_ok = True
    for k in K_VALUES:
        matrix, pairs, stats, ok = evaluate_geometry(masks, k)
        all_ok = all_ok and ok
        results["runs"][str(k)] = {"matrix": matrix.to_dict(), "pairs": pairs.to_dict(orient="records"), "stats": stats.to_dict(orient="records"), "pass": bool(ok)}
    results["final_pass"] = bool(all_ok)
    results["runtime_sec"] = round(time.time() - started, 2)
    with open("tree_geometry_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal Test 1 verdict: {'PASS' if all_ok else 'FAIL'}")
    print(f"Runtime: {results['runtime_sec']} sec")
    print("Saved tree_geometry_results.json")

if __name__ == "__main__":
    main()
