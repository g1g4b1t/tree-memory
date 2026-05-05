import math, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
MODEL_NAME, DEVICE = "gpt2", torch.device("cpu")
torch.manual_seed(0)
# Load GPT-2 small and keep one AdamW optimizer for the toy run.
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)
stacks: dict[str, torch.Tensor] = {}
last_mask: torch.Tensor | None = None

def _loss_for(text: str) -> torch.Tensor:
    # Tokenize text and use the same ids as causal LM labels.
    batch = tokenizer(text, return_tensors="pt").to(DEVICE)
    return model(**batch, labels=batch["input_ids"]).loss

def _flat_abs_grads() -> torch.Tensor:
    # Concatenate absolute gradients in model parameter order.
    parts = []
    for p in model.parameters():
        if p.requires_grad:
            g = p.grad
            parts.append(torch.zeros(p.numel(), device=DEVICE) if g is None else g.detach().abs().reshape(-1))
    return torch.cat(parts)

def jaccard(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    # Return intersection / union for two boolean masks.
    m1, m2 = mask1.bool().reshape(-1), mask2.bool().reshape(-1)
    inter = torch.logical_and(m1, m2).sum().item()
    union = torch.logical_or(m1, m2).sum().item()
    return inter / union if union else 1.0

def build_stack(text_prompt: str, stack_name: str, k: float = 0.03):
    global last_mask
    # Backprop once to discover which weights this prompt activates.
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss = _loss_for(text_prompt)
    loss.backward(); grads = _flat_abs_grads()
    # Store a 1-bit mask for the top-k percent of gradient magnitudes.
    keep = max(1, int(grads.numel() * k))
    _, idx = torch.topk(grads, keep)
    mask = torch.zeros(grads.numel(), dtype=torch.bool, device=DEVICE); mask[idx] = True
    stacks[stack_name] = mask.detach().cpu()
    score = float("nan") if last_mask is None else jaccard(stacks[stack_name], last_mask)
    pct = stacks[stack_name].float().mean().item() * 100.0
    print(f"Build prompt: {text_prompt!r}, loss: {loss.item():.4f}")
    print(f"Stack {stack_name}: {pct:.2f}% params, Jaccard vs last: {score:.4f}")
    last_mask = stacks[stack_name]
    optimizer.zero_grad(set_to_none=True)

def train_with_stacks(text: str, active_stacks: list[str]):
    # Train normally, except protected stack bits get zeroed gradients.
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss = _loss_for(text)
    loss.backward()
    combined = None
    if active_stacks:
        missing = [name for name in active_stacks if name not in stacks]
        if missing: raise KeyError(f"Unknown stack(s): {missing}")
        combined = torch.zeros_like(stacks[active_stacks[0]])
        for name in active_stacks:
            combined |= stacks[name]
    protected_pct = 0.0 if combined is None else combined.float().mean().item() * 100.0
    if combined is not None:
        offset = 0
        for p in model.parameters():
            if not p.requires_grad: continue
            n = p.numel()
            mask = combined[offset : offset + n].view_as(p)
            if p.grad is not None: p.grad.data.masked_fill_(mask.to(p.grad.device), 0.0)
            state = optimizer.state.get(p, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state: state[key].masked_fill_(mask.to(state[key].device), 0.0)
            offset += n
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print(f"Train text: {text!r}, active stacks: {active_stacks}, loss: {loss.item():.4f}")
    print(f"Protected during step: {protected_pct:.2f}% params")

def eval_perplexity(text: str) -> float:
    # Compute exp(loss) without gradient updates.
    model.eval()
    with torch.no_grad():
        loss = _loss_for(text)
    ppl = math.exp(loss.item())
    print(f"Eval text: {text!r}, loss: {loss.item():.4f}, ppl: {ppl:.4f}")
    return ppl

if __name__ == "__main__":
    print(f"Model: {MODEL_NAME}, trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    build_stack("The car has rubber tires for grip", "car_tires")
    build_stack("The car engine uses fuel combustion", "car_engine")
    sim = jaccard(stacks["car_tires"], stacks["car_engine"])
    print(f"Jaccard similarity car_tires vs car_engine: {sim:.4f}"); print("Concepts are separable." if sim < 0.5 else "Concepts are not clearly separable.")
    train_with_stacks("Michelin produces car tires", active_stacks=[]); ppl1 = eval_perplexity("Michelin produces")
    train_with_stacks("Tesla car engine is electric", active_stacks=["car_tires"]); ppl2 = eval_perplexity("Michelin produces")
    delta = ppl2 - ppl1
    print(f"Forgetting: {ppl1:.4f} -> {ppl2:.4f}, delta = {delta:.4f}")
    print("Success: delta < 20% of ppl1" if delta < 0.2 * ppl1 else "Failure: delta >= 20% of ppl1")
