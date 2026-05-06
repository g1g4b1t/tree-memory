# Results

This document summarizes the current deterministic benchmark for TreeMemory.

The benchmark compares:

- `FlatMemory`: one global lexical search over all facts
- `HybridTreeMemory`: beam-routed hierarchical retrieval with local updates and fallback

Benchmark file:

```text
benchmarks/flat_vs_tree_5tasks.py
```

Larger benchmark file:

```text
benchmarks/scaled_memory_benchmark.py
```

Run:

```bash
python benchmarks/flat_vs_tree_5tasks.py
```

## Benchmark Tasks

The benchmark contains five task groups.

| Task | Goal |
|---|---|
| Ambiguity | Distinguish entities that share names, such as `Python code` vs `python snake`. |
| Noisy queries | Handle queries that explicitly reject a wrong meaning, such as `Mercury in astronomy, not chemistry`. |
| Local update | Replace one fact in one branch while keeping nearby branches stable. |
| Conflict isolation | Check that updates do not leak into semantically nearby but distinct branches. |
| Context efficiency | Measure how many retrieved facts are needed before the answer appears. |

## Current Overall Result

Latest local run:

```text
Final Best Tree vs Flat verdict: PASS
```

Overall metrics:

| Metric | FlatMemory | HybridTreeMemory | Better |
|---|---:|---:|---|
| Top-1 accuracy | 0.852 | 0.852 | Tie |
| Hit@K | 0.963 | 0.926 | FlatMemory |
| Avg context items to answer | 1.333 | 1.556 | FlatMemory |
| Path precision | 0.272 | 0.840 | HybridTreeMemory |
| Wrong path hits | 4.370 | 0.370 | HybridTreeMemory |
| Wrong branch hits | 3.259 | 0.370 | HybridTreeMemory |
| Conflict hits | 0.111 | 0.000 | HybridTreeMemory |

## Interpretation

HybridTreeMemory does not currently beat FlatMemory on raw top-1 accuracy. It ties overall top-1 accuracy in this benchmark.

Its advantage is retrieval cleanliness:

- much higher path precision
- far fewer wrong-path facts
- far fewer wrong-branch facts
- no stale conflict leakage after local updates

This matters because an LLM does not only need the correct answer to appear somewhere in the retrieved context. It also needs the surrounding context not to contain misleading facts.

FlatMemory often retrieves the right fact together with many unrelated facts. HybridTreeMemory returns a more local and inspectable context.

## Example: Local Update

Initial fact:

```text
artifacts/vehicles/car_tires:
Michelin produces premium car tires.
```

Update:

```text
artifacts/vehicles/car_tires:
Bridgestone produces premium car tires.
```

Preserved neighboring branch:

```text
culture/food/restaurants:
Michelin stars are restaurant awards for excellent dining.
```

Expected behavior:

```text
Who produces premium car tires now? -> Bridgestone
What are Michelin stars, not tires? -> restaurant awards
```

TreeMemory passes this example.

## Scaled Benchmark

The scaled benchmark tests a larger synthetic memory:

```text
Concepts: 37
Base facts: 111
Updates: 17
Queries: 256
```

It compares:

- `flat_append`: flat memory that appends updates
- `flat_replace`: stronger flat memory with local replacement
- `hard_tree`: strict top-1 tree routing
- `hybrid_tree`: beam tree routing with compact fallback
- `gated_hybrid_tree`: strict routing when confidence is high, fallback only when confidence is low

Latest result:

```text
Final Scaled Memory Benchmark verdict: PASS
```

Overall summary:

| Memory | Top-1 | Hit@K | Path Precision | Wrong Branch Hits | Context Contamination | AI Context Risk |
|---|---:|---:|---:|---:|---:|---:|
| flat_append | 0.934 | 1.000 | 0.241 | 4.230 | 0.767 | 0.106 |
| flat_replace | 0.934 | 1.000 | 0.233 | 4.281 | 0.767 | 0.089 |
| gated_hybrid_tree | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| hard_tree | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| hybrid_tree | 0.957 | 1.000 | 0.962 | 0.078 | 0.038 | 0.028 |

Interpretation:

On this generated dataset, explicit domain hints make strict tree routing very strong. Plain hybrid tree routing is less clean because fallback can reintroduce unrelated context. Gated hybrid routing fixes that by using fallback only when routing confidence is low. This supports the practical claim that hierarchical memory can reduce context contamination and wrong-branch retrieval.

## Real LLM Benchmark

The first Colab run used:

```text
google/flan-t5-small
```

Summary:

| Memory | LLM Accuracy | Context Contamination | AI Context Risk |
|---|---:|---:|---:|
| no_context | 0.00 | 0.000 | 0.000 |
| flat_replace | 0.70 | 0.875 | 0.106 |
| hybrid_tree | 0.70 | 0.492 | 0.358 |
| hard_tree | 0.95 | 0.000 | 0.000 |
| gated_hybrid_tree | 0.95 | 0.000 | 0.000 |

Interpretation:

Flat memory gave the model the right answer somewhere in context, but with substantial wrong-branch contamination. Confidence-gated TreeMemory returned clean context and improved LLM answer accuracy from 70% to 95% in this run.

For details, see:

```text
docs/llm_results.md
```

## LoRA Comparison

The first Colab LoRA comparison used:

```text
google/flan-t5-small
```

Summary:

| Strategy | LLM Accuracy | Context Contamination | Trainable LoRA Params |
|---|---:|---:|---:|
| no_context | 0.031 | 0.000 | 0 |
| flat_context | 0.625 | 0.855 | 0 |
| gated_tree_context | 0.906 | 0.094 | 0 |
| lora_only | 0.094 | 0.000 | 344,064 |
| lora_plus_gated_tree | 0.938 | 0.094 | 344,064 |

Interpretation:

LoRA alone did not function well as factual memory in this small run. External TreeMemory was much stronger for answer accuracy, and combining LoRA with TreeMemory performed best.

For details, see:

```text
docs/lora_results.md
```

## What This Result Supports

The current result supports a cautious claim:

> Hierarchical external memory can reduce context contamination and make fact updates more local while preserving competitive answer retrieval.

It does not prove that TreeMemory is generally better than all flat retrieval systems. The benchmark is still synthetic and small.

## Failure Modes Found Earlier

Earlier versions used hard top-1 tree routing:

```text
query -> choose one branch -> retrieve only from that branch
```

That failed because the system lost recall when the router picked the wrong branch.

The current version uses:

```text
query -> beam route to several branches -> local candidates + fallback -> rerank
```

This hybrid design keeps the benefits of tree locality while reducing the risk of hard-routing failures.

## Next Results To Collect

The next benchmark should test:

1. Larger memory size, for example 500-5,000 facts.
2. Embedding-based routing instead of lexical routing.
3. LLM-generated answers from retrieved context.
4. Human-written ambiguous queries.
5. Realistic personal assistant memories.
6. Ablations:
   - no tree
   - hard tree
   - beam tree
   - beam tree + fallback
   - beam tree + fallback + local update
