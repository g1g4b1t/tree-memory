# Results

This document summarizes the current deterministic benchmark for TreeMemory.

The benchmark compares:

- `FlatMemory`: one global lexical search over all facts
- `HybridTreeMemory`: beam-routed hierarchical retrieval with local updates and fallback

Benchmark file:

```text
benchmarks/flat_vs_tree_5tasks.py
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

