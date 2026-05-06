# Robust Query Benchmark Results

This benchmark tests TreeMemory with more natural and less explicit queries.

Unlike the scaled benchmark, many questions avoid simple phrases such as:

```text
For python in reptile...
```

and instead use forms such as:

```text
What does the snake called python shed?
Which Mercury is closest to the Sun?
Not the fruit: what does Apple make?
After the package update, what installs Python packages?
```

## Setup

Benchmark:

```text
benchmarks/robust_query_benchmark.py
```

Query count:

```text
59
```

Compared strategies:

- `flat_replace`
- `hard_tree`
- `hybrid_tree`
- `gated_hybrid_tree`

## Result

| Memory | Top-1 Accuracy | Hit@K | Retrieved Count | Path Precision | Wrong Branch Hits | Context Contamination | AI Context Risk |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat_replace | 0.746 | 0.983 | 8.000 | 0.182 | 5.610 | 0.818 | 0.133 |
| hard_tree | 0.746 | 0.831 | 1.424 | 0.847 | 0.153 | 0.153 | 0.136 |
| hybrid_tree | 0.797 | 0.983 | 1.831 | 0.855 | 0.288 | 0.145 | 0.103 |
| gated_hybrid_tree | 0.797 | 0.983 | 1.780 | 0.869 | 0.254 | 0.131 | 0.093 |

## Interpretation

This is a harder test. The first version exposed a slot-selection failure mode: TreeMemory often found the correct branch but selected the wrong attribute inside that branch. Slot-aware reranking was added to address this.

After slot-aware reranking, TreeMemory improves top-1 accuracy while keeping the memory hygiene result strong:

```text
flat_replace top1:       0.746
gated_hybrid_tree top1:  0.797

flat_replace contamination:       0.818
gated_hybrid_tree contamination:  0.131

flat_replace AI context risk:      0.133
gated_hybrid_tree AI context risk: 0.093
```

The cautious conclusion is:

> On naturalized queries, confidence-gated TreeMemory with slot-aware reranking improves top-1 retrieval accuracy and substantially reduces context contamination compared with flat retrieval.

The remaining errors suggest a next step: improve slot aliases and eventually replace hand-written slot rules with a learned or embedding-based reranker.
