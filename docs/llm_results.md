# Real LLM Benchmark Results

This document records the first Colab run of the real LLM context benchmark.

## Setup

Model:

```text
google/flan-t5-small
```

Benchmark:

```text
benchmarks/llm_context_benchmark.py
```

Notebook:

```text
notebooks/tree_memory_llm_benchmark_colab.ipynb
```

The benchmark compares LLM answer quality when the model receives context from different memory strategies.

## Result

| Memory | LLM Accuracy | Hit@K | Retrieved Count | Path Precision | Wrong Branch Hits | Context Contamination | AI Context Risk |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_context | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| flat_replace | 0.70 | 1.00 | 8.00 | 0.125 | 6.00 | 0.875 | 0.106 |
| hybrid_tree | 0.70 | 1.00 | 2.30 | 0.508 | 1.00 | 0.492 | 0.358 |
| hard_tree | 0.95 | 1.00 | 1.00 | 1.000 | 0.00 | 0.000 | 0.000 |
| gated_hybrid_tree | 0.95 | 1.00 | 1.00 | 1.000 | 0.00 | 0.000 | 0.000 |

## Interpretation

The result supports the main TreeMemory claim:

> Cleaner hierarchical memory context can improve downstream LLM answers.

In this run, flat memory retrieved the correct answer somewhere in the context, but it also retrieved many wrong-branch facts. The LLM reached 70% accuracy with that noisy context.

Strict tree routing and confidence-gated tree routing returned a clean single-branch context. Both reached 95% accuracy with zero measured context contamination.

The plain hybrid tree did not improve over flat memory in this run because fallback retrieval reintroduced distracting context. The confidence gate fixes this by using fallback only when the router is not confident.

## Current Claim

The strongest current claim is:

> In a real LLM benchmark with `google/flan-t5-small`, confidence-gated TreeMemory improved answer accuracy from 70% to 95% compared with flat retrieval while reducing measured context contamination from 0.875 to 0.000.

## Caveats

This is still an early result:

- the dataset is synthetic
- the run used a small model
- the run used a limited query sample
- results may change with larger models and human-written queries

The next research step is to repeat this benchmark with more queries, stronger models, and less explicit routing hints.
