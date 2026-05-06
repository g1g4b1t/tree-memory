# Real LLM Context Benchmark

This optional benchmark connects TreeMemory to a real language model.

The question changes from:

```text
Does TreeMemory retrieve cleaner context?
```

to:

```text
Does a real LLM answer better when it receives cleaner TreeMemory context?
```

## Colab

Use:

```text
notebooks/tree_memory_llm_benchmark_colab.ipynb
```

Direct link:

```text
https://colab.research.google.com/github/g1g4b1t/tree-memory/blob/main/notebooks/tree_memory_llm_benchmark_colab.ipynb
```

Recommended runtime:

```text
T4 GPU
```

CPU works with small settings, but it is slower.

## Local Run

Install optional dependencies:

```bash
pip install -r requirements-llm.txt
```

Then install PyTorch for your platform if it is not already installed. In Colab, PyTorch is already included, so the notebook does not reinstall it.

Run a quick local test:

```bash
python benchmarks/llm_context_benchmark.py --model google/flan-t5-small --max-queries 20
```

For a stronger but slower run, use:

```bash
python benchmarks/llm_context_benchmark.py --model google/flan-t5-base --max-queries 40
```

The benchmark compares:

- `no_context`
- `flat_replace`
- `hard_tree`
- `hybrid_tree`
- `gated_hybrid_tree`

`gated_hybrid_tree` uses the clean single-branch route when the router is confident and only falls back to beam/fallback retrieval when confidence is low.

It measures:

- LLM answer accuracy
- Hit@K of the retrieval context
- Path precision
- Wrong-branch hits
- Context contamination
- AI context risk

## Interpretation

This benchmark is intentionally not part of the main CI suite. Unlike the deterministic memory benchmarks, it depends on model behavior, hardware, and installed ML library versions.

The goal is to test the practical claim:

> Cleaner hierarchical memory context should make downstream LLM answers less likely to mix unrelated meanings.

Latest recorded result:

```text
flat_replace       accuracy 0.70, context contamination 0.875
gated_hybrid_tree  accuracy 0.95, context contamination 0.000
```

See:

```text
docs/llm_results.md
```
