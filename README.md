# TreeMemory

TreeMemory is an experimental external memory system for AI assistants. The core idea is simple:

> Facts should not be stored as one flat pile. They should be organized into semantic branches, so updates and retrieval stay local.

This project explores whether a hierarchical memory can reduce context contamination, wrong-branch retrieval, and stale fact conflicts compared with flat lexical memory.

## Motivation

Long-term AI memory has a practical problem: similar words can refer to unrelated concepts.

Examples:

- `Michelin tires` vs `Michelin stars`
- `Python code` vs `python snake`
- `Apple company` vs `apple fruit`
- `Jaguar car` vs `jaguar animal`
- `Mercury planet` vs `mercury element`

A flat memory search can retrieve the right answer, but often includes many unrelated facts in the context. That is dangerous for an AI assistant because the language model may mix branches and answer from the wrong memory.

TreeMemory tests a different approach:

```text
root
+-- artifacts
|   +-- vehicles
|   |   +-- car_tires
|   |   +-- car_engine
|   +-- computing
|       +-- python_code
+-- living
|   +-- reptiles
|   |   +-- python_snake
|   +-- mammals
|       +-- jaguar_animal
+-- culture
    +-- food
        +-- restaurants
```

Each fact is stored in a semantic path. Updates replace only the local fact in the same path and slot.

## Hypothesis

External hierarchical memory should improve AI memory quality by:

1. Reducing retrieval from unrelated branches.
2. Localizing updates to the correct branch.
3. Preserving nearby but distinct facts.
4. Producing cleaner context for downstream LLM answers.
5. Making memory behavior easier to inspect and debug.

This project does not claim that LLM weights naturally store knowledge as a clean tree. Earlier experiments in this folder tested gradient masks in GPT-2. The stronger practical direction is external TreeMemory: a designed memory layer that an AI system can retrieve from.

## Current Design

The main implementation is:

```text
tree_memory_engine.py
```

It implements a hybrid tree retrieval system:

- semantic paths such as `artifacts/vehicles/car_tires`
- local fact updates by `path + slot`
- beam routing to multiple likely branches
- global fallback retrieval
- final reranking
- compact context output
- retrieval explanations
- JSON save/load

The important design choice is that routing is not hard top-1 routing. A naive hard tree failed because if it chose the wrong branch, it lost recall. TreeMemory uses beam routing plus fallback.

## Quickstart

Run the demo:

```bash
python examples/demo.py
```

Run in Google Colab:

```text
notebooks/tree_memory_colab_demo.ipynb
```

Direct Colab link:

```text
https://colab.research.google.com/github/g1g4b1t/tree-memory/blob/main/notebooks/tree_memory_colab_demo.ipynb
```

Colab is optional. TreeMemory runs on CPU, but the notebook is useful for a zero-install demo. See:

```text
docs/colab.md
```

Run the interactive CLI:

```bash
python examples/interactive_cli.py
```

Example CLI commands:

```text
ask Who produces premium car tires now?
update
ask What are Michelin stars, not tires?
explain What are Michelin stars, not tires?
save my_memory.json
quit
```

Expected behavior:

```text
Before update:
Who produces premium car tires? -> Michelin
What are Michelin stars? -> restaurant awards

After local update:
Who produces premium car tires now? -> Bridgestone
What are Michelin stars, not tires? -> restaurant awards
What power source do Tesla car engines use after the tire update? -> electric motors
```

This shows a local update:

```text
artifacts/vehicles/car_tires:
Michelin -> Bridgestone
```

without corrupting:

```text
culture/food/restaurants:
Michelin stars -> restaurant awards
```

## Basic Usage

```python
from tree_memory_engine import TreeMemory

memory = TreeMemory()

memory.add_alias(
    "artifacts/vehicles/car_tires",
    "car tires rubber road grip michelin bridgestone vehicle maker",
)

memory.add_fact(
    path="artifacts/vehicles/car_tires",
    text="Michelin produces premium car tires.",
    slot="car_tires.maker",
    answer="Michelin",
    tags="vehicle tires maker",
)

print(memory.answer("Who produces premium car tires?")["answer"])

memory.update_fact(
    path="artifacts/vehicles/car_tires",
    slot="car_tires.maker",
    text="Bridgestone produces premium car tires in the updated memory.",
    answer="Bridgestone",
    tags="vehicle tires maker update",
)

print(memory.answer("Who produces premium car tires now?")["answer"])
```

## Retrieval Explanation

TreeMemory can explain its retrieval:

```python
explanation = memory.explain_retrieval("What are Michelin stars, not tires?")
```

The explanation includes:

- routed branches
- matched terms
- retrieved facts
- scores
- source fact IDs
- final answer

This is useful for debugging memory behavior and for showing why the assistant answered a certain way.

## Benchmark

The current comparison benchmark is:

```text
benchmarks/flat_vs_tree_5tasks.py
```

It compares:

- `FlatMemory`: one global lexical search over all facts
- `HybridTreeMemory`: beam-routed tree retrieval with local updates and fallback

The benchmark includes five task types:

1. Ambiguous entities:
   `python code` vs `python snake`, `apple company` vs `apple fruit`
2. Noisy queries:
   questions that explicitly reject a wrong meaning
3. Local updates:
   replacing facts inside one branch
4. Conflict isolation:
   checking that nearby branches are not corrupted
5. Context efficiency:
   measuring how many retrieved facts are needed to reach the answer

Run:

```bash
python benchmarks/flat_vs_tree_5tasks.py
```

Current local result:

```text
Final Best Tree vs Flat verdict: PASS
```

The larger scaled benchmark is:

```text
benchmarks/scaled_memory_benchmark.py
```

It compares four retrieval strategies:

- `flat_append`: flat memory where updates are appended and stale facts remain retrievable
- `flat_replace`: stronger flat memory with exact path+slot replacement
- `hard_tree`: strict top-1 branch routing
- `hybrid_tree`: beam tree routing with compact fallback

Current scaled result:

```text
Final Scaled Memory Benchmark verdict: PASS
```

Latest scaled summary:

```text
Concepts: 37
Base facts: 111
Updates: 17
Queries: 256

Overall top-1 accuracy:
flat_replace 0.934
hard_tree    1.000
hybrid_tree  0.957

Context contamination:
flat_replace 0.767
hard_tree    0.000
hybrid_tree  0.038

AI context risk:
flat_replace 0.089
hard_tree    0.000
hybrid_tree  0.028
```

This result supports the main claim: hierarchical memory can keep retrieval context much cleaner than flat memory while preserving strong answer retrieval.

Summary from the latest run:

```text
Overall top1 accuracy:
flat        0.852
hybrid_tree 0.852

Path precision:
flat        0.272
hybrid_tree 0.840

Wrong branch hits:
flat        3.259
hybrid_tree 0.370

Conflict hits:
flat        0.111
hybrid_tree 0.000
```

Interpretation:

TreeMemory did not improve overall top-1 accuracy in this benchmark, but it produced much cleaner retrieval. It reduced wrong-branch facts and eliminated update conflicts while keeping answer accuracy competitive.

For a fuller write-up, see:

```text
docs/results.md
```

## Why This Matters

For an AI assistant, retrieval quality is not only about whether the answer appears somewhere in the context. It is also about what else appears with it.

Flat memory can return the correct answer alongside many misleading facts. TreeMemory aims to return a smaller, cleaner, more local context.

This matters for:

- long-term assistant memory
- personal knowledge bases
- RAG systems
- local fact updates
- auditable memory
- reducing hallucination from contaminated context

## Earlier Experiments

This repository also contains earlier exploratory experiments under:

```text
experiments/gpt2_gradient_masks/
```

Those files tested whether GPT-2 gradient masks behave like semantic memory branches. The results were mixed:

- gradient footprints show some semantic structure
- naive parameter freezing does not scale reliably
- conflict-aware gradient masking works better

The current direction is external TreeMemory, because it is more practical and controllable than trying to force long-term facts into model weights.

Older external-memory iterations are kept under:

```text
experiments/external_memory_iterations/
```

Benchmark and demo outputs are stored under:

```text
artifacts/results/
artifacts/demo/
```

## Repository Layout

```text
tree_memory_engine.py              Core TreeMemory implementation
examples/demo.py                   Small runnable demo
examples/interactive_cli.py        Tiny manual testing CLI
benchmarks/flat_vs_tree_5tasks.py  FlatMemory vs HybridTree benchmark
benchmarks/_flat_vs_tree_5tasks_impl.py
                                    Benchmark implementation
benchmarks/scaled_memory_benchmark.py
                                    Larger flat vs hard-tree vs hybrid-tree benchmark
docs/hypothesis.md                 Research hypothesis and predictions
docs/architecture.md               System architecture
docs/results.md                    Benchmark summary and interpretation
docs/github_setup.md               Publishing notes for GitHub
docs/colab.md                      Google Colab quickstart
experiments/                       Archived exploratory experiments
artifacts/                         Local demo and benchmark outputs
scripts/validate.py                Full local validation suite
notebooks/tree_memory_colab_demo.ipynb
                                    Zero-install Colab demo
tests/test_tree_memory_engine.py   Basic regression tests
.github/workflows/ci.yml           GitHub Actions validation
LICENSE                            MIT License
README.md                          Project overview
requirements.txt                   Minimal benchmark dependency
```

Run tests:

```bash
python -m unittest discover -s tests
```

Run the full local validation suite:

```bash
python scripts/validate.py
```

This checks syntax, unit tests, demo behavior, CLI smoke behavior, and the 5-task benchmark.

GitHub Actions runs the same validation suite on Ubuntu and Windows. See:

```text
.github/workflows/ci.yml
docs/github_setup.md
```

## Limitations

This is an early prototype.

Known limitations:

- routing is still lexical, not embedding-based
- path aliases are manually provided
- no LLM is used yet for fact extraction or answer generation
- benchmark facts are synthetic
- no persistence backend beyond JSON
- no conflict resolution UI
- no automatic tree growth

These limitations are intentional at this stage. The goal is to isolate and test the memory structure before adding more moving parts.

## Roadmap

Near-term:

- Add automatic fact extraction from text.
- Add automatic path routing using an LLM or embeddings.
- Add larger benchmark datasets.
- Add benchmark scripts for ablations.

Medium-term:

- SQLite storage.
- Version history per branch.
- Conflict reports.
- Memory deletion and forgetting.
- Branch summaries.
- LLM answer generation from retrieved context.

Research direction:

- Compare flat RAG vs TreeMemory RAG.
- Measure context contamination.
- Test on larger ambiguous datasets.
- Add human-readable retrieval traces.

## Project Status

Status: experimental prototype.

TreeMemory is not a solved memory system. It is a small research project exploring whether hierarchical external memory can make AI retrieval cleaner, more local, and easier to update.
