# TreeMemory Hypothesis

## Problem

AI assistants need long-term memory, but flat memory can mix unrelated facts that share surface words.

Examples:

- Michelin tires vs Michelin stars
- Python code vs python snake
- Apple company vs apple fruit
- Jaguar car vs jaguar animal

If all facts are retrieved from one flat pool, the correct fact may be present, but the context can also contain misleading nearby facts. A language model may then answer from the wrong branch.

## Hypothesis

External memory should be organized hierarchically:

```text
root/artifacts/vehicles/car_tires
root/artifacts/vehicles/car_engine
root/living/reptiles/python_snake
root/artifacts/computing/python_code
root/culture/food/restaurants
```

The main claim is not that existing LLM weights already contain a clean memory tree. The practical claim is:

> A designed hierarchical external memory can reduce context contamination and make updates local.

## Predictions

Compared with flat memory, TreeMemory should:

1. Retrieve fewer facts from unrelated branches.
2. Preserve nearby facts during local updates.
3. Reduce stale conflict leakage.
4. Produce shorter and cleaner context.
5. Provide an inspectable retrieval trace.

## Current Result

The current best prototype is `HybridTreeMemory`:

- beam route to multiple candidate branches
- retrieve local facts
- add a small global fallback
- rerank candidates
- return a compact context

In the current 5-task synthetic benchmark, TreeMemory matches flat memory's overall top-1 accuracy while improving path precision and reducing wrong-branch retrieval.

## Limitations

This is still early:

- benchmark facts are synthetic
- path aliases are manually written
- routing is lexical rather than embedding-based
- no LLM currently extracts facts or writes paths
- JSON persistence is only a prototype backend

## Next Research Step

The next step is to connect this memory engine to an LLM:

```text
user input -> fact extraction -> path routing -> TreeMemory update
user question -> TreeMemory retrieval -> LLM answer with cited memory context
```

