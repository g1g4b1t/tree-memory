# Architecture

TreeMemory is a small external memory engine for AI assistants. It separates memory storage from the language model.

The language model should not be forced to keep all long-term facts inside its weights. Instead, TreeMemory stores facts in an inspectable hierarchy and returns clean context for the model.

## Data Model

Each memory is an atomic fact:

```text
Fact
+-- id
+-- path
+-- slot
+-- text
+-- answer
+-- tags
+-- source
+-- confidence
+-- version
+-- active
+-- supersedes
```

Example:

```text
path:   artifacts/vehicles/car_tires
slot:   car_tires.maker
text:   Bridgestone produces premium car tires in the updated memory.
answer: Bridgestone
```

## Paths

Paths are semantic locations:

```text
artifacts/vehicles/car_tires
culture/food/restaurants
artifacts/computing/python_code
living/reptiles/python_snake
```

This lets nearby facts remain separate:

```text
Michelin tires  -> artifacts/vehicles/car_tires
Michelin stars  -> culture/food/restaurants
```

## Slots

A slot is the local attribute being remembered.

Examples:

```text
car_tires.maker
python.lists
python_snake.skin
apple_company.products
apple_fruit.color
```

Updates are local to:

```text
path + slot
```

If a new fact updates the same path and slot, the old fact is marked inactive and the new fact becomes the active version.

## Retrieval Pipeline

TreeMemory uses hybrid retrieval:

```text
query
  -> tokenize
  -> beam route to candidate paths
  -> retrieve local facts from candidate paths
  -> add small global fallback
  -> rerank facts
  -> return compact context
```

Hard top-1 routing was tested and failed because one bad route can lose the correct answer. Beam routing keeps multiple branch hypotheses alive.

## Routing

Routing scores paths using:

- path aliases
- words from active facts
- words from parent nodes
- matched query terms
- simple negation handling such as `not tires`

Example:

```text
query: What are Michelin stars, not tires?

route 1: culture/food/restaurants
route 2: artifacts/vehicles/car_tires
```

The phrase `not tires` penalizes the tire branch.

## Reranking

Retrieved facts are reranked using:

- lexical overlap
- route rank
- fact version
- confidence

The output is not only an answer. It includes the memory context used to answer.

## Explainability

`explain_retrieval(query)` returns:

- candidate routes
- matched terms
- retrieved facts
- scores
- source fact IDs
- final answer

This is important for auditable AI memory. A user or developer can inspect why a memory was used.

## Local Update Example

Initial memory:

```text
artifacts/vehicles/car_tires:
Michelin produces premium car tires.
```

Update:

```text
artifacts/vehicles/car_tires:
Bridgestone produces premium car tires.
```

Nearby branch remains unchanged:

```text
culture/food/restaurants:
Michelin stars are restaurant awards.
```

Expected answers:

```text
Who produces premium car tires now? -> Bridgestone
What are Michelin stars, not tires? -> restaurant awards
```

## Current Limitations

- Lexical routing only.
- Manual path aliases.
- No embedding search yet.
- No automatic fact extraction.
- JSON persistence only.
- No branch summary generation.

## Intended LLM Integration

Future system:

```text
user message
  -> LLM extracts facts
  -> LLM or embedding router chooses path
  -> TreeMemory stores or updates facts

user question
  -> TreeMemory retrieves clean context
  -> LLM answers with cited memory facts
```

TreeMemory is the memory layer, not the whole assistant.

