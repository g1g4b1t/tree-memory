# TreeMemory vs LoRA Benchmark

This optional benchmark compares two ways of giving an AI system new factual memory.

## Compared Approaches

TreeMemory:

```text
facts stay outside the model
updates are local memory edits
the LLM receives retrieved context
```

LoRA:

```text
facts are trained into adapter weights
updates require more training
the LLM answers without retrieval context
```

## Colab

Use:

```text
notebooks/tree_memory_lora_benchmark_colab.ipynb
```

Direct link:

```text
https://colab.research.google.com/github/g1g4b1t/tree-memory/blob/main/notebooks/tree_memory_lora_benchmark_colab.ipynb
```

Recommended runtime:

```text
T4 GPU
```

## Local Run

Install optional dependencies:

```bash
pip install -r requirements-lora.txt
```

Then install PyTorch for your platform if it is not already installed. In Colab, PyTorch is already included.

Some Colab runtimes include an old `torchao` package that is incompatible with recent `peft`. The notebook removes `torchao` automatically because this benchmark does not need it.

Run:

```bash
python benchmarks/lora_vs_tree_benchmark.py --model google/flan-t5-small
```

## Metrics

The benchmark measures:

- LLM answer accuracy
- retrieval context contamination
- AI context risk
- LoRA base training time
- LoRA update training time
- trainable LoRA parameters

## Research Question

The benchmark tests this claim:

> For frequently updated factual memory, external TreeMemory should provide stronger update locality and lower operational cost than LoRA fine-tuning while preserving competitive answer accuracy.

LoRA may still be better for learning style, skills, and broad domain behavior. TreeMemory is intended for inspectable, editable factual memory.

Latest recorded result:

```text
gated_tree_context    accuracy 0.906
lora_only             accuracy 0.094
lora_plus_gated_tree  accuracy 0.938
```

See:

```text
docs/lora_results.md
```
