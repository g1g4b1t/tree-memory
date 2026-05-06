# TreeMemory vs LoRA Results

This document records the first Colab run of the TreeMemory vs LoRA benchmark.

## Setup

Model:

```text
google/flan-t5-small
```

Benchmark:

```text
benchmarks/lora_vs_tree_benchmark.py
```

Notebook:

```text
notebooks/tree_memory_lora_benchmark_colab.ipynb
```

LoRA configuration:

```text
trainable parameters: 344,064
base train time: 4.936 sec
update train time: 1.981 sec
```

## Result

| Strategy | LLM Accuracy | Hit@K | Retrieved Count | Path Precision | Wrong Branch Hits | Context Contamination | AI Context Risk |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_context | 0.031 | 0.000 | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| flat_context | 0.625 | 0.906 | 8.0 | 0.156 | 4.938 | 0.855 | 0.078 |
| gated_tree_context | 0.906 | 0.906 | 1.0 | 1.000 | 0.000 | 0.094 | 0.094 |
| lora_only | 0.094 | 0.000 | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| lora_plus_gated_tree | 0.938 | 0.906 | 1.0 | 1.000 | 0.000 | 0.094 | 0.094 |

## Interpretation

In this run, LoRA alone did not work well as a factual memory mechanism. Even after training, `lora_only` reached only 9.4% accuracy.

External TreeMemory was much stronger:

```text
flat_context        62.5% accuracy
gated_tree_context  90.6% accuracy
lora_only            9.4% accuracy
lora_plus_tree      93.8% accuracy
```

The strongest strategy was `lora_plus_gated_tree`, but most of the gain came from the retrieved TreeMemory context, not from LoRA alone.

## Current Claim

The strongest current claim is:

> In a Colab benchmark with `google/flan-t5-small`, confidence-gated TreeMemory reached 90.6% answer accuracy while LoRA-only factual memorization reached 9.4%. Combining LoRA with TreeMemory reached 93.8%, suggesting LoRA may complement but not replace external factual memory.

## Caveats

This is still an early benchmark:

- LoRA hyperparameters were not heavily tuned.
- The dataset is synthetic.
- The model is small.
- Larger models may memorize facts more effectively.
- LoRA may be better for skills, style, or domain adaptation than for frequently updated factual memory.

The next step is to repeat the benchmark with more LoRA settings and larger query sets.
