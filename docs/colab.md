# Colab Quickstart

TreeMemory does not need a GPU. Colab is useful only as a zero-install demo environment.

## Run in Colab

Open:

```text
notebooks/tree_memory_colab_demo.ipynb
```

Direct Colab link:

```text
https://colab.research.google.com/github/g1g4b1t/tree-memory/blob/main/notebooks/tree_memory_colab_demo.ipynb
```

The notebook will:

1. Clone the repository.
2. Install `requirements.txt`.
3. Run the basic demo.
4. Run the full validation suite.
5. Display the scaled benchmark summary.
6. Let the user query TreeMemory directly.

## Expected Result

The validation cell should end with:

```text
All checks passed.
```

The scaled benchmark should include:

```text
Final Scaled Memory Benchmark verdict: PASS
```

## CPU or GPU?

Use CPU. The current TreeMemory prototype is a retrieval and memory-structure benchmark, not a neural network training job.
