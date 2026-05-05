# GitHub Setup

This project is ready to publish as a small research prototype.

## Before Upload

Run the full validation suite:

```bash
python scripts/validate.py
```

Expected final line:

```text
All checks passed.
```

## Suggested Repository Description

```text
Experimental hierarchical external memory for AI assistants, designed to reduce context contamination and localize memory updates.
```

## Suggested Topics

```text
ai-memory
rag
continual-learning
semantic-memory
retrieval
python
research-prototype
```

## First Commit Message

```text
Initial TreeMemory research prototype
```

## After Upload

GitHub Actions will run:

```bash
python scripts/validate.py
```

on both Ubuntu and Windows. A green CI check means the demo, tests, CLI smoke test, and benchmark all pass in a clean environment.
