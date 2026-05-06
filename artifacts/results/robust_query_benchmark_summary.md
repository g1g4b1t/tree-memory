# Robust Query Benchmark Summary

## Overall

| memory | top1_correct | hit_at_k | retrieved_count | path_precision | wrong_branch_hits | context_contamination | ai_context_risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat_replace | 0.746 | 0.983 | 8.000 | 0.182 | 5.610 | 0.818 | 0.133 |
| gated_hybrid_tree | 0.797 | 0.983 | 1.780 | 0.869 | 0.254 | 0.131 | 0.093 |
| hard_tree | 0.746 | 0.831 | 1.424 | 0.847 | 0.153 | 0.153 | 0.136 |
| hybrid_tree | 0.797 | 0.983 | 1.831 | 0.855 | 0.288 | 0.145 | 0.103 |

## By Task

| task | memory | top1_correct | hit_at_k | retrieved_count | path_precision | wrong_branch_hits | context_contamination | ai_context_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| implicit_natural | flat_replace | 0.676 | 0.971 | 8.000 | 0.184 | 5.412 | 0.816 | 0.165 |
| implicit_natural | gated_hybrid_tree | 0.735 | 0.971 | 1.676 | 0.904 | 0.235 | 0.096 | 0.076 |
| implicit_natural | hard_tree | 0.735 | 0.853 | 1.441 | 0.882 | 0.147 | 0.118 | 0.088 |
| implicit_natural | hybrid_tree | 0.735 | 0.971 | 1.765 | 0.880 | 0.294 | 0.120 | 0.093 |
| natural_conflict | flat_replace | 0.600 | 1.000 | 8.000 | 0.175 | 6.000 | 0.825 | 0.175 |
| natural_conflict | gated_hybrid_tree | 0.800 | 1.000 | 2.000 | 0.733 | 0.600 | 0.267 | 0.267 |
| natural_conflict | hard_tree | 0.800 | 0.800 | 1.400 | 0.800 | 0.200 | 0.200 | 0.200 |
| natural_conflict | hybrid_tree | 0.800 | 1.000 | 2.000 | 0.733 | 0.600 | 0.267 | 0.267 |
| natural_update | flat_replace | 1.000 | 1.000 | 8.000 | 0.125 | 6.333 | 0.875 | 0.000 |
| natural_update | gated_hybrid_tree | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| natural_update | hard_tree | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| natural_update | hybrid_tree | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| negated_natural | flat_replace | 0.882 | 1.000 | 8.000 | 0.191 | 5.765 | 0.809 | 0.081 |
| negated_natural | gated_hybrid_tree | 0.882 | 1.000 | 2.059 | 0.814 | 0.235 | 0.186 | 0.093 |
| negated_natural | hard_tree | 0.706 | 0.765 | 1.471 | 0.765 | 0.176 | 0.235 | 0.235 |
| negated_natural | hybrid_tree | 0.882 | 1.000 | 2.059 | 0.814 | 0.235 | 0.186 | 0.093 |

## Checks

- gated_accuracy_ge_flat: True
- gated_contamination_lt_flat: True
- gated_ai_risk_lt_flat: True
- gated_hit_at_k_ge_hybrid: True
- gated_accuracy_ge_hard_minus_10pct: True
- final_pass: True